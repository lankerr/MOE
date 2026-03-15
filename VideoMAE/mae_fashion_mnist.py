"""
Fashion-MNIST 2D Masked Autoencoder (MAE) — 轻量自包含实现
============================================================
核心验证: mask 75% patches → encode 到浅空间 → decode 重建全图

架构参考: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
代码参考: https://github.com/facebookresearch/mae (简化版，自包含)

用法:
  python mae_fashion_mnist.py                     # 默认训练 50 epoch
  python mae_fashion_mnist.py --epochs 100        # 训练 100 epoch
  python mae_fashion_mnist.py --mask_ratio 0.5    # 50% 掩码
  python mae_fashion_mnist.py --mask_ratio 0.9    # 90% 掩码（极端）
  python mae_fashion_mnist.py --vis_only           # 只做可视化
  python mae_fashion_mnist.py --latent_analysis    # 分析隐空间

特点:
  - 自包含: 不依赖外部 MAE 仓库
  - Fashion-MNIST 28×28 → 4×4 patch → 49 tokens
  - ViT-Tiny 级别: embed=128, depth=6, heads=4 → ~1.5M 参数
  - CPU 也能跑 (batch=64, ~2 min/epoch)
  - GPU 极快 (batch=256, ~5 sec/epoch)
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
#  1. ViT 基础组件 (最小实现)
# ============================================================

class PatchEmbed2D(nn.Module):
    """2D Patch Embedding: (B, 1, H, W) → (B, N, embed_dim)"""
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)           # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
#  2. MAE 模型 (核心)
# ============================================================

class MAE(nn.Module):
    """
    Masked Autoencoder (He et al., CVPR 2022) — 2D 轻量版

    工作流程:
    1. 图像 → patch embedding → 49 tokens
    2. 随机 mask 75% → 只保留 25% 可见 tokens
    3. Encoder(可见 tokens) → 隐空间表示
    4. 插入 [MASK] tokens → Decoder 重建全部 patches
    5. Loss = MSE(预测像素, 真实像素) — 只在被 mask 的 patches 上计算
    """

    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_chans=1,
        # encoder
        embed_dim=128,
        depth=6,
        num_heads=4,
        # decoder
        decoder_embed_dim=64,
        decoder_depth=3,
        decoder_num_heads=4,
        # masking
        mask_ratio=0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        pixel_per_patch = patch_size * patch_size * in_chans

        # ---- Encoder ----
        self.patch_embed = PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                       requires_grad=False)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Decoder ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, pixel_per_patch)

        self._init_weights()

    def _init_weights(self):
        # 正弦位置编码
        pos = self._sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls=True)
        self.pos_embed.data.copy_(pos)
        dec_pos = self._sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                          self.num_patches, cls=True)
        self.decoder_pos_embed.data.copy_(dec_pos)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # 其他层
        self.apply(self._init_layer)

    @staticmethod
    def _init_layer(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _sincos_pos_embed(embed_dim, num_patches, cls=True):
        """2D 正弦余弦位置编码"""
        grid_size = int(num_patches ** 0.5)
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=-1)
        grid = grid.reshape(-1, 2)

        half = embed_dim // 4
        omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
        out_h = grid[:, 0:1] * omega.unsqueeze(0)
        out_w = grid[:, 1:2] * omega.unsqueeze(0)
        pe = torch.cat([torch.sin(out_h), torch.cos(out_h),
                         torch.sin(out_w), torch.cos(out_w)], dim=-1)

        # 截断或填充到 embed_dim
        if pe.shape[-1] < embed_dim:
            pe = F.pad(pe, (0, embed_dim - pe.shape[-1]))
        else:
            pe = pe[:, :embed_dim]

        if cls:
            pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
        return pe.unsqueeze(0)

    # ---- Masking ----
    def random_masking(self, x, mask_ratio):
        """
        x: (B, N, D)
        返回: x_visible, mask, ids_restore
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - mask_ratio))

        # 随机排列
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 取前 n_keep 个
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, dim=1,
                                  index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # 二值 mask: 1=masked, 0=visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    # ---- Forward ----
    def forward_encoder(self, x, mask_ratio):
        """
        x: (B, 1, H, W)
        返回: latent, mask, ids_restore
        """
        # patch embed
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]  # 不加 CLS 的位置编码

        # mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append CLS
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N_vis, D)

        # Transformer
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        x: (B, 1+N_vis, enc_dim) — encoder 输出
        ids_restore: (B, N) — 用于恢复原始顺序
        返回: (B, N, patch_pixels)
        """
        x = self.decoder_embed(x)  # (B, 1+N_vis, dec_dim)

        # 追加 mask tokens
        B = x.shape[0]
        n_mask = self.num_patches + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 去掉CLS，追加mask

        # 恢复原始顺序
        x_ = torch.gather(x_, dim=1,
                           index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))

        # 重新加上 CLS
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # 加解码器位置编码
        x = x + self.decoder_pos_embed

        # Transformer decoder
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 预测像素
        x = self.decoder_pred(x)

        # 去掉 CLS
        x = x[:, 1:, :]
        return x

    def patchify(self, imgs):
        """(B, 1, H, W) → (B, N, patch_size^2)"""
        p = self.patch_size
        h = w = self.img_size // p
        x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(imgs.shape[0], h * w, p * p)
        return x

    def unpatchify(self, x):
        """(B, N, patch_size^2) → (B, 1, H, W)"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p)
        x = x.permute(0, 1, 3, 2, 4).reshape(x.shape[0], 1, h * p, w * p)
        return x

    def forward(self, imgs, mask_ratio=None):
        """
        完整前向: 图像 → mask → encode → decode → loss
        返回: loss, pred, mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # (B, N, p*p)

        target = self.patchify(imgs)

        # loss 只在 masked patches 上计算
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)   # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # 只算 masked

        return loss, pred, mask

    @torch.no_grad()
    def reconstruct(self, imgs, mask_ratio=None):
        """重建: 可见 patch 保留原值，masked patch 用预测值填充"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)

        target = self.patchify(imgs)
        # 可见区域用原始值，masked 区域用预测值
        recon = target * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
        recon_img = self.unpatchify(recon)
        return recon_img, mask

    @torch.no_grad()
    def encode(self, imgs):
        """编码到隐空间 (不做mask): 用于下游任务或隐空间分析"""
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x[:, 0]  # 返回 CLS token 作为全局表示


# ============================================================
#  3. 训练
# ============================================================

def get_fashion_mnist(data_root="./data"):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], (1,28,28)
    ])
    train_ds = datasets.FashionMNIST(root=data_root, train=True,
                                      download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_root, train=False,
                                     download=True, transform=transform)
    return train_ds, test_ds

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 55)
    print("  Fashion-MNIST MAE — 掩码自编码器实验")
    print("=" * 55)
    print(f"  Device:     {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Image:      28×28 → {args.patch_size}×{args.patch_size} patches "
          f"→ {(28//args.patch_size)**2} tokens")
    print(f"  Mask ratio: {args.mask_ratio*100:.0f}% "
          f"({int((28//args.patch_size)**2 * args.mask_ratio)} masked, "
          f"{int((28//args.patch_size)**2 * (1-args.mask_ratio))} visible)")
    print(f"  Encoder:    dim={args.embed_dim}, depth={args.depth}, heads={args.num_heads}")
    print(f"  Decoder:    dim={args.dec_dim}, depth={args.dec_depth}")
    print()

    # data
    train_ds, test_ds = get_fashion_mnist(args.data_root)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    # model
    model = MAE(
        img_size=28, patch_size=args.patch_size, in_chans=1,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.dec_dim, decoder_depth=args.dec_depth,
        decoder_num_heads=args.num_heads, mask_ratio=args.mask_ratio,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    enc_p = sum(p.numel() for n, p in model.named_parameters()
                if 'decoder' not in n and 'mask_token' not in n)
    dec_p = total_p - enc_p
    print(f"  参数量:     总={total_p/1e6:.2f}M  Enc={enc_p/1e6:.2f}M  Dec={dec_p/1e6:.2f}M")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.05)
    use_fp16 = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_loss = float('inf')
    ckpt_path = os.path.join(args.save_dir, 'best_mae.pt')

    # 加载 checkpoint
    start_epoch = 0
    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('loss', float('inf'))
        print(f"  ★ 加载 checkpoint: epoch {start_epoch}, loss={best_loss:.5f}")

    warmup_epochs = max(3, args.epochs // 10)
    print(f"\n  训练 {args.epochs} epochs (warmup={warmup_epochs})")
    print("-" * 55)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # LR schedule: warmup + cosine
        if epoch <= warmup_epochs:
            lr = args.lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        total_loss, n_batch = 0., 0
        t0 = time.time()

        for imgs, _ in train_dl:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_fp16):
                loss, _, _ = model(imgs)
            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        train_loss = total_loss / n_batch
        elapsed = time.time() - t0

        # Validate
        val_str = ""
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            model.eval()
            v_loss, v_n = 0., 0
            with torch.no_grad():
                for imgs, _ in test_dl:
                    imgs = imgs.to(device)
                    with torch.amp.autocast('cuda', enabled=use_fp16):
                        loss, _, _ = model(imgs)
                    v_loss += loss.item()
                    v_n += 1
            val_loss = v_loss / v_n
            val_str = f" | val={val_loss:.5f}"

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                             'loss': val_loss}, ckpt_path)

        # Emit
        print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.5f}{val_str}"
              f"  lr={lr:.1e}  {elapsed:.1f}s")

        # 显存 (第1个epoch)
        if device.type == 'cuda' and epoch == start_epoch + 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [显存] 峰值 {peak:.3f} GB")

        # 中间可视化
        if epoch % 10 == 0 or epoch == args.epochs:
            visualize(model, test_ds, device, args, suffix=f"_ep{epoch}")

    print(f"\n  训练完成! best val_loss = {best_loss:.5f}")
    print(f"  checkpoint: {ckpt_path}")

    # 最终可视化
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
    visualize(model, test_ds, device, args, suffix="_final")

    # 隐空间分析
    if args.latent_analysis:
        analyze_latent(model, test_ds, device, args)


# ============================================================
#  4. 可视化
# ============================================================

def visualize(model, dataset, device, args, suffix="", n_samples=8):
    """生成掩码→重建对比图"""
    model.eval()
    save_dir = args.save_dir

    # 取固定样本
    torch.manual_seed(42)
    indices = list(range(n_samples))
    imgs = torch.stack([dataset[i][0] for i in indices]).to(device)  # (N, 1, 28, 28)

    with torch.no_grad():
        recon_imgs, masks = model.reconstruct(imgs, args.mask_ratio)
        recon_imgs = recon_imgs.clamp(0, 1)

    # 转 numpy
    orig = imgs.cpu().numpy()[:, 0]       # (N, 28, 28)
    recon = recon_imgs.cpu().numpy()[:, 0]  # (N, 28, 28)
    masks_np = masks.cpu().numpy()          # (N, num_patches)

    # 生成 masked 可视化
    p = args.patch_size
    h = w = 28 // p
    masked_vis = orig.copy()
    for b in range(n_samples):
        for i in range(h * w):
            if masks_np[b, i] > 0.5:  # masked
                r, c = i // w, i % w
                masked_vis[b, r*p:(r+1)*p, c*p:(c+1)*p] = 0.15

    # 保存为 PNG (需要 PIL) 或 PGM
    try:
        from PIL import Image
        # 拼接: 原图 / masked / 重建 / 差异
        scale = 4
        rows = []
        for b in range(n_samples):
            row = np.concatenate([orig[b], masked_vis[b], recon[b],
                                   np.clip(np.abs(recon[b]-orig[b])*3, 0, 1)], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)  # (N*28, 4*28)
        grid = np.clip(grid * 255, 0, 255).astype(np.uint8)

        # 放大
        img = Image.fromarray(grid, mode='L')
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        path = os.path.join(save_dir, f"mae_recon{suffix}.png")
        img.save(path)
        print(f"  [可视化] {path}")
    except ImportError:
        # 无 PIL 则用 PGM
        grid_flat = np.concatenate([
            np.concatenate([orig[b], masked_vis[b], recon[b]], axis=1) for b in range(n_samples)
        ], axis=0)
        grid_flat = np.clip(grid_flat * 255, 0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f"mae_recon{suffix}.pgm")
        h_g, w_g = grid_flat.shape
        with open(path, 'wb') as f:
            f.write(f'P5\n{w_g} {h_g}\n255\n'.encode())
            f.write(grid_flat.tobytes())
        print(f"  [可视化] {path}")


# ============================================================
#  5. 隐空间分析
# ============================================================

def analyze_latent(model, dataset, device, args):
    """分析隐空间: t-SNE / 类别聚类"""
    print("\n  [隐空间分析]")
    model.eval()

    # 编码全部测试集
    dl = DataLoader(dataset, batch_size=256, shuffle=False,
                    num_workers=args.num_workers)
    all_z, all_y = [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device)
            z = model.encode(imgs)  # (B, embed_dim) CLS token
            all_z.append(z.cpu())
            all_y.append(labels)

    Z = torch.cat(all_z, dim=0).numpy()   # (10000, 128)
    Y = torch.cat(all_y, dim=0).numpy()   # (10000,)

    print(f"  隐空间维度: {Z.shape}")
    print(f"  类别分布: {np.bincount(Y)}")

    # 类内/类间距离
    from collections import defaultdict
    class_centers = {}
    for c in range(10):
        class_centers[c] = Z[Y == c].mean(axis=0)

    intra_dists = []
    for c in range(10):
        dists = np.linalg.norm(Z[Y == c] - class_centers[c], axis=1)
        intra_dists.append(dists.mean())
    inter_dists = []
    for i in range(10):
        for j in range(i+1, 10):
            inter_dists.append(np.linalg.norm(class_centers[i] - class_centers[j]))

    print(f"  类内平均距离: {np.mean(intra_dists):.3f}")
    print(f"  类间平均距离: {np.mean(inter_dists):.3f}")
    print(f"  分离度 (inter/intra): {np.mean(inter_dists)/np.mean(intra_dists):.3f}")

    # t-SNE 可视化
    try:
        from sklearn.manifold import TSNE
        from PIL import Image

        print("  计算 t-SNE (2000 样本)...")
        n_vis = 2000
        idx = np.random.RandomState(42).choice(len(Z), n_vis, replace=False)
        Z_sub = Z[idx]
        Y_sub = Y[idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        Z_2d = tsne.fit_transform(Z_sub)

        # 保存为简单的散点图 (纯 PIL, 不依赖 matplotlib)
        W, H = 600, 600
        margin = 40
        img = Image.new('RGB', (W, H), 'white')
        pixels = img.load()

        z_min = Z_2d.min(axis=0)
        z_max = Z_2d.max(axis=0)
        z_range = z_max - z_min + 1e-8

        # 10 种颜色
        colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207),
        ]

        for i in range(n_vis):
            px = int(margin + (Z_2d[i, 0] - z_min[0]) / z_range[0] * (W - 2*margin))
            py = int(margin + (Z_2d[i, 1] - z_min[1]) / z_range[1] * (H - 2*margin))
            c = colors[Y_sub[i]]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x_ = min(max(px + dx, 0), W - 1)
                    y_ = min(max(py + dy, 0), H - 1)
                    pixels[x_, y_] = c

        path = os.path.join(args.save_dir, "latent_tsne.png")
        img.save(path)
        print(f"  t-SNE 保存: {path}")
        print(f"  颜色图例: " + ", ".join(f"{i}={CLASS_NAMES[i]}" for i in range(10)))
    except ImportError as e:
        print(f"  跳过 t-SNE (缺少依赖: {e})")

    # 保存隐空间数据
    np.savez(os.path.join(args.save_dir, "latent_data.npz"),
             Z=Z, Y=Y, class_names=CLASS_NAMES)
    print(f"  隐空间数据保存: latent_data.npz")


# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Fashion-MNIST MAE")
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1.5e-3)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--dec_dim', type=int, default=64)
    p.add_argument('--dec_depth', type=int, default=3)
    p.add_argument('--save_dir', type=str, default='mae_fashion_output')
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--vis_only', action='store_true')
    p.add_argument('--latent_analysis', action='store_true')
    p.add_argument('--fresh', action='store_true', help='从头训练，忽略已有checkpoint')
    args = p.parse_args()

    if args.vis_only:
        # 只做可视化（不训练）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(args.save_dir, exist_ok=True)
        model = MAE(28, args.patch_size, 1,
                     args.embed_dim, args.depth, args.num_heads,
                     args.dec_dim, args.dec_depth, args.num_heads,
                     args.mask_ratio).to(device)
        ckpt_path = os.path.join(args.save_dir, 'best_mae.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            print(f"加载权重: epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?')}")
        _, test_ds = get_fashion_mnist(args.data_root)
        visualize(model, test_ds, device, args, suffix="_vis")
        if args.latent_analysis:
            analyze_latent(model, test_ds, device, args)
        return

    train(args)


if __name__ == "__main__":
    main()
