"""
VideoMAE ViT-Tiny Radar Pretraining — 自包含版
=================================================
完全独立实现，不需要克隆 VideoMAE 官方仓库。
包含: ViT-Tiny Encoder + 轻量 Decoder + Tube Masking + 训练循环

用法:
  # CPU 调试 (RTX 3050Ti 或无GPU)
  python run_pretrain_radar.py --debug

  # GPU 训练 (RTX 5070 8.55GB)
  python run_pretrain_radar.py

  # 指定数据路径
  python run_pretrain_radar.py --data_root X:\\datasets\\sevir
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
from torch.utils.data import DataLoader, Dataset
from functools import partial

# ============================================================
#  1. ViT 基础组件
# ============================================================

def get_sinusoid_encoding(n_position, d_hid):
    """正弦位置编码"""
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    table = torch.zeros(n_position, d_hid)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table.unsqueeze(0)  # (1, n_position, d_hid)


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding: (B, C, T, H, W) → (B, N, embed_dim)"""
    def __init__(self, img_size=128, patch_size=16, tubelet_size=2,
                 in_chans=1, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        num_spatial = (img_size // patch_size) ** 2
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.num_patches_spatial = num_spatial
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
#  2. VideoMAE 预训练模型 (ViT-Tiny)
# ============================================================

class VideoMAEEncoder(nn.Module):
    """VideoMAE Encoder — 只处理可见 token"""
    def __init__(self, img_size=128, patch_size=16, tubelet_size=2,
                 in_chans=1, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        # 动态计算最大 token 数，留足余量 (384x384 + 20帧 = 5760)
        max_tokens = max(8192, (img_size // patch_size) ** 2 * 64)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        
        self._init_pos_embed(max_tokens, embed_dim)
    
    def _init_pos_embed(self, max_len, dim):
        """初始化正弦位置编码"""
        pe = get_sinusoid_encoding(max_len, dim)
        self.pos_embed.data.copy_(pe)
    
    def forward(self, x, mask):
        """
        x: (B, C, T, H, W)
        mask: (B, N_total) bool — True=masked
        Returns: (B, N_visible, embed_dim)
        """
        # Patch embedding
        tokens = self.patch_embed(x)  # (B, N, D)
        B, N, D = tokens.shape
        
        # 加位置编码
        tokens = tokens + self.pos_embed[:, :N, :]
        
        # 只保留可见 token (mask=False 的位置)
        visible_mask = ~mask  # True=visible
        # 收集可见 token
        visible_tokens = []
        for b in range(B):
            visible_tokens.append(tokens[b, visible_mask[b]])
        
        # Pad to same length (所有样本的可见数量应相同, tube mask保证)
        x = torch.stack(visible_tokens, dim=0)  # (B, N_vis, D)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x


class VideoMAEDecoder(nn.Module):
    """VideoMAE Decoder — 重建被遮挡的 token"""
    def __init__(self, embed_dim=192, decoder_embed_dim=96, decoder_depth=4,
                 decoder_num_heads=3, mlp_ratio=4., num_classes=512, patch_size=16):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        max_tokens = 8192  # 足够覆盖 384×384 + 32帧
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, decoder_embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        # 预测原始像素: in_chans * tubelet_size * patch_size^2
        self.head = nn.Linear(decoder_embed_dim, num_classes)
        
        pe = get_sinusoid_encoding(max_tokens, decoder_embed_dim)
        self.pos_embed.data.copy_(pe)
    
    def forward(self, visible_tokens, mask):
        """
        visible_tokens: (B, N_vis, encoder_embed_dim)
        mask: (B, N_total) bool — True=masked
        Returns: (B, N_total, num_classes) 重建的像素值
        """
        B = visible_tokens.shape[0]
        N_total = mask.shape[1]
        
        # 线性投影
        vis = self.decoder_embed(visible_tokens)  # (B, N_vis, dec_dim)
        
        # 构建完整序列: 可见 + mask_token
        dec_dim = vis.shape[-1]
        full_tokens = torch.zeros(B, N_total, dec_dim, device=vis.device, dtype=vis.dtype)
        
        for b in range(B):
            visible_idx = (~mask[b]).nonzero(as_tuple=True)[0]
            masked_idx = mask[b].nonzero(as_tuple=True)[0]
            full_tokens[b, visible_idx] = vis[b]
            full_tokens[b, masked_idx] = self.mask_token.squeeze(0).to(dtype=vis.dtype)
        
        # 加位置编码
        full_tokens = full_tokens + self.pos_embed[:, :N_total, :]
        
        # Decoder blocks
        for blk in self.blocks:
            full_tokens = blk(full_tokens)
        
        full_tokens = self.norm(full_tokens)
        pred = self.head(full_tokens)  # (B, N_total, num_classes)
        return pred


class VideoMAEPretrainModel(nn.Module):
    """
    完整的 VideoMAE 预训练模型
    encoder(可见tokens) → decoder(重建全部tokens) → MSE loss
    """
    def __init__(self, img_size=128, patch_size=16, tubelet_size=2,
                 in_chans=1,
                 # Encoder (ViT-Tiny)
                 encoder_embed_dim=192, encoder_depth=12, encoder_num_heads=3,
                 # Decoder
                 decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4.):
        super().__init__()
        
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        
        # 像素重建目标维度
        num_classes = in_chans * tubelet_size * patch_size * patch_size
        
        self.encoder = VideoMAEEncoder(
            img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
        )
        self.decoder = VideoMAEDecoder(
            embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, num_classes=num_classes, patch_size=patch_size,
        )
    
    def patchify(self, video):
        """
        将视频切成 patch 作为重建目标
        video: (B, C, T, H, W)
        Returns: (B, N, patch_pixels) — patch_pixels = C * t * p * p
        """
        B, C, T, H, W = video.shape
        p = self.patch_size
        t = self.tubelet_size
        
        # (B, C, T/t, t, H/p, p, W/p, p)
        x = video.reshape(B, C, T // t, t, H // p, p, W // p, p)
        # (B, T/t, H/p, W/p, C, t, p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # (B, N, C*t*p*p)
        x = x.reshape(B, -1, C * t * p * p)
        return x
    
    def forward(self, video, mask):
        """
        video: (B, C, T, H, W) float32
        mask: (B, N) bool — True=masked
        Returns: loss (scalar)
        """
        # Encoder: 只看可见 token
        vis_tokens = self.encoder(video, mask)
        
        # Decoder: 重建所有 token
        pred = self.decoder(vis_tokens, mask)  # (B, N, C*t*p*p)
        
        # 目标: 原始像素的 patchified 版本
        target = self.patchify(video)  # (B, N, C*t*p*p)
        
        # ★ Patch Normalization (VideoMAE 标准做法)
        # 每个 patch 归一化到零均值单位方差，防止模型学到 trivial solution（全预测零）
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        
        # 只在 masked token 上计算 loss
        loss = (pred - target) ** 2  # (B, N, D)
        loss = loss.mean(dim=-1)     # (B, N)
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        return loss
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
#  3. 数据集 (支持 SEVIR 或 Dummy)
# ============================================================

class DummyRadarDataset(Dataset):
    """假数据, 用于无 SEVIR 时调试"""
    def __init__(self, num_samples=100, num_frames=8, input_size=128,
                 mask_ratio=0.9, tubelet_size=2, patch_size=16):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.num_spatial = (input_size // patch_size) ** 2
        self.num_temporal = num_frames // tubelet_size
        self.total_tokens = self.num_spatial * self.num_temporal
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        frames = torch.rand(1, self.num_frames, self.input_size, self.input_size)
        # Tube mask
        spatial_mask = np.zeros(self.num_spatial, dtype=bool)
        n_mask = int(self.num_spatial * self.mask_ratio)
        spatial_mask[np.random.choice(self.num_spatial, n_mask, replace=False)] = True
        mask = np.tile(spatial_mask, self.num_temporal)
        return frames, mask


def build_dataset(args, split="train"):
    """构建数据集: 优先用 SEVIR, 不存在则用 Dummy"""
    sevir_root = args.data_root
    catalog = os.path.join(sevir_root, "CATALOG.csv")
    
    if os.path.exists(catalog) and not args.debug:
        print(f"[数据] 使用 SEVIR ({split}): {sevir_root}")
        from radar_dataset import SEVIRVideoMAEDataset
        return SEVIRVideoMAEDataset(
            sevir_root=sevir_root,
            num_frames=args.num_frames,
            input_size=args.input_size,
            mask_ratio=args.mask_ratio,
            tubelet_size=args.tubelet_size,
            patch_size=args.patch_size,
            split=split,
            max_samples=args.max_samples,
            augment=(split == "train"),
        )
    else:
        n = 50 if args.debug else 500
        print(f"[数据] 使用 DummyRadarDataset ({n} samples)")
        return DummyRadarDataset(
            num_samples=n,
            num_frames=args.num_frames,
            input_size=args.input_size,
            mask_ratio=args.mask_ratio,
            tubelet_size=args.tubelet_size,
            patch_size=args.patch_size,
        )


# ============================================================
#  4. 训练循环
# ============================================================

@torch.no_grad()
def validate(model, dataloader, device, args):
    """验证集评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for frames, mask in dataloader:
        frames = frames.to(device, dtype=torch.float32)
        mask = torch.from_numpy(np.stack(mask)).to(device) if isinstance(mask, list) else mask.to(device)
        with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
            loss = model(frames, mask)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    num_batches = 0
    accum_steps = args.gradient_accumulation
    optimizer.zero_grad()
    
    t0 = time.time()
    for step, (frames, mask) in enumerate(dataloader):
        frames = frames.to(device, dtype=torch.float32)
        mask = torch.from_numpy(np.stack(mask)).to(device) if isinstance(mask, list) else mask.to(device)
        
        # Mixed precision
        with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
            loss = model(frames, mask)
            loss = loss / accum_steps
        
        if args.fp16 and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % accum_steps == 0:
            if args.fp16 and device.type == 'cuda':
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        num_batches += 1
        
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{step}/{len(dataloader)}] loss={loss.item()*accum_steps:.4f}  "
                  f"elapsed={elapsed:.1f}s", end='\r')
    
    avg_loss = total_loss / max(num_batches, 1)
    elapsed = time.time() - t0
    return avg_loss, elapsed


def main():
    parser = argparse.ArgumentParser(description='VideoMAE ViT-Tiny Radar Pretraining')
    
    # 数据
    parser.add_argument('--data_root', type=str, default=r'X:\datasets\sevir',
                        help='SEVIR 数据根目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制样本数 (调试用)')
    
    # 模型
    parser.add_argument('--input_size', type=int, default=192,
                        help='空间分辨率 (从384下采样, 建议192+)')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='输入帧数')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--in_chans', type=int, default=1,
                        help='输入通道 (1=VIL)')
    parser.add_argument('--mask_ratio', type=float, default=0.9,
                        help='掩码比例')
    
    # ViT-Small (默认) — 22M 参数, NeurIPS 2022 标配
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=192)
    parser.add_argument('--decoder_depth', type=int, default=4)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=40,
                        help='Warmup epochs (建议总epoch的5%%左右)')
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help='梯度累积步数 (等效batch = batch_size * accum)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='混合精度训练')
    parser.add_argument('--no_fp16', action='store_true')
    
    # 杂项
    parser.add_argument('--debug', action='store_true',
                        help='CPU 调试模式 (DummyData, 2 epochs)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_videomae')
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.no_fp16:
        args.fp16 = False
    if args.debug:
        args.fp16 = False
        args.epochs = 2
        args.batch_size = 2
        args.max_samples = 20
    
    # ============================
    # Device
    # ============================
    if args.debug:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        args.fp16 = False
    
    # 自动检测模型大小
    model_name = 'ViT-Small' if args.embed_dim >= 384 else 'ViT-Tiny'
    print("=" * 60)
    print(f"VideoMAE {model_name} Radar Pretraining")
    print("=" * 60)
    print(f"Device:     {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU:        {gpu_name} ({gpu_mem:.2f} GB)")
    print(f"Input:      ({args.in_chans}, {args.num_frames}, {args.input_size}, {args.input_size})")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Batch:      {args.batch_size} × accum {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"FP16:       {args.fp16}")
    
    # ============================
    # 模型
    # ============================
    model = VideoMAEPretrainModel(
        img_size=args.input_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        in_chans=args.in_chans,
        encoder_embed_dim=args.embed_dim,
        encoder_depth=args.depth,
        encoder_num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.num_heads,
    ).to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Model:      {model_name} (embed={args.embed_dim}, depth={args.depth}, heads={args.num_heads})")
    print(f"Params:     {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    
    # Token 信息
    n_spatial = (args.input_size // args.patch_size) ** 2
    n_temporal = args.num_frames // args.tubelet_size
    n_total = n_spatial * n_temporal
    n_visible = int(n_total * (1 - args.mask_ratio))
    print(f"Tokens:     {n_total} total → {n_visible} visible ({(1-args.mask_ratio)*100:.0f}%)")
    print(f"Attention:  {n_visible}×{n_visible} matrix")
    
    # ============================
    # 数据
    # ============================
    dataset = build_dataset(args, split="train")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    print(f"Train set:  {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # 验证集 (检测过拟合)
    val_dataset = build_dataset(args, split="val")
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), drop_last=False,
    )
    print(f"Val set:    {len(val_dataset)} samples, {len(val_dataloader)} batches/epoch")
    
    # ============================
    # 优化器
    # ============================
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    # ============================
    # 训练
    # ============================
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nSave dir:   {args.save_dir}")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Warmup + cosine lr
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        avg_loss, elapsed = train_one_epoch(
            model, dataloader, optimizer, scaler, device, epoch, args
        )
        
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"lr={lr:.2e} | time={elapsed:.1f}s")
        
        # 验证集评估 (每 5 个 epoch 或最后一个 epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            val_loss = validate(model, val_dataloader, device, args)
            print(f"  [验证] val_loss={val_loss:.4f}  "
                  f"{'⚠️ 过拟合!' if val_loss > avg_loss * 1.5 else '✓'}")
        
        # 显存监控
        if device.type == 'cuda' and epoch == 1:
            mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
            mem_rsv = torch.cuda.max_memory_reserved() / 1024**3
            print(f"  [显存] 峰值: allocated={mem_alloc:.2f}GB, reserved={mem_rsv:.2f}GB")
            if mem_alloc > gpu_mem * 0.9:
                print(f"  ⚠️ 显存占用过高! 建议降低 --input_size 或 --batch_size")
        
        # 保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best.pt'))
        
        # 定期保存
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.save_dir, f'epoch_{epoch}.pt'))
    
    print("\n" + "=" * 60)
    print(f"训练完成! Best loss = {best_loss:.4f}")
    print(f"Checkpoint: {args.save_dir}/best.pt")
    print("=" * 60)
    
    # 最终保存
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'loss': best_loss,
        'args': vars(args),
    }, os.path.join(args.save_dir, 'final.pt'))
    print(f"Encoder 权重已保存: {args.save_dir}/final.pt")
    print("可用 model.encoder.state_dict() 提取特征提取器权重")


if __name__ == '__main__':
    main()
