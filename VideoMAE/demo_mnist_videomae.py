"""
Moving MNIST / Fashion-MNIST → VideoMAE 直觉演示
=================================================
用 MNIST/FashionMNIST 生成 "Moving Digit" 视频，
然后用 VideoMAE 做 masked 重建，直观看到效果。

数据量: 10000+ 个视频 (从 60000 张图片生成)
帧数: 16 帧, 64×64
训练: 只需 50~100 epoch 就能看到明显效果

用法:
  python demo_mnist_videomae.py                    # MNIST (默认)
  python demo_mnist_videomae.py --fashion          # FashionMNIST
  python demo_mnist_videomae.py --epochs 100       # 训练100轮
  python demo_mnist_videomae.py --vis_only         # 只用已有checkpoint做可视化
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
from torch.utils.data import Dataset, DataLoader

# 复用已有的 VideoMAE 模型
sys.path.insert(0, os.path.dirname(__file__))
from run_pretrain_radar import VideoMAEPretrainModel


# ============================================================
#  1. Moving MNIST 数据集 (自动下载, 生成移动视频)
# ============================================================

class MovingMNISTDataset(Dataset):
    """
    从 MNIST/FashionMNIST 生成移动数字视频
    
    每个视频: 1~2 个数字在 canvas_size×canvas_size 的画布上随机运动
    输出: (1, T, H, W) + tube_mask
    
    10000+ 唯一视频 — 足够 VideoMAE 学习
    """
    
    def __init__(
        self,
        num_frames=16,
        canvas_size=64,
        num_digits=2,
        mask_ratio=0.9,
        patch_size=8,
        tubelet_size=2,
        split="train",
        fashion=False,
        data_root="./data_mnist",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.canvas_size = canvas_size
        self.num_digits = num_digits
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        
        # 下载 MNIST / FashionMNIST
        from torchvision import datasets
        DatasetClass = datasets.FashionMNIST if fashion else datasets.MNIST
        is_train = (split == "train")
        ds = DatasetClass(root=data_root, train=is_train, download=True)
        
        # 提取所有图片 (28×28 numpy)
        self.images = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
        self.labels = ds.targets.numpy()
        
        # 掩码参数
        n_spatial = (canvas_size // patch_size) ** 2
        n_temporal = num_frames // tubelet_size
        self.total_tokens = n_spatial * n_temporal
        self.num_spatial = n_spatial
        self.num_temporal = n_temporal
        
        name = "FashionMNIST" if fashion else "MNIST"
        print(f"[MovingMNIST] {name} {split}: {len(self.images)} images → "
              f"∞ moving videos ({num_digits} digits, {num_frames}f, {canvas_size}px)")
        print(f"  tokens={self.total_tokens}, mask={int(self.total_tokens*mask_ratio)}/{self.total_tokens}")
    
    def __len__(self):
        # 每个图片组合可以生成不同轨迹 → 等效无限数据
        # 但为了 epoch 概念, 设定一个合理的大小
        return len(self.images) * 2
    
    def _make_moving_video(self):
        """生成一段移动数字视频"""
        T = self.num_frames
        H = W = self.canvas_size
        digit_size = 28
        
        video = np.zeros((T, H, W), dtype=np.float32)
        
        for _ in range(self.num_digits):
            # 随机选一张数字
            idx = np.random.randint(len(self.images))
            digit = self.images[idx]
            
            # 随机起始位置和速度
            max_pos = H - digit_size
            x = np.random.randint(0, max(max_pos, 1))
            y = np.random.randint(0, max(max_pos, 1))
            vx = np.random.uniform(-3, 3)
            vy = np.random.uniform(-3, 3)
            
            for t in range(T):
                # 放置数字 (叠加, 取最大值)
                xi, yi = int(round(x)), int(round(y))
                xi = np.clip(xi, 0, max_pos)
                yi = np.clip(yi, 0, max_pos)
                
                region = video[t, yi:yi+digit_size, xi:xi+digit_size]
                video[t, yi:yi+digit_size, xi:xi+digit_size] = np.maximum(region, digit)
                
                # 移动 + 反弹
                x += vx
                y += vy
                if x <= 0 or x >= max_pos:
                    vx = -vx
                    x = np.clip(x, 0, max_pos)
                if y <= 0 or y >= max_pos:
                    vy = -vy
                    y = np.clip(y, 0, max_pos)
        
        return video  # (T, H, W) float [0,1]
    
    def __getitem__(self, idx):
        video = self._make_moving_video()  # (T, H, W)
        
        # (T, H, W) → (1, T, H, W)
        frames = torch.from_numpy(video).unsqueeze(0)
        
        # Tube mask
        spatial_mask = np.zeros(self.num_spatial, dtype=bool)
        n_mask = int(self.num_spatial * self.mask_ratio)
        spatial_mask[np.random.choice(self.num_spatial, n_mask, replace=False)] = True
        mask = np.tile(spatial_mask, self.num_temporal)
        
        return frames, mask


# ============================================================
#  2. 训练 + 可视化
# ============================================================

def save_image_grid(filename, frames_np, n_cols=8, scale=2):
    """保存帧网格为 PGM 图片 (可选放大)"""
    T, H, W = frames_np.shape
    if scale > 1:
        # 最近邻放大
        big = np.zeros((T, H*scale, W*scale), dtype=frames_np.dtype)
        for t in range(T):
            for dy in range(scale):
                for dx in range(scale):
                    big[t, dy::scale, dx::scale] = frames_np[t]
        frames_np = big
        H, W = H*scale, W*scale
    
    n_rows = math.ceil(T / n_cols)
    pad = 2
    grid_h = n_rows * (H + pad) + pad
    grid_w = n_cols * (W + pad) + pad
    grid = np.ones((grid_h, grid_w), dtype=np.float32) * 0.3
    
    for t in range(T):
        r, c = t // n_cols, t % n_cols
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        grid[y:y+H, x:x+W] = frames_np[t]
    
    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    h, w = grid.shape
    with open(filename, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(grid.tobytes())
    
    # 尝试转 PNG
    try:
        from PIL import Image
        Image.open(filename).save(filename.replace('.pgm', '.png'))
    except ImportError:
        pass


def reconstruct_video(model, video_np, mask_np, device, patch_size, tubelet_size):
    """用模型重建被遮挡区域, 返回重建帧"""
    model.eval()
    with torch.no_grad():
        video_t = torch.from_numpy(video_np).float().unsqueeze(0).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        
        vis = model.encoder(video_t, mask_t)
        pred = model.decoder(vis, mask_t)
        pred_np = pred[0].cpu().numpy()  # (N, C*t*p*p)
    
    # 还原为帧
    T, H, W = video_np.shape
    p = patch_size
    t = tubelet_size
    n_h, n_w = H // p, W // p
    
    result = video_np.copy()
    for i in range(len(mask_np)):
        if not mask_np[i]:
            continue
        t_idx = i // (n_h * n_w)
        s_idx = i % (n_h * n_w)
        h_idx, w_idx = s_idx // n_w, s_idx % n_w
        
        patch = pred_np[i].reshape(1, t, p, p)[0]  # (t, p, p)
        for dt in range(t):
            tt = t_idx * t + dt
            if tt < T:
                result[tt, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = patch[dt]
    
    return result


def visualize_mask_overlay(video_np, mask_np, patch_size, tubelet_size):
    """在被遮挡区域叠加红色标记 (返回灰度近似)"""
    T, H, W = video_np.shape
    result = video_np.copy()
    p = patch_size
    t = tubelet_size
    n_h, n_w = H // p, W // p
    
    for i, masked in enumerate(mask_np):
        if not masked:
            continue
        t_idx = i // (n_h * n_w)
        s_idx = i % (n_h * n_w)
        h_idx, w_idx = s_idx // n_w, s_idx % n_w
        for dt in range(t):
            tt = t_idx * t + dt
            if tt < T:
                result[tt, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 0.15  # 暗色表示遮挡
    return result


@torch.no_grad()
def validate(model, dataloader, device, fp16):
    model.eval()
    total, n = 0.0, 0
    for frames, mask in dataloader:
        frames = frames.to(device, dtype=torch.float32)
        mask = torch.from_numpy(np.stack(mask)).to(device) if isinstance(mask, list) else mask.to(device)
        with torch.amp.autocast('cuda', enabled=fp16 and device.type == 'cuda'):
            loss = model(frames, mask)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description='Moving MNIST VideoMAE Demo')
    parser.add_argument('--fashion', action='store_true', help='用 FashionMNIST')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--canvas_size', type=int, default=64, help='画布大小')
    parser.add_argument('--num_frames', type=int, default=16, help='视频帧数')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch 大小')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    parser.add_argument('--num_digits', type=int, default=2, help='每视频数字个数')
    parser.add_argument('--embed_dim', type=int, default=192, help='ViT-Tiny')
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--decoder_dim', type=int, default=96)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--no_fp16', action='store_true')
    parser.add_argument('--save_dir', type=str, default='mnist_videomae_output')
    parser.add_argument('--vis_only', action='store_true', help='只做可视化')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    
    if args.no_fp16:
        args.fp16 = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        args.fp16 = False
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ---- 打印配置 ----
    name = "FashionMNIST" if args.fashion else "MNIST"
    print("=" * 60)
    print(f"  Moving {name} → VideoMAE 直觉演示")
    print("=" * 60)
    print(f"  Device:    {device}" + 
          (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Video:     {args.num_digits} digits, {args.num_frames}f × {args.canvas_size}px")
    print(f"  Patch:     {args.patch_size}×{args.patch_size} × {args.tubelet_size}f")
    n_spatial = (args.canvas_size // args.patch_size) ** 2
    n_temporal = args.num_frames // args.tubelet_size
    n_total = n_spatial * n_temporal
    n_vis = int(n_total * (1 - args.mask_ratio))
    print(f"  Tokens:    {n_total} total, {n_vis} visible ({(1-args.mask_ratio)*100:.0f}%)")
    print(f"  Mask:      {args.mask_ratio*100:.0f}% tube masking")
    print(f"  Model:     ViT-Tiny (dim={args.embed_dim}, depth={args.depth})")
    print()
    
    # ---- 数据 ----
    train_ds = MovingMNISTDataset(
        num_frames=args.num_frames, canvas_size=args.canvas_size,
        num_digits=args.num_digits, mask_ratio=args.mask_ratio,
        patch_size=args.patch_size, tubelet_size=args.tubelet_size,
        split="train", fashion=args.fashion,
    )
    val_ds = MovingMNISTDataset(
        num_frames=args.num_frames, canvas_size=args.canvas_size,
        num_digits=args.num_digits, mask_ratio=args.mask_ratio,
        patch_size=args.patch_size, tubelet_size=args.tubelet_size,
        split="test", fashion=args.fashion,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    
    # ---- 模型 ----
    model = VideoMAEPretrainModel(
        img_size=args.canvas_size,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        in_chans=1,
        encoder_embed_dim=args.embed_dim,
        encoder_depth=args.depth,
        encoder_num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.num_heads,
    ).to(device)
    
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params:    {total_p/1e6:.1f}M")
    
    ckpt_path = os.path.join(args.save_dir, 'best.pt')
    
    # ---- 加载已有 checkpoint ----
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  ★ 加载 checkpoint: epoch {start_epoch}, loss={ckpt.get('loss', '?')}")
    
    if args.vis_only:
        print("\n  [只做可视化, 跳过训练]")
        do_visualization(model, val_ds, device, args)
        return
    
    # ---- 训练 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    warmup_epochs = max(5, args.epochs // 10)
    best_val = float('inf')
    
    print(f"\n  训练 {args.epochs} epochs (warmup={warmup_epochs})")
    print(f"  Batch: {args.batch_size}, steps/epoch: {len(train_dl)}")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # LR schedule
        if epoch <= warmup_epochs:
            lr = args.lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Train
        model.train()
        total_loss, n_batch = 0.0, 0
        t0 = time.time()
        
        for frames, mask in train_dl:
            frames = frames.to(device, dtype=torch.float32)
            mask = torch.from_numpy(np.stack(mask)).to(device) if isinstance(mask, list) else mask.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
                loss = model(frames, mask)
            
            if args.fp16 and device.type == 'cuda':
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
        
        train_loss = total_loss / max(n_batch, 1)
        elapsed = time.time() - t0
        
        # Val (每 5 个 epoch)
        val_str = ""
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            val_loss = validate(model, val_dl, device, args.fp16)
            val_str = f" | val={val_loss:.4f}"
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': val_loss,
                }, ckpt_path)
        
        print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f}{val_str}"
              f" | lr={lr:.1e} | {elapsed:.1f}s")
        
        # 显存 (仅第1个epoch)
        if device.type == 'cuda' and epoch == 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [显存] 峰值 {peak:.2f} GB")
        
        # 中间可视化 (每 10 个 epoch)
        if epoch % 10 == 0 or epoch == args.epochs:
            do_visualization(model, val_ds, device, args, suffix=f"_ep{epoch}")
    
    print("\n" + "=" * 60)
    print(f"  训练完成! best val_loss = {best_val:.4f}")
    print(f"  输出目录: {args.save_dir}/")
    print("=" * 60)
    
    # 最终可视化
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
    do_visualization(model, val_ds, device, args, suffix="_final")


def do_visualization(model, dataset, device, args, suffix=""):
    """生成可视化对比图"""
    model.eval()
    save_dir = args.save_dir
    
    # 用固定种子生成可复现的样本
    rng_state = np.random.get_state()
    np.random.seed(42)
    
    for sample_idx in range(3):
        video_np = dataset._make_moving_video()  # (T, H, W)
        
        # 固定 mask
        n_sp = dataset.num_spatial
        sp_mask = np.zeros(n_sp, dtype=bool)
        sp_mask[np.random.choice(n_sp, int(n_sp * args.mask_ratio), replace=False)] = True
        mask_np = np.tile(sp_mask, dataset.num_temporal)
        
        # 重建
        recon = reconstruct_video(model, video_np, mask_np, device,
                                  args.patch_size, args.tubelet_size)
        recon = np.clip(recon, 0, 1)
        
        # 遮挡可视化
        masked_vis = visualize_mask_overlay(video_np, mask_np, 
                                           args.patch_size, args.tubelet_size)
        
        # 保存
        prefix = f"s{sample_idx}{suffix}"
        save_image_grid(os.path.join(save_dir, f'{prefix}_1_original.pgm'), video_np, scale=2)
        save_image_grid(os.path.join(save_dir, f'{prefix}_2_masked.pgm'), masked_vis, scale=2)
        save_image_grid(os.path.join(save_dir, f'{prefix}_3_recon.pgm'), recon, scale=2)
        
        # 差异图
        diff = np.abs(recon - video_np)
        save_image_grid(os.path.join(save_dir, f'{prefix}_4_error.pgm'), 
                        np.clip(diff * 5, 0, 1), scale=2)
        
        mse = np.mean((recon - video_np) ** 2)
        # 只计算非空区域的 SSIM 近似
        mask_area = video_np > 0.05
        if mask_area.sum() > 0:
            content_mse = np.mean((recon[mask_area] - video_np[mask_area]) ** 2)
        else:
            content_mse = mse
        
        print(f"  样本{sample_idx}: MSE={mse:.4f}, 内容区MSE={content_mse:.4f}")
    
    np.random.set_state(rng_state)
    print(f"  可视化已保存到 {save_dir}/")
    print(f"    *_1_original  — 原始移动数字视频")
    print(f"    *_2_masked    — 遮挡90%后的视图 (暗色=遮挡)")
    print(f"    *_3_recon     — 模型重建结果")
    print(f"    *_4_error     — 重建误差 (越黑越准)")


if __name__ == '__main__':
    main()
