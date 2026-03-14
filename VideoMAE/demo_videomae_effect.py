"""
VideoMAE 效果演示
=================
直观展示 VideoMAE 做了什么:
  1. 输入一段视频帧序列
  2. 随机遮挡 90% 的 patch (tube masking)
  3. 模型从剩余 10% 预测/重建被遮挡的部分
  4. 保存可视化图片: 原始 → 遮挡 → 重建

这就是自监督预训练: 模型被迫学习"理解视频的时空结构"才能重建。
预训练完成后, Encoder 提取的特征可以用于下游任务(预测、分类等)。

JEPA vs MAE 的区别:
  - MAE (本脚本): 在像素空间重建 → 直接看到重建效果
  - JEPA: 在特征空间预测 → 不重建像素, 但学到更好的表示
  两者的 Encoder 结构完全一样, 区别在于训练目标。

用法:
  conda activate rtx3050ti_cu128
  python demo_videomae_effect.py
  python demo_videomae_effect.py --use_sevir  # 用真实 SEVIR 雷达数据
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import gc

sys.path.insert(0, '.')
from run_pretrain_radar import VideoMAEPretrainModel

# ============================================================
#  可视化工具 (纯 numpy, 无需 matplotlib)
# ============================================================

def save_pgm(filename, img_2d, width=None):
    """保存灰度图为 PGM 格式 (任何系统都能打开)"""
    h, w = img_2d.shape
    img = np.clip(img_2d * 255, 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())


def save_ppm(filename, img_rgb):
    """保存 RGB 图为 PPM 格式"""
    h, w, c = img_rgb.shape
    img = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    with open(filename, 'wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())


def frames_to_grid(frames, n_cols=8, pad=2):
    """
    将多帧排成网格
    frames: (T, H, W) numpy array, range [0, 1]
    Returns: (grid_H, grid_W) 灰度图
    """
    T, H, W = frames.shape
    n_rows = math.ceil(T / n_cols)
    grid_h = n_rows * (H + pad) + pad
    grid_w = n_cols * (W + pad) + pad
    grid = np.ones((grid_h, grid_w)) * 0.3  # 灰色背景

    for t in range(T):
        r = t // n_cols
        c = t % n_cols
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        grid[y:y+H, x:x+W] = frames[t]

    return grid


def visualize_mask(frames, mask, patch_size=16, tubelet_size=2):
    """
    将 mask 叠加到帧上 (被 mask 的区域变红)
    frames: (T, H, W) numpy [0,1]
    mask: (N_total,) bool — True=masked
    Returns: (T, H, W, 3) RGB
    """
    T, H, W = frames.shape
    n_h = H // patch_size
    n_w = W // patch_size
    n_t = T // tubelet_size

    result = np.stack([frames, frames, frames], axis=-1)  # (T, H, W, 3)

    for i, is_masked in enumerate(mask):
        t_idx = i // (n_h * n_w)
        spatial_idx = i % (n_h * n_w)
        h_idx = spatial_idx // n_w
        w_idx = spatial_idx % n_w

        t_start = t_idx * tubelet_size
        t_end = t_start + tubelet_size
        h_start = h_idx * patch_size
        h_end = h_start + patch_size
        w_start = w_idx * patch_size
        w_end = w_start + patch_size

        if is_masked:
            # 被遮挡 → 半透明红色
            for t in range(t_start, min(t_end, T)):
                result[t, h_start:h_end, w_start:w_end, 0] = 0.7  # R
                result[t, h_start:h_end, w_start:w_end, 1] *= 0.3
                result[t, h_start:h_end, w_start:w_end, 2] *= 0.3

    return result


def reconstruct_from_pred(pred, mask, original, patch_size=16, tubelet_size=2, in_chans=1):
    """
    将模型的 patch 预测还原为完整帧
    pred: (N_total, C*t*p*p) numpy
    mask: (N_total,) bool
    original: (T, H, W) numpy
    Returns: (T, H, W) numpy — 可见部分用原图, masked 部分用重建
    """
    T, H, W = original.shape
    p = patch_size
    t = tubelet_size
    n_h = H // p
    n_w = W // p
    n_t = T // t

    result = original.copy()

    for i in range(len(mask)):
        if not mask[i]:
            continue  # 可见 token, 保持原图

        t_idx = i // (n_h * n_w)
        spatial_idx = i % (n_h * n_w)
        h_idx = spatial_idx // n_w
        w_idx = spatial_idx % n_w

        # pred[i] shape: (in_chans * t * p * p)
        patch = pred[i].reshape(in_chans, t, p, p)

        t_start = t_idx * tubelet_size
        h_start = h_idx * patch_size
        w_start = w_idx * patch_size

        for dt in range(t):
            if t_start + dt < T:
                result[t_start+dt, h_start:h_start+p, w_start:w_start+p] = patch[0, dt]

    return result


# ============================================================
#  生成测试数据 (合成移动方块视频)
# ============================================================

def make_moving_square_video(num_frames=8, size=128, sq_size=24):
    """
    创建一个移动方块的合成视频
    比随机噪声更有意义 — 模型需要理解运动规律
    """
    frames = np.zeros((num_frames, size, size), dtype=np.float32)
    # 方块从左上到右下对角线移动
    for t in range(num_frames):
        progress = t / max(num_frames - 1, 1)
        cx = int(sq_size/2 + progress * (size - sq_size))
        cy = int(sq_size/2 + progress * (size - sq_size))
        y1, y2 = cy - sq_size//2, cy + sq_size//2
        x1, x2 = cx - sq_size//2, cx + sq_size//2
        frames[t, y1:y2, x1:x2] = 0.8

        # 加一个静止的圆形背景物体
        yc, xc = size//4, 3*size//4
        Y, X = np.ogrid[:size, :size]
        circle = ((Y - yc)**2 + (X - xc)**2) < (size//8)**2
        frames[t][circle] = 0.5

    # 加少量噪声
    frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02
    return np.clip(frames, 0, 1)


def make_expanding_ring_video(num_frames=8, size=128):
    """扩散环视频 — 模拟雷达回波扩散"""
    frames = np.zeros((num_frames, size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)

    for t in range(num_frames):
        radius = 10 + t * (size // 3) / num_frames
        ring = np.exp(-((dist - radius) ** 2) / (2 * 5**2))
        frames[t] = ring * 0.8

    frames += np.random.randn(*frames.shape).astype(np.float32) * 0.02
    return np.clip(frames, 0, 1)


# ============================================================
#  主演示
# ============================================================

def demo(use_sevir=False, num_train_steps=200, save_dir='demo_output'):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    print("=" * 60)
    print(" VideoMAE 效果演示")
    print("=" * 60)
    if device.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({mem:.2f} GB)")
    print()

    # --- 参数 ---
    img_size = 128
    num_frames = 8
    patch_size = 16
    tubelet_size = 2
    mask_ratio = 0.9
    embed_dim = 384   # ViT-Small
    depth = 12
    num_heads = 6
    decoder_dim = 192
    decoder_depth = 4

    n_spatial = (img_size // patch_size) ** 2  # 64
    n_temporal = num_frames // tubelet_size     # 4
    n_total = n_spatial * n_temporal             # 256
    n_visible = int(n_total * (1 - mask_ratio)) # 25

    print(f"ViT-Small: embed={embed_dim}, depth={depth}, heads={num_heads}")
    print(f"Input: 1×{num_frames}×{img_size}×{img_size}")
    print(f"Tokens: {n_total} total, {n_visible} visible ({(1-mask_ratio)*100:.0f}%)")
    print(f"Mask ratio: {mask_ratio} (遮挡 {mask_ratio*100:.0f}% 的 patch)")
    print()

    # --- 创建测试视频 ---
    print("[1/4] 创建测试视频...")
    if use_sevir:
        sevir_root = r'X:\datasets\sevir'
        catalog = os.path.join(sevir_root, "CATALOG.csv")
        if os.path.exists(catalog):
            from radar_dataset import SEVIRVideoMAEDataset
            ds = SEVIRVideoMAEDataset(
                sevir_root=sevir_root, num_frames=num_frames,
                input_size=img_size, mask_ratio=mask_ratio,
                tubelet_size=tubelet_size, patch_size=patch_size,
                split="train", max_samples=10,
            )
            video_tensor, mask_np = ds[0]
            test_video_np = video_tensor[0].numpy()  # (T, H, W)
            print(f"  使用 SEVIR 真实雷达数据")
        else:
            print(f"  ⚠ SEVIR 未找到, 使用合成数据")
            use_sevir = False

    if not use_sevir:
        video1 = make_moving_square_video(num_frames, img_size)
        video2 = make_expanding_ring_video(num_frames, img_size)
        test_video_np = video1  # 主测试用移动方块
        print(f"  合成视频: 移动方块 + 扩散环")

    # --- 构建模型 ---
    print("[2/4] 构建 VideoMAE ViT-Small...")
    model = VideoMAEPretrainModel(
        img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
        in_chans=1,
        encoder_embed_dim=embed_dim, encoder_depth=depth, encoder_num_heads=num_heads,
        decoder_embed_dim=decoder_dim, decoder_depth=decoder_depth,
        decoder_num_heads=num_heads,
    ).to(device)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {params:.1f}M")

    # --- 生成固定 mask ---
    np.random.seed(42)
    sp_mask = np.zeros(n_spatial, dtype=bool)
    sp_mask[np.random.choice(n_spatial, int(n_spatial * mask_ratio), replace=False)] = True
    tube_mask = np.tile(sp_mask, n_temporal)  # (N_total,)

    # --- 训练前: 随机初始化的重建效果 ---
    print("[3/4] 训练前的重建效果 (随机权重)...")
    model.eval()
    with torch.no_grad():
        video_t = torch.from_numpy(test_video_np).float().unsqueeze(0).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(tube_mask).unsqueeze(0).to(device)

        vis_tokens = model.encoder(video_t, mask_t)
        pred_before = model.decoder(vis_tokens, mask_t)
        pred_before_np = pred_before[0].cpu().numpy()  # (N_total, C*t*p*p)

    recon_before = reconstruct_from_pred(
        pred_before_np, tube_mask, test_video_np,
        patch_size, tubelet_size
    )
    print(f"  重建 MSE (训练前): {np.mean((recon_before - test_video_np)**2):.4f}")

    # --- 短训练 ---
    print(f"[4/4] 训练 {num_train_steps} steps...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 准备多个训练视频
    train_videos = [
        make_moving_square_video(num_frames, img_size, sq_size=np.random.randint(16, 40))
        for _ in range(20)
    ]
    train_videos += [
        make_expanding_ring_video(num_frames, img_size)
        for _ in range(20)
    ]

    losses = []
    for step in range(1, num_train_steps + 1):
        # 随机选视频
        vid = train_videos[np.random.randint(len(train_videos))]
        vid_t = torch.from_numpy(vid).float().unsqueeze(0).unsqueeze(0).to(device)

        # 随机生成新 mask
        sp_m = np.zeros(n_spatial, dtype=bool)
        sp_m[np.random.choice(n_spatial, int(n_spatial * mask_ratio), replace=False)] = True
        m = torch.from_numpy(np.tile(sp_m, n_temporal)).unsqueeze(0).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loss = model(vid_t, m)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == 1:
            avg = np.mean(losses[-50:])
            print(f"  step {step:4d}/{num_train_steps} | loss={avg:.4f}")

    # --- 训练后: 重建效果 ---
    print("\n重建效果对比:")
    model.eval()
    with torch.no_grad():
        vis_tokens = model.encoder(video_t, mask_t)
        pred_after = model.decoder(vis_tokens, mask_t)
        pred_after_np = pred_after[0].cpu().numpy()

    recon_after = reconstruct_from_pred(
        pred_after_np, tube_mask, test_video_np,
        patch_size, tubelet_size
    )
    mse_before = np.mean((recon_before - test_video_np)**2)
    mse_after = np.mean((recon_after - test_video_np)**2)
    print(f"  训练前 MSE: {mse_before:.4f}")
    print(f"  训练后 MSE: {mse_after:.4f}")
    print(f"  改善: {(1 - mse_after/mse_before)*100:.1f}%")

    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  峰值显存: {peak:.2f} GB")

    # --- 保存可视化 ---
    os.makedirs(save_dir, exist_ok=True)

    # 1. 原始视频帧
    grid_orig = frames_to_grid(test_video_np)
    save_pgm(os.path.join(save_dir, '1_original.pgm'), grid_orig)

    # 2. 被遮挡的视频 (mask 区域变红)
    masked_rgb = visualize_mask(test_video_np, tube_mask, patch_size, tubelet_size)
    for t in range(num_frames):
        # 保存每帧
        save_ppm(os.path.join(save_dir, f'2_masked_frame{t:02d}.ppm'), masked_rgb[t])

    # 汇总: 每帧取 R 通道做灰度网格
    grid_masked = frames_to_grid(masked_rgb[:, :, :, 0])
    save_pgm(os.path.join(save_dir, '2_masked_grid.pgm'), grid_masked)

    # 3. 训练前重建
    grid_recon_before = frames_to_grid(np.clip(recon_before, 0, 1))
    save_pgm(os.path.join(save_dir, '3_recon_before_train.pgm'), grid_recon_before)

    # 4. 训练后重建
    grid_recon_after = frames_to_grid(np.clip(recon_after, 0, 1))
    save_pgm(os.path.join(save_dir, '4_recon_after_train.pgm'), grid_recon_after)

    # 5. 差异图 (训练后)
    diff = np.abs(recon_after - test_video_np)
    grid_diff = frames_to_grid(np.clip(diff * 3, 0, 1))  # 放大 3 倍方便看
    save_pgm(os.path.join(save_dir, '5_error_map.pgm'), grid_diff)

    # 如果 PIL 可用, 保存为 PNG
    try:
        from PIL import Image

        def pgm_to_png(pgm_path):
            png_path = pgm_path.replace('.pgm', '.png')
            Image.open(pgm_path).save(png_path)
            return png_path

        def ppm_to_png(ppm_path):
            png_path = ppm_path.replace('.ppm', '.png')
            Image.open(ppm_path).save(png_path)
            return png_path

        pgm_to_png(os.path.join(save_dir, '1_original.pgm'))
        pgm_to_png(os.path.join(save_dir, '2_masked_grid.pgm'))
        pgm_to_png(os.path.join(save_dir, '3_recon_before_train.pgm'))
        pgm_to_png(os.path.join(save_dir, '4_recon_after_train.pgm'))
        pgm_to_png(os.path.join(save_dir, '5_error_map.pgm'))
        for t in range(num_frames):
            ppm_to_png(os.path.join(save_dir, f'2_masked_frame{t:02d}.ppm'))
        print(f"\n  ✅ PNG 图片已保存到 {save_dir}/")
        has_png = True
    except ImportError:
        print(f"\n  📁 PGM/PPM 图片已保存到 {save_dir}/")
        print("  (安装 Pillow 可生成 PNG: pip install Pillow)")
        has_png = False

    # --- Loss 曲线 (ASCII) ---
    print("\n  训练 Loss 曲线:")
    n_bins = 20
    chunk = len(losses) // n_bins
    for i in range(n_bins):
        avg = np.mean(losses[i*chunk:(i+1)*chunk])
        bar_len = int(avg * 100)
        bar = "█" * min(bar_len, 50)
        print(f"  step {i*chunk+1:4d}-{(i+1)*chunk:4d} | {avg:.4f} | {bar}")

    # --- 总结 ---
    print("\n" + "=" * 60)
    print(" VideoMAE 做了什么?")
    print("=" * 60)
    print("""
  VideoMAE 的核心思想:
  
  1. 输入: 一段视频 (8帧, 128×128)
  2. 切成 patch: 每个 patch 16×16 像素 × 2帧 = 一个 "tube"
  3. 随机遮挡 90%: 只让模型看到 10% 的 patch
  4. 模型任务: 从 10% 的信息重建 90% 被遮挡的内容
  
  为什么这能学到好的特征?
  - 要从 10% 重建 90%, 模型必须学会"理解"视频的时空结构
  - 移动的物体 → 模型学会运动规律
  - 扩散的回波 → 模型学会扩散模式
  - 这些"理解"编码在 Encoder 的权重中
  
  预训练完成后:
  - 丢弃 Decoder (只是辅助训练的)
  - 保留 Encoder → 这就是学到的"特征提取器"
  - 用这个 Encoder 做下游任务: 雷达预测、分类等
  
  JEPA vs MAE:
  - MAE (本演示): 重建像素 → 你能直接看到重建效果
  - JEPA: 在特征空间预测 → 不重建像素, 但避免了"像素级细节"的干扰
  - JEPA 通常学到更抽象、更有用的表示
  - 两者 Encoder 架构完全一样, 可以互换
""")

    print(f"  查看 {save_dir}/ 目录下的图片:")
    ext = "png" if has_png else "pgm"
    print(f"    1_original.{ext}         — 原始视频帧")
    print(f"    2_masked_grid.{ext}      — 被遮挡的帧 (红色=被遮挡)")
    print(f"    3_recon_before_train.{ext} — 训练前重建 (随机噪声)")
    print(f"    4_recon_after_train.{ext}  — 训练后重建 (应能看出形状)")
    print(f"    5_error_map.{ext}        — 重建误差图 (越黑越准)")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sevir', action='store_true', help='使用真实 SEVIR 数据')
    parser.add_argument('--steps', type=int, default=200, help='训练步数')
    parser.add_argument('--save_dir', type=str, default='demo_output')
    args = parser.parse_args()

    demo(use_sevir=args.use_sevir, num_train_steps=args.steps, save_dir=args.save_dir)
