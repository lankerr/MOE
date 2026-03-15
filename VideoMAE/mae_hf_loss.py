"""
MAE 高频损失实验 — 打破 Spectral Bias
========================================
在 mae_pro.py 基础上，加入多种高频损失，跑 Fashion-MNIST 消融实验

核心论点：MSE 天然偏向低频（F-Principle），高频细节（边缘、文字、细纹理）
在神经网络优化中总是最后才被拟合，且进度极慢。

解决方案（均为即插即用 Loss，不改模型架构）：
1. FFT Loss      — 频谱幅值+相位 L1 距离
2. Focal Freq    — 自适应聚焦高频难重建分量 (Jiang ICCV'21)
3. Sobel Edge    — 梯度空间 L1，逼迫边缘锐利
4. Wavelet Loss  — DWT 分解后重点惩罚 LH/HL/HH 高频子带

论文参考:
  - Focal Frequency Loss: arXiv:2012.12821, ICCV 2021
  - MWCNN (wavelet): arXiv:1805.07071, CVPR 2018
  - Wave-ViT: arXiv:2207.04978, ECCV 2022
  - FNO: arXiv:2010.08895, ICLR 2021
  - FNet: arXiv:2105.03824, NAACL 2022
  - Spectral Bias / F-Principle: arXiv:1905.10264

GitHub:
  - focal-frequency-loss: https://github.com/EndlessSora/focal-frequency-loss
  - pytorch_wavelets:     https://github.com/fbcotter/pytorch_wavelets
  - neuraloperator (FNO): https://github.com/neuraloperator/neuraloperator

用法:
  # 单实验 (MSE only baseline)
  python mae_hf_loss.py --loss_mode mse --epochs 100 --fresh

  # FFT Loss
  python mae_hf_loss.py --loss_mode fft --epochs 100 --fresh

  # 全部消融对比 (推荐!)
  python mae_hf_loss.py --sweep --sweep_epochs 100

  # 只跑最佳组合
  python mae_hf_loss.py --loss_mode combo --epochs 200 --fresh
"""

import os
import sys
import math
import time
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============================================================
#  复用 mae_pro 的模型组件
# ============================================================
from mae_pro import (
    MAEPro, MODEL_CONFIGS, get_dataset,
    FASHION_NAMES, CIFAR10_NAMES,
    analyze_latent,
)


# ############################################################
#  高频损失函数模块
# ############################################################

class FFTLoss(nn.Module):
    """
    频谱损失: 将预测和目标做 2D FFT，在频域计算 L1 距离
    同时约束幅值谱和相位谱

    高频分量在 FFT 频谱中位于边缘，低频在中心。
    直接在频域做 L1 可以赋予所有频率等权重（而不是像 MSE 那样被低频主导）。
    """
    def __init__(self, weight=1.0, log_amp=False):
        super().__init__()
        self.weight = weight
        self.log_amp = log_amp

    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, C, H, W)  [pixel space, 0~1]
        """
        # 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 幅值损失
        pred_amp = pred_fft.abs()
        target_amp = target_fft.abs()
        if self.log_amp:
            pred_amp = torch.log1p(pred_amp)
            target_amp = torch.log1p(target_amp)
        amp_loss = F.l1_loss(pred_amp, target_amp)

        # 相位损失
        phase_loss = F.l1_loss(pred_fft.angle(), target_fft.angle())

        return self.weight * (amp_loss + 0.1 * phase_loss)


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021)
    arXiv:2012.12821

    核心思想: 类似 Focal Loss 在分类中的作用——
    对 "容易重建的频率" 降权，对 "难重建的频率" 升权。
    这个权重是动态的，随训练进展自动调整。

    W(u,v) = |F_pred(u,v) - F_target(u,v)|^alpha
    L_FFL = mean(W * |F_pred - F_target|^2)

    alpha 越大 → 越聚焦于高误差(通常是高频)分量
    """
    def __init__(self, weight=1.0, alpha=1.0):
        super().__init__()
        self.weight = weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 频域距离 (复数距离)
        diff = pred_fft - target_fft
        diff_abs = torch.sqrt(diff.real ** 2 + diff.imag ** 2 + 1e-12)

        # 动态聚焦权重: 误差越大权重越高
        focal_weight = diff_abs.detach() ** self.alpha  # detach 防止梯度传入权重

        # 加权频域 L2
        loss = (focal_weight * diff_abs ** 2).mean()
        return self.weight * loss


class SobelEdgeLoss(nn.Module):
    """
    Sobel 边缘损失: 用固定 Sobel 算子提取水平/垂直梯度，
    然后在梯度空间计算 L1 距离。

    直观理解: 不仅要求 "像素值对"，还要求 "变化率（边缘）对"。
    这从数学上封死了模型 "用模糊平滑来偷懒" 的退路。
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        # Sobel 算子 (不可学习)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel(self, x):
        """对每个通道分别做 Sobel"""
        B, C, H, W = x.shape
        # (B*C, 1, H, W)
        x = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx.reshape(B, C, H, W), gy.reshape(B, C, H, W)

    def forward(self, pred, target):
        pred_gx, pred_gy = self._sobel(pred)
        target_gx, target_gy = self._sobel(target)
        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        return self.weight * loss


class WaveletLoss(nn.Module):
    """
    小波损失: 用 Haar DWT 将图像分解为 LL(低频) + LH/HL/HH(高频)，
    然后对高频子带施加更大的惩罚权重。

    Haar DWT 是最简单的小波变换，可以用 2x2 filter 实现：
      LL = (a+b+c+d)/2   (低频: 平均)
      LH = (a-b+c-d)/2   (水平高频)
      HL = (a+b-c-d)/2   (垂直高频)
      HH = (a-b-c+d)/2   (对角高频)

    不需要 pytorch_wavelets 库，纯 PyTorch 实现。
    """
    def __init__(self, weight=1.0, hf_weight=2.0):
        super().__init__()
        self.weight = weight
        self.hf_weight = hf_weight  # 高频子带的额外权重倍数

        # Haar 小波滤波器
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).reshape(1, 1, 2, 2) / 2.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32).reshape(1, 1, 2, 2) / 2.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32).reshape(1, 1, 2, 2) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).reshape(1, 1, 2, 2) / 2.0
        self.register_buffer('filter_ll', ll)
        self.register_buffer('filter_lh', lh)
        self.register_buffer('filter_hl', hl)
        self.register_buffer('filter_hh', hh)

    def _haar_dwt(self, x):
        """单层 Haar DWT: (B, C, H, W) → (LL, LH, HL, HH)"""
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        ll = F.conv2d(x, self.filter_ll, stride=2)
        lh = F.conv2d(x, self.filter_lh, stride=2)
        hl = F.conv2d(x, self.filter_hl, stride=2)
        hh = F.conv2d(x, self.filter_hh, stride=2)
        shape = (B, C, H // 2, W // 2)
        return ll.reshape(shape), lh.reshape(shape), hl.reshape(shape), hh.reshape(shape)

    def forward(self, pred, target):
        p_ll, p_lh, p_hl, p_hh = self._haar_dwt(pred)
        t_ll, t_lh, t_hl, t_hh = self._haar_dwt(target)

        loss_ll = F.l1_loss(p_ll, t_ll)
        loss_hf = (F.l1_loss(p_lh, t_lh) +
                   F.l1_loss(p_hl, t_hl) +
                   F.l1_loss(p_hh, t_hh))

        return self.weight * (loss_ll + self.hf_weight * loss_hf)


class LaplacianLoss(nn.Module):
    """
    拉普拉斯损失: 二阶导数，对细节变化更敏感
    Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('laplacian', lap)

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B * C, 1, H, W)
        target_flat = target.reshape(B * C, 1, H, W)
        pred_lap = F.conv2d(pred_flat, self.laplacian, padding=1)
        target_lap = F.conv2d(target_flat, self.laplacian, padding=1)
        return self.weight * F.l1_loss(pred_lap, target_lap)


# ############################################################
#  频谱分析工具 (可视化用)
# ############################################################

def compute_radial_spectrum(img_tensor):
    """
    计算径向平均功率谱 (Radial Average Power Spectrum)
    用于量化不同频率的能量分布

    Args:
        img_tensor: (C, H, W) 单张图像
    Returns:
        freqs: 归一化频率 (0~0.5)
        power: 对应功率
    """
    if img_tensor.dim() == 3:
        img = img_tensor.mean(dim=0)  # 灰度化
    else:
        img = img_tensor

    H, W = img.shape
    # 2D FFT
    fft = torch.fft.fft2(img)
    fft_shift = torch.fft.fftshift(fft)
    power = (fft_shift.abs() ** 2).numpy()

    # 径向平均
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)

    radial_power = np.zeros(max_r)
    for r in range(max_r):
        mask = (R == r)
        if mask.any():
            radial_power[r] = power[mask].mean()

    freqs = np.arange(max_r) / (2 * max_r)  # 归一化到 0~0.5
    return freqs, radial_power


# ############################################################
#  修改后的训练循环 (支持多种 Loss)
# ############################################################

LOSS_MODES = {
    'mse':      'MSE only (baseline)',
    'fft':      'MSE + FFT Loss',
    'focal':    'MSE + Focal Frequency Loss',
    'sobel':    'MSE + Sobel Edge Loss',
    'wavelet':  'MSE + Wavelet (Haar DWT) Loss',
    'laplacian':'MSE + Laplacian Loss',
    'combo':    'MSE + FFT + Sobel + Wavelet (最强组合)',
    'combo_focal': 'MSE + FocalFreq + Sobel + Wavelet',
}


def build_hf_losses(mode, device):
    """根据 loss_mode 构建高频损失函数列表"""
    losses = []
    if mode in ('fft', 'combo'):
        losses.append(('FFT', FFTLoss(weight=0.5).to(device)))
    if mode in ('focal', 'combo_focal'):
        losses.append(('FocalFreq', FocalFrequencyLoss(weight=0.5, alpha=1.0).to(device)))
    if mode in ('sobel', 'combo', 'combo_focal'):
        losses.append(('Sobel', SobelEdgeLoss(weight=1.0).to(device)))
    if mode in ('wavelet', 'combo', 'combo_focal'):
        losses.append(('Wavelet', WaveletLoss(weight=0.5, hf_weight=2.0).to(device)))
    if mode == 'laplacian':
        losses.append(('Laplacian', LaplacianLoss(weight=1.0).to(device)))
    return losses


def train_hf(args):
    """带高频损失的训练循环"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds, test_ds, img_size, in_chans, class_names = get_dataset(
        args.dataset, args.data_root, augment=args.augment)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    mcfg = MODEL_CONFIGS[args.model_size]

    print("=" * 62)
    print(f"  MAE HF-Loss — {args.dataset.upper()} ({img_size}×{img_size}×{in_chans})")
    print(f"  Loss Mode: {args.loss_mode} — {LOSS_MODES[args.loss_mode]}")
    print("=" * 62)

    model = MAEPro(
        img_size=img_size, patch_size=args.patch_size, in_chans=in_chans,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=False,  # ★ 关键: 关闭 norm_pix_loss 以获取真实像素空间的预测
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        **mcfg,
    ).to(device)

    # 构建高频损失
    hf_losses = build_hf_losses(args.loss_mode, device)

    n_patches = model.num_patches
    n_masked = int(n_patches * args.mask_ratio)

    print(f"  Device:     {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Patches:    {img_size}÷{args.patch_size} = {int(img_size/args.patch_size)}×"
          f"{int(img_size/args.patch_size)} = {n_patches} tokens")
    print(f"  Mask:       {args.mask_ratio*100:.0f}% ({n_masked} masked, "
          f"{n_patches - n_masked} visible)")
    print(f"  Model:      {args.model_size} (dim={mcfg['embed_dim']}, "
          f"depth={mcfg['depth']}, heads={mcfg['num_heads']})")
    print(f"  norm_pix:   OFF (使用像素空间 HF Loss)")
    print(f"  HF Losses:  {', '.join(n for n, _ in hf_losses) if hf_losses else 'None (MSE only)'}")

    total_p = sum(p.numel() for p in model.parameters())
    enc_p = sum(p.numel() for n, p in model.named_parameters()
                if 'decoder' not in n and 'mask_token' not in n)
    print(f"  Params:     {total_p/1e6:.2f}M (enc={enc_p/1e6:.2f}M)")

    eff_batch = args.batch_size * args.accum_steps
    lr = args.blr * eff_batch / 256
    print(f"  Optimizer:  AdamW (blr={args.blr}, eff_lr={lr:.1e}, batch={eff_batch})")
    print(f"  Epochs:     {args.epochs} (warmup={args.warmup_epochs})")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=args.wd)
    use_fp16 = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_loss = float('inf')
    ckpt_path = os.path.join(args.save_dir, 'best_mae.pt')
    history = {"train_loss": [], "val_loss": [], "lr": [],
               "train_mse": [], "train_hf": []}

    start_epoch = 0
    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('loss', float('inf'))
        print(f"  ★ Resume from epoch {start_epoch}, loss={best_loss:.5f}")

    print("-" * 62)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # LR schedule
        if epoch <= args.warmup_epochs:
            cur_lr = lr * epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        cur_lr = max(cur_lr, lr * 1e-3)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        model.train()
        total_loss, total_mse, total_hf, n_batch = 0., 0., 0., 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (imgs, _) in enumerate(train_dl):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_fp16):
                # MAE forward: 得到 MSE loss + pred patches + mask
                mse_loss, pred_patches, mask = model(imgs)

                hf_loss = torch.tensor(0.0, device=device)

                if hf_losses:
                    # ★ 关键: 在像素空间计算高频 loss
                    # 需要反 patchify 得到完整重建图像
                    with torch.no_grad():
                        target_patches = model.patchify(imgs)

                    # 构建全图预测 (visible 部分用原图, masked 部分用预测)
                    full_pred = target_patches * (1 - mask.unsqueeze(-1)) + \
                                pred_patches * mask.unsqueeze(-1)
                    pred_img = model.unpatchify(full_pred)
                    pred_img = pred_img.clamp(0, 1)

                    for _, loss_fn in hf_losses:
                        hf_loss = hf_loss + loss_fn(pred_img, imgs)

                loss = mse_loss + hf_loss
                loss = loss / args.accum_steps

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.accum_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_steps
            total_mse += mse_loss.item()
            total_hf += hf_loss.item()
            n_batch += 1

        train_loss = total_loss / n_batch
        train_mse = total_mse / n_batch
        train_hf = total_hf / n_batch
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_hf"].append(train_hf)
        history["lr"].append(cur_lr)

        # Validate (仅 MSE，公平比较)
        val_str = ""
        if epoch % 5 == 0 or epoch == args.epochs or epoch <= 3:
            model.eval()
            v_loss, v_n = 0., 0
            with torch.no_grad():
                for imgs_v, _ in test_dl:
                    imgs_v = imgs_v.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_fp16):
                        loss_v, _, _ = model(imgs_v)
                    v_loss += loss_v.item()
                    v_n += 1
            val_loss = v_loss / v_n
            val_str = f" | val={val_loss:.5f}"
            history["val_loss"].append((epoch, val_loss))

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                             'loss': val_loss,
                             'loss_mode': args.loss_mode,
                             'config': {
                                 'img_size': img_size, 'patch_size': args.patch_size,
                                 'in_chans': in_chans, 'model_size': args.model_size,
                                 'mask_ratio': args.mask_ratio,
                                 'norm_pix_loss': False,
                             }}, ckpt_path)

        hf_str = f"  hf={train_hf:.5f}" if hf_losses else ""
        print(f"  [{epoch:3d}/{args.epochs}]  total={train_loss:.5f}  mse={train_mse:.5f}"
              f"{hf_str}{val_str}  lr={cur_lr:.1e}  {elapsed:.1f}s", flush=True)

        if device.type == 'cuda' and epoch == start_epoch + 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  VRAM peak: {peak:.3f} GB", flush=True)

        # 可视化 (每 1/5 处 + 最后一个 epoch)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            visualize_hf(model, test_ds, device, args, class_names, in_chans,
                         suffix=f"_ep{epoch}")

    print(f"\n  ✓ 训练完成! best val_loss = {best_loss:.5f}")
    print(f"    loss_mode: {args.loss_mode}")
    print(f"    checkpoint: {ckpt_path}")

    hist_path = os.path.join(args.save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f)

    # 加载 best 模型
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])

    # 最终可视化
    visualize_hf(model, test_ds, device, args, class_names, in_chans, suffix="_final")

    # 频谱分析
    spectrum_analysis(model, test_ds, device, args, class_names, in_chans)

    # 隐空间分析
    if args.latent_analysis:
        analyze_latent(model, test_ds, device, args, class_names)

    return best_loss


# ############################################################
#  增强可视化: 加入频谱对比
# ############################################################

def visualize_hf(model, dataset, device, args, class_names, in_chans, suffix="", n_samples=10):
    """可视化: 原图 | Masked | 重建 | 误差×5 | 边缘对比"""
    model.eval()
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    torch.manual_seed(42)
    indices = list(range(n_samples))
    imgs = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = [dataset[i][1] for i in indices]

    with torch.no_grad():
        recon_imgs, masks = model.reconstruct(imgs, args.mask_ratio)
        recon_imgs = recon_imgs.clamp(0, 1)

    orig = imgs.cpu()
    recon = recon_imgs.cpu()
    masks_np = masks.cpu().numpy()

    # Sobel 边缘提取 (用于可视化)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    p = args.patch_size
    img_size = orig.shape[-1]
    h_g = w_g = img_size // p

    # 生成 masked 可视化
    masked_vis = orig.numpy().copy()
    for b in range(n_samples):
        for i in range(h_g * w_g):
            if masks_np[b, i] > 0.5:
                r, c = i // w_g, i % w_g
                masked_vis[b, :, r*p:(r+1)*p, c*p:(c+1)*p] = 0.15

    # 拼接大图: 原图 | Masked | 重建 | |误差|×5 | Sobel原图 | Sobel重建
    scale = 3 if img_size <= 32 else 2
    cell = img_size * scale
    pad = 2
    n_cols = 6
    title_h = 30
    label_w = 80

    total_w = label_w + n_cols * (cell + pad) + pad
    total_h = title_h + n_samples * (cell + pad) + pad
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = ["原图", f"Masked({int(args.mask_ratio*100)}%)",
               "重建", "|误差|×5", "Sobel(原)", "Sobel(重建)"]
    for col, hdr in enumerate(headers):
        x = label_w + pad + col * (cell + pad) + cell // 2 - len(hdr) * 4
        draw.text((x, 8), hdr, fill=(0, 0, 0))

    for b in range(n_samples):
        y_off = title_h + pad + b * (cell + pad)
        lbl = class_names[labels[b]]
        draw.text((4, y_off + cell // 2 - 6), lbl, fill=(50, 50, 50))

        # 提取灰度 (用于 Sobel)
        if in_chans == 1:
            orig_gray = orig[b:b+1]
            recon_gray = recon[b:b+1]
        else:
            orig_gray = orig[b:b+1].mean(dim=1, keepdim=True)
            recon_gray = recon[b:b+1].mean(dim=1, keepdim=True)

        orig_edge_x = F.conv2d(orig_gray, sobel_x, padding=1)
        orig_edge_y = F.conv2d(orig_gray, sobel_y, padding=1)
        orig_edge = torch.sqrt(orig_edge_x**2 + orig_edge_y**2)

        recon_edge_x = F.conv2d(recon_gray, sobel_x, padding=1)
        recon_edge_y = F.conv2d(recon_gray, sobel_y, padding=1)
        recon_edge = torch.sqrt(recon_edge_x**2 + recon_edge_y**2)

        # normalize edges to 0~1
        edge_max = max(orig_edge.max().item(), 1e-5)
        orig_edge_np = (orig_edge[0, 0].numpy() / edge_max).clip(0, 1)
        recon_edge_np = (recon_edge[0, 0].numpy() / edge_max).clip(0, 1)

        # 构建 6 个面板
        panels = []
        if in_chans == 1:
            panels.append(orig[b, 0].numpy())
            panels.append(masked_vis[b, 0])
            panels.append(recon[b, 0].numpy())
            panels.append(np.clip(np.abs(recon[b, 0].numpy() - orig[b, 0].numpy()) * 5, 0, 1))
        else:
            panels.append(np.transpose(orig[b].numpy(), (1, 2, 0)))
            panels.append(np.transpose(masked_vis[b], (1, 2, 0)))
            panels.append(np.transpose(recon[b].numpy(), (1, 2, 0)))
            err = np.clip(np.abs(np.transpose(
                (recon[b] - orig[b]).numpy(), (1, 2, 0))) * 5, 0, 1)
            panels.append(err)
        panels.append(orig_edge_np)
        panels.append(recon_edge_np)

        for col, panel in enumerate(panels):
            arr = np.clip(panel * 255, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                img_pil = Image.fromarray(arr, mode='L').convert('RGB')
            else:
                img_pil = Image.fromarray(arr, mode='RGB')
            img_pil = img_pil.resize((cell, cell), Image.NEAREST)
            x_off = label_w + pad + col * (cell + pad)
            canvas.paste(img_pil, (x_off, y_off))

    path = os.path.join(args.save_dir, f"recon{suffix}.png")
    canvas.save(path)
    print(f"  [可视化] {path}", flush=True)


def spectrum_analysis(model, dataset, device, args, class_names, in_chans):
    """
    频谱分析: 对比原图 vs 重建图的径向功率谱
    可以清晰看出高频分量的恢复程度
    """
    model.eval()
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    print("\n  [频谱分析]", flush=True)

    # 收集一批样本
    torch.manual_seed(42)
    n_analyse = 100
    indices = list(range(min(n_analyse, len(dataset))))
    imgs = torch.stack([dataset[i][0] for i in indices]).to(device)

    with torch.no_grad():
        recon_imgs, _ = model.reconstruct(imgs, args.mask_ratio)
        recon_imgs = recon_imgs.clamp(0, 1)

    orig_cpu = imgs.cpu()
    recon_cpu = recon_imgs.cpu()

    # 计算平均径向功率谱
    all_orig_spec = []
    all_recon_spec = []
    for i in range(len(indices)):
        freqs, orig_spec = compute_radial_spectrum(orig_cpu[i])
        _, recon_spec = compute_radial_spectrum(recon_cpu[i])
        all_orig_spec.append(orig_spec)
        all_recon_spec.append(recon_spec)

    avg_orig = np.mean(all_orig_spec, axis=0)
    avg_recon = np.mean(all_recon_spec, axis=0)

    # 分频段统计
    n_freq = len(freqs)
    low = slice(0, n_freq // 3)
    mid = slice(n_freq // 3, 2 * n_freq // 3)
    high = slice(2 * n_freq // 3, n_freq)

    orig_power = {'low': avg_orig[low].sum(), 'mid': avg_orig[mid].sum(), 'high': avg_orig[high].sum()}
    recon_power = {'low': avg_recon[low].sum(), 'mid': avg_recon[mid].sum(), 'high': avg_recon[high].sum()}

    print(f"  频段能量对比:")
    print(f"    {'频段':<8} {'原图':>12} {'重建':>12} {'恢复率':>10}")
    for band in ['low', 'mid', 'high']:
        ratio = recon_power[band] / (orig_power[band] + 1e-10) * 100
        print(f"    {band:<8} {orig_power[band]:>12.1f} {recon_power[band]:>12.1f} {ratio:>9.1f}%")

    # 绘制频谱对比图 (纯 PIL)
    W, H = 600, 400
    margin = 60
    plot = Image.new('RGB', (W, H), (255, 255, 255))
    pdraw = ImageDraw.Draw(plot)

    # 用 log scale
    eps = 1e-10
    log_orig = np.log10(avg_orig + eps)
    log_recon = np.log10(avg_recon + eps)

    y_min = min(log_orig.min(), log_recon.min()) - 0.5
    y_max = max(log_orig.max(), log_recon.max()) + 0.5
    x_max = len(freqs)

    def to_px(xi, yi):
        px = margin + (xi / x_max) * (W - 2 * margin)
        py = H - margin - ((yi - y_min) / (y_max - y_min + 1e-8)) * (H - 2 * margin)
        return int(px), int(py)

    # 坐标轴
    pdraw.line([(margin, margin), (margin, H - margin), (W - margin, H - margin)],
               fill=(0, 0, 0), width=1)
    pdraw.text((W // 2 - 50, H - 20), "频率 (归一化)", fill=(0, 0, 0))
    pdraw.text((5, margin - 15), "log₁₀(Power)", fill=(0, 0, 0))

    # 频段分界线
    for frac, label in [(1/3, "低|中"), (2/3, "中|高")]:
        bx = int(margin + frac * (W - 2 * margin))
        pdraw.line([(bx, margin), (bx, H - margin)], fill=(200, 200, 200), width=1)
        pdraw.text((bx - 10, margin - 12), label, fill=(150, 150, 150))

    # 原图频谱 (蓝)
    for i in range(1, len(freqs)):
        x1, y1 = to_px(i - 1, log_orig[i - 1])
        x2, y2 = to_px(i, log_orig[i])
        pdraw.line([(x1, y1), (x2, y2)], fill=(31, 119, 180), width=2)

    # 重建频谱 (橙)
    for i in range(1, len(freqs)):
        x1, y1 = to_px(i - 1, log_recon[i - 1])
        x2, y2 = to_px(i, log_recon[i])
        pdraw.line([(x1, y1), (x2, y2)], fill=(255, 127, 14), width=2)

    # 图例和标题
    pdraw.text((W - margin - 100, margin + 5), "— 原图", fill=(31, 119, 180))
    pdraw.text((W - margin - 100, margin + 20), "— 重建", fill=(255, 127, 14))

    # 恢复率标注
    hi_ratio = recon_power['high'] / (orig_power['high'] + 1e-10) * 100
    pdraw.text((W - margin - 130, margin + 45),
               f"高频恢复: {hi_ratio:.1f}%", fill=(214, 39, 40))

    title = f"径向功率谱: {args.loss_mode} ({args.dataset})"
    pdraw.text((W // 2 - len(title) * 3, 5), title, fill=(0, 0, 0))

    path = os.path.join(args.save_dir, "spectrum_analysis.png")
    plot.save(path)
    print(f"  [频谱图] {path}", flush=True)

    # 保存频谱数据
    np.savez(os.path.join(args.save_dir, "spectrum_data.npz"),
             freqs=freqs, orig=avg_orig, recon=avg_recon)


# ############################################################
#  Sweep: 多种 Loss 对比
# ############################################################

def run_hf_sweep(args):
    """系统性对比所有高频 Loss"""
    results = []
    base_dir = args.save_dir

    experiments = [
        # (名称, loss_mode, 额外参数)
        ("1_mse_baseline",    "mse",       {}),
        ("2_fft",             "fft",       {}),
        ("3_focal_freq",      "focal",     {}),
        ("4_sobel_edge",      "sobel",     {}),
        ("5_wavelet_haar",    "wavelet",   {}),
        ("6_laplacian",       "laplacian", {}),
        ("7_combo_best",      "combo",     {}),
        ("8_combo_focal",     "combo_focal", {}),
    ]

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  高频 Loss 消融实验: {len(experiments)} experiments      ║")
    print(f"╚══════════════════════════════════════════════╝")

    for i, (name, loss_mode, overrides) in enumerate(experiments):
        print(f"\n{'='*62}")
        print(f"  [{i+1}/{len(experiments)}] {name}: {LOSS_MODES[loss_mode]}")
        print(f"{'='*62}")

        exp_args = copy.deepcopy(args)
        exp_args.save_dir = os.path.join(base_dir, name)
        exp_args.loss_mode = loss_mode
        exp_args.fresh = True
        exp_args.epochs = args.sweep_epochs
        exp_args.latent_analysis = False  # sweep 跳过 t-SNE 节省时间
        for k, v in overrides.items():
            setattr(exp_args, k, v)

        try:
            best = train_hf(exp_args)
            results.append({"name": name, "loss_mode": loss_mode,
                            "best_val_loss": best})
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"name": name, "loss_mode": loss_mode, "error": str(e)})

    # 汇总
    print(f"\n{'='*62}")
    print(f"  高频 Loss 消融结果汇总")
    print(f"{'='*62}")
    print(f"  {'实验':<25} {'Loss模式':<15} {'best_val':>12}")
    print(f"  {'-'*55}")
    for r in sorted(results, key=lambda x: x.get('best_val_loss', 999)):
        if 'best_val_loss' in r:
            print(f"  {r['name']:<25} {r['loss_mode']:<15} {r['best_val_loss']:>12.5f}")
        else:
            print(f"  {r['name']:<25} {r['loss_mode']:<15} {'FAILED':>12}")

    log_path = os.path.join(base_dir, "hf_sweep_results.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  保存: {log_path}")

    # ★ 生成扫描对比大图
    make_sweep_comparison(base_dir, results, args)


def make_sweep_comparison(base_dir, results, args):
    """生成各 Loss 模式的横向对比图"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    # 收集各实验的频谱数据
    spectra = {}
    for r in results:
        if 'error' in r:
            continue
        spec_path = os.path.join(base_dir, r['name'], 'spectrum_data.npz')
        if os.path.exists(spec_path):
            data = np.load(spec_path)
            spectra[r['name']] = {
                'freqs': data['freqs'],
                'orig': data['orig'],
                'recon': data['recon'],
                'loss_mode': r['loss_mode'],
                'val_loss': r.get('best_val_loss', 999),
            }

    if not spectra:
        print("  [对比图] 没有可用的频谱数据")
        return

    # 1. 频谱对比大图
    n_exps = len(spectra)
    W, H = 600, 300
    total_h = n_exps * H
    canvas = Image.new('RGB', (W, total_h), (255, 255, 255))

    colors_recon = [
        (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194),
        (188, 189, 34), (23, 190, 207),
    ]

    for idx, (name, spec) in enumerate(spectra.items()):
        sub = Image.new('RGB', (W, H), (255, 255, 255))
        pdraw = ImageDraw.Draw(sub)

        freqs = spec['freqs']
        orig = np.log10(spec['orig'] + 1e-10)
        recon = np.log10(spec['recon'] + 1e-10)

        y_min = min(orig.min(), recon.min()) - 0.5
        y_max = max(orig.max(), recon.max()) + 0.5
        margin = 60
        x_max = len(freqs)

        def to_px(xi, yi):
            px = margin + (xi / x_max) * (W - 2 * margin)
            py = H - margin - ((yi - y_min) / (y_max - y_min + 1e-8)) * (H - 2 * margin)
            return int(px), int(py)

        # 轴
        pdraw.line([(margin, margin), (margin, H - margin), (W - margin, H - margin)],
                   fill=(0,0,0), width=1)

        # 原图 (蓝)
        for i in range(1, len(freqs)):
            x1, y1 = to_px(i-1, orig[i-1])
            x2, y2 = to_px(i, orig[i])
            pdraw.line([(x1,y1),(x2,y2)], fill=(31,119,180), width=2)

        # 重建 (彩色)
        c = colors_recon[idx % len(colors_recon)]
        for i in range(1, len(freqs)):
            x1, y1 = to_px(i-1, recon[i-1])
            x2, y2 = to_px(i, recon[i])
            pdraw.line([(x1,y1),(x2,y2)], fill=c, width=2)

        # 高频恢复率
        n_f = len(freqs)
        hi_orig = spec['orig'][2*n_f//3:].sum()
        hi_recon = spec['recon'][2*n_f//3:].sum()
        hi_ratio = hi_recon / (hi_orig + 1e-10) * 100

        pdraw.text((10, 5),
                   f"{spec['loss_mode']} | val={spec['val_loss']:.5f} | HF={hi_ratio:.1f}%",
                   fill=(0,0,0))

        canvas.paste(sub, (0, idx * H))

    path = os.path.join(base_dir, "sweep_spectrum_compare.png")
    canvas.save(path)
    print(f"  [频谱对比] {path}")

    # 2. 柱状图: 各模式的高频恢复率
    bar_data = []
    for name, spec in spectra.items():
        n_f = len(spec['freqs'])
        hi_orig = spec['orig'][2*n_f//3:].sum()
        hi_recon = spec['recon'][2*n_f//3:].sum()
        ratio = hi_recon / (hi_orig + 1e-10) * 100
        bar_data.append((spec['loss_mode'], ratio, spec['val_loss']))

    BW, BH = 700, 400
    bar_img = Image.new('RGB', (BW, BH), (255, 255, 255))
    bdraw = ImageDraw.Draw(bar_img)
    margin = 80
    n_bar = len(bar_data)
    bar_w = max(20, (BW - 2 * margin) // (n_bar * 2))
    gap = bar_w

    max_ratio = max(d[1] for d in bar_data) * 1.2

    bdraw.text((BW // 2 - 80, 5), "高频恢复率 (%) — 越高越好", fill=(0,0,0))
    bdraw.line([(margin, BH - margin), (BW - margin, BH - margin)], fill=(0,0,0))
    bdraw.line([(margin, margin), (margin, BH - margin)], fill=(0,0,0))

    for i, (mode, ratio, vloss) in enumerate(bar_data):
        x = margin + i * (bar_w + gap) + gap // 2
        bar_h = int((ratio / max_ratio) * (BH - 2 * margin))
        y_top = BH - margin - bar_h

        c = colors_recon[i % len(colors_recon)]
        bdraw.rectangle([(x, y_top), (x + bar_w, BH - margin)], fill=c)
        bdraw.text((x, BH - margin + 5), mode, fill=(0,0,0))
        bdraw.text((x, y_top - 15), f"{ratio:.1f}%", fill=(0,0,0))

    path = os.path.join(base_dir, "sweep_hf_recovery_bar.png")
    bar_img.save(path)
    print(f"  [柱状图] {path}")


# ############################################################
#  Main
# ############################################################

def main():
    p = argparse.ArgumentParser("MAE High-Frequency Loss Experiments")
    p.add_argument('--dataset', default='fashion', choices=['fashion', 'cifar10', 'mnist'])
    p.add_argument('--loss_mode', default='mse', choices=list(LOSS_MODES.keys()))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--blr', type=float, default=1.5e-4)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--patch_size', type=int, default=4)
    p.add_argument('--model_size', default='small', choices=['tiny', 'small', 'base'])
    p.add_argument('--augment', action='store_true', default=True)
    p.add_argument('--no_augment', dest='augment', action='store_false')
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--drop_rate', type=float, default=0.0)
    p.add_argument('--attn_drop_rate', type=float, default=0.0)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--latent_analysis', action='store_true')
    p.add_argument('--fresh', action='store_true')
    # Sweep
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--sweep_epochs', type=int, default=80)
    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = f"mae_hf_{args.dataset}"

    if args.sweep:
        run_hf_sweep(args)
    else:
        train_hf(args)


if __name__ == "__main__":
    main()
