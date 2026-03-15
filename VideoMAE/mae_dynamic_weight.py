"""
MAE 动态 Loss 加权 + 批量可视化
=================================
1. 动态加权实验: Uncertainty / EMA-Ratio / DWA / RLW
2. 加载全部已有模型，批量生成 recon 大图 + 频谱对比

动态加权论文:
  - Uncertainty Weighting: Kendall CVPR'18, arXiv:1705.07115
  - GradNorm: Chen ICML'18, arXiv:1711.02257
  - DWA: Liu CVPR'19, arXiv:1803.10704
  - RLW: Lin TMLR'22, arXiv:2111.10603

用法:
  # 动态加权 sweep (4种方法 + fixed baseline)
  python mae_dynamic_weight.py --sweep --sweep_epochs 80

  # 单独跑 uncertainty weighting
  python mae_dynamic_weight.py --weight_mode uncertainty --epochs 100

  # 批量可视化所有已有模型 (读取已训练模型生成多角度PNG)
  python mae_dynamic_weight.py --visualize_all
"""

import os, sys, math, time, json, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mae_pro import (
    MAEPro, MODEL_CONFIGS, get_dataset,
    FASHION_NAMES, CIFAR10_NAMES, analyze_latent,
)
from mae_hf_loss import (
    FFTLoss, FocalFrequencyLoss, SobelEdgeLoss,
    WaveletLoss, LaplacianLoss, compute_radial_spectrum,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
#  动态加权模块
# ============================================================

class UncertaintyWeighting(nn.Module):
    """
    Kendall CVPR'18 — 同方差不确定性加权
    L_total = Σ (1/2σ²_i) * L_i + log(σ_i)
    实现: s_i = log(σ²_i) 作为可学习参数
    """
    def __init__(self, n_losses):
        super().__init__()
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_losses)])
        self.n = n_losses

    def forward(self, losses):
        total = torch.tensor(0.0, device=losses[0].device)
        weights = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i][0])
            total = total + precision * loss + self.log_vars[i][0]
            weights.append(precision.item())
        return total, weights

    def get_sigma_str(self):
        return " | ".join([f"σ²={torch.exp(s[0]).item():.4f}" for s in self.log_vars])


class EMALossBalancer:
    """
    EMA Loss Ratio — 用指数移动平均归一化各 loss
    损失大的项权重低，保证各项贡献均衡
    """
    def __init__(self, n_losses, alpha=0.99):
        self.ema = [1.0] * n_losses
        self.alpha = alpha
        self.n = n_losses

    def get_weights(self, losses_values):
        weights = []
        for i, lv in enumerate(losses_values):
            self.ema[i] = self.alpha * self.ema[i] + (1 - self.alpha) * lv
            weights.append(1.0 / (self.ema[i] + 1e-8))
        s = sum(weights)
        return [w * self.n / s for w in weights]


class DWAWeighting:
    """
    DWA (Liu CVPR'19) — 根据 loss 下降速率动态调权
    下降慢的 loss 获得更大权重
    """
    def __init__(self, n_losses, temperature=2.0):
        self.n = n_losses
        self.T = temperature
        self.prev_losses = [None, None]  # [t-2, t-1]

    def get_weights(self, current_losses):
        if self.prev_losses[0] is None or self.prev_losses[1] is None:
            self.prev_losses[1] = self.prev_losses[0]
            self.prev_losses[0] = current_losses
            return [1.0] * self.n

        ratios = []
        for i in range(self.n):
            r = current_losses[i] / (self.prev_losses[0][i] + 1e-8)
            ratios.append(r)

        # softmax with temperature
        exp_r = [math.exp(r / self.T) for r in ratios]
        s = sum(exp_r)
        weights = [self.n * e / s for e in exp_r]

        self.prev_losses[1] = self.prev_losses[0]
        self.prev_losses[0] = current_losses
        return weights


class RLWWeighting:
    """
    RLW (Lin TMLR'22) — 每步从 Dirichlet 分布随机采样权重
    """
    def __init__(self, n_losses):
        self.n = n_losses
        self.dist = torch.distributions.Dirichlet(torch.ones(n_losses))

    def get_weights(self):
        w = self.dist.sample()
        return (w * self.n).tolist()


# ============================================================
#  动态加权训练
# ============================================================

WEIGHT_MODES = {
    'fixed':       'Fixed weights (λ=1.0 each)',
    'uncertainty': 'Kendall Uncertainty (learned σ²)',
    'ema':         'EMA Loss Ratio Balancing',
    'dwa':         'Dynamic Weight Averaging (T=2)',
    'rlw':         'Random Loss Weighting (Dirichlet)',
}


def train_dynamic(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds, test_ds, img_size, in_chans, class_names = get_dataset(
        args.dataset, args.data_root, augment=args.augment)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    mcfg = MODEL_CONFIGS[args.model_size]

    # 固定 combo loss: MSE + FFT + Sobel (基于上轮实验结论)
    loss_names = ['MSE', 'FFT', 'Sobel']
    hf_modules = [
        FFTLoss(weight=1.0).to(device),
        SobelEdgeLoss(weight=1.0).to(device),
    ]

    print("=" * 62)
    print(f"  MAE 动态加权 — {args.dataset.upper()} ({img_size}×{img_size}×{in_chans})")
    print(f"  Weight Mode: {args.weight_mode} — {WEIGHT_MODES[args.weight_mode]}")
    print(f"  Losses: {' + '.join(loss_names)}")
    print("=" * 62)

    model = MAEPro(
        img_size=img_size, patch_size=args.patch_size, in_chans=in_chans,
        mask_ratio=args.mask_ratio, norm_pix_loss=False,
        **mcfg,
    ).to(device)

    # 动态加权器
    n_losses = len(loss_names)
    uncertainty_module = None
    ema_balancer = None
    dwa_weighter = None
    rlw_weighter = None

    if args.weight_mode == 'uncertainty':
        uncertainty_module = UncertaintyWeighting(n_losses).to(device)
    elif args.weight_mode == 'ema':
        ema_balancer = EMALossBalancer(n_losses, alpha=0.99)
    elif args.weight_mode == 'dwa':
        dwa_weighter = DWAWeighting(n_losses, temperature=2.0)
    elif args.weight_mode == 'rlw':
        rlw_weighter = RLWWeighting(n_losses)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params:     {total_p/1e6:.2f}M")

    eff_batch = args.batch_size * args.accum_steps
    lr = args.blr * eff_batch / 256

    # 优化器: 如果 uncertainty 模式, 加入 log_vars 参数
    params = list(model.parameters())
    if uncertainty_module is not None:
        params += list(uncertainty_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                                   weight_decay=args.wd)

    use_fp16 = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_loss = float('inf')
    ckpt_path = os.path.join(args.save_dir, 'best_mae.pt')
    history = {"train_loss": [], "val_loss": [], "lr": [],
               "weights": [], "per_loss": {n: [] for n in loss_names}}

    start_epoch = 0
    if os.path.exists(ckpt_path) and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('loss', float('inf'))
        print(f"  ✓ 断点续训: 从 epoch {start_epoch} 继续 (best={best_loss:.5f})")
        # 加载已有 history
        hist_path_resume = os.path.join(args.save_dir, 'history.json')
        if os.path.exists(hist_path_resume):
            with open(hist_path_resume) as f:
                old_hist = json.load(f)
            history["train_loss"] = old_hist.get("train_loss", [])
            history["val_loss"] = old_hist.get("val_loss", [])
            history["lr"] = old_hist.get("lr", [])
            history["weights"] = old_hist.get("weights", [])
            for n in loss_names:
                history["per_loss"][n] = old_hist.get("per_loss", {}).get(n, [])

    # 已经训练完成则跳过
    hist_path_check = os.path.join(args.save_dir, 'history.json')
    if os.path.exists(hist_path_check) and not args.fresh:
        with open(hist_path_check) as f:
            old_hist = json.load(f)
        done_epochs = len(old_hist.get('train_loss', []))
        if done_epochs >= args.epochs:
            print(f"  ⏭ 已完成 {done_epochs} epochs >= {args.epochs}, 跳过")
            return best_loss

    print(f"  Optimizer: AdamW (lr={lr:.1e}), Epochs: {args.epochs}, Patience: {args.patience}")
    print("-" * 62)

    patience_counter = 0

    # cosine schedule 周期: 不超过200, 之后保持 min_lr
    cosine_T = min(args.epochs, 200)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # LR schedule
        if epoch <= args.warmup_epochs:
            cur_lr = lr * epoch / args.warmup_epochs
        elif epoch <= cosine_T:
            progress = (epoch - args.warmup_epochs) / max(cosine_T - args.warmup_epochs, 1)
            cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            cur_lr = lr * 1e-3  # cosine 结束后保持 min_lr
        cur_lr = max(cur_lr, lr * 1e-3)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        model.train()
        if uncertainty_module:
            uncertainty_module.train()

        totals = {n: 0.0 for n in loss_names}
        total_combined = 0.0
        n_batch = 0
        epoch_weights = []
        t0 = time.time()
        optimizer.zero_grad()

        for step, (imgs, _) in enumerate(train_dl):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_fp16):
                mse_loss, pred_patches, mask = model(imgs)

                # 像素空间重建
                with torch.no_grad():
                    target_patches = model.patchify(imgs)
                full_pred = target_patches * (1 - mask.unsqueeze(-1)) + \
                            pred_patches * mask.unsqueeze(-1)
                pred_img = model.unpatchify(full_pred).clamp(0, 1)

                fft_loss = hf_modules[0](pred_img, imgs)
                sobel_loss = hf_modules[1](pred_img, imgs)

                individual_losses = [mse_loss, fft_loss, sobel_loss]

                # 动态加权
                if args.weight_mode == 'fixed':
                    weights = [1.0, 0.5, 1.0]
                    combined = sum(w * l for w, l in zip(weights, individual_losses))

                elif args.weight_mode == 'uncertainty':
                    combined, weights = uncertainty_module(individual_losses)

                elif args.weight_mode == 'ema':
                    loss_vals = [l.item() for l in individual_losses]
                    weights = ema_balancer.get_weights(loss_vals)
                    combined = sum(w * l for w, l in zip(weights, individual_losses))

                elif args.weight_mode == 'dwa':
                    loss_vals = [l.item() for l in individual_losses]
                    weights = dwa_weighter.get_weights(loss_vals)
                    combined = sum(w * l for w, l in zip(weights, individual_losses))

                elif args.weight_mode == 'rlw':
                    weights = rlw_weighter.get_weights()
                    combined = sum(w * l for w, l in zip(weights, individual_losses))

                combined = combined / args.accum_steps

            if use_fp16:
                scaler.scale(combined).backward()
            else:
                combined.backward()

            if (step + 1) % args.accum_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_combined += combined.item() * args.accum_steps
            for i, n in enumerate(loss_names):
                totals[n] += individual_losses[i].item()
            epoch_weights.append([float(w) if isinstance(w, (int, float)) else w for w in weights])
            n_batch += 1

        train_loss = total_combined / n_batch
        avg_weights = [np.mean([w[i] for w in epoch_weights]) for i in range(n_losses)]
        history["train_loss"].append(train_loss)
        history["lr"].append(cur_lr)
        history["weights"].append(avg_weights)
        for i, n in enumerate(loss_names):
            history["per_loss"][n].append(totals[n] / n_batch)

        elapsed = time.time() - t0

        # Validate
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
                patience_counter = 0
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                             'loss': val_loss, 'weight_mode': args.weight_mode,
                             'config': {
                                 'img_size': img_size, 'patch_size': args.patch_size,
                                 'in_chans': in_chans, 'model_size': args.model_size,
                                 'mask_ratio': args.mask_ratio,
                             }}, ckpt_path)
            else:
                patience_counter += 1

        w_str = " ".join([f"w{i}={avg_weights[i]:.2f}" for i in range(n_losses)])
        sigma_str = ""
        if uncertainty_module:
            sigma_str = f"  [{uncertainty_module.get_sigma_str()}]"
        print(f"  [{epoch:3d}/{args.epochs}]  loss={train_loss:.5f}{val_str}"
              f"  {w_str}{sigma_str}  lr={cur_lr:.1e}  {elapsed:.1f}s", flush=True)

        if device.type == 'cuda' and epoch == start_epoch + 1:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  VRAM peak: {peak:.3f} GB", flush=True)

        if epoch % 20 == 0:
            visualize_recon(model, test_ds, device, args, class_names, in_chans,
                            suffix=f"_ep{epoch}")

        # 每 epoch 保存 history (防中断丢失)
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (patience={args.patience}, "
                  f"best_val={best_loss:.5f})")
            break

    print(f"\n  ✓ 完成! best val_loss = {best_loss:.5f}, mode={args.weight_mode}")

    hist_path = os.path.join(args.save_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f)

    # 绘制 loss 曲线 + 权重变化
    plot_training_curves(history, args.save_dir, args.weight_mode)

    # 加载 best, 最终可视化
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
    visualize_recon(model, test_ds, device, args, class_names, in_chans, suffix="_final")
    spectrum_analysis_simple(model, test_ds, device, args)

    # 绘制 loss 曲线 + 权重变化
    plot_training_curves(history, args.save_dir, args.weight_mode)

    return best_loss


# ============================================================
#  训练曲线绘制
# ============================================================

def plot_training_curves(history, save_dir, weight_mode):
    """绘制 train/val loss + 各分项 loss + 权重变化曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves — {weight_mode}", fontsize=14)

    # 1. Train & Val Loss
    ax = axes[0, 0]
    epochs_train = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs_train, history["train_loss"], 'b-', alpha=0.7, label='train (combined)')
    if history["val_loss"]:
        val_e = [v[0] for v in history["val_loss"]]
        val_l = [v[1] for v in history["val_loss"]]
        ax.plot(val_e, val_l, 'r-o', markersize=3, label='val (MSE)')
        best_idx = np.argmin(val_l)
        ax.axvline(val_e[best_idx], color='g', linestyle='--', alpha=0.5,
                    label=f'best={val_l[best_idx]:.5f} @ep{val_e[best_idx]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train & Val Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 各分项 loss
    ax = axes[0, 1]
    for name, vals in history["per_loss"].items():
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, label=name, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Individual Losses')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 权重变化
    ax = axes[1, 0]
    if history["weights"]:
        w_arr = np.array(history["weights"])
        loss_names = list(history["per_loss"].keys())
        for i, name in enumerate(loss_names):
            if i < w_arr.shape[1]:
                ax.plot(range(1, len(w_arr) + 1), w_arr[:, i], label=f'w_{name}', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_title(f'Dynamic Weights ({weight_mode})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. LR schedule
    ax = axes[1, 1]
    if history["lr"]:
        ax.plot(range(1, len(history["lr"]) + 1), history["lr"], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [曲线图] {path}", flush=True)


# ============================================================
#  可视化工具
# ============================================================

def visualize_recon(model, dataset, device, args, class_names, in_chans,
                    suffix="", n_samples=10, seed=42):
    model.eval()
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    torch.manual_seed(seed)
    indices = list(range(n_samples))
    imgs = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = [dataset[i][1] for i in indices]

    with torch.no_grad():
        recon_imgs, masks = model.reconstruct(imgs, args.mask_ratio)
        recon_imgs = recon_imgs.clamp(0, 1)

    orig = imgs.cpu().numpy()
    recon = recon_imgs.cpu().numpy()
    masks_np = masks.cpu().numpy()

    # Sobel
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).reshape(1,1,3,3)
    sobel_y = sobel_x.transpose(2, 3)

    p = args.patch_size
    img_size = orig.shape[-1]
    h_g = w_g = img_size // p
    masked_vis = orig.copy()
    for b in range(n_samples):
        for i in range(h_g * w_g):
            if masks_np[b, i] > 0.5:
                r, c = i // w_g, i % w_g
                masked_vis[b, :, r*p:(r+1)*p, c*p:(c+1)*p] = 0.15

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
        draw.text((4, y_off + cell // 2 - 6), class_names[labels[b]], fill=(50, 50, 50))

        orig_t = torch.from_numpy(orig[b:b+1])
        recon_t = torch.from_numpy(recon[b:b+1])
        if in_chans == 1:
            og = orig_t
            rg = recon_t
        else:
            og = orig_t.mean(dim=1, keepdim=True)
            rg = recon_t.mean(dim=1, keepdim=True)
        oex = F.conv2d(og, sobel_x, padding=1)
        oey = F.conv2d(og, sobel_y, padding=1)
        oe = torch.sqrt(oex**2 + oey**2)
        rex = F.conv2d(rg, sobel_x, padding=1)
        rey = F.conv2d(rg, sobel_y, padding=1)
        re = torch.sqrt(rex**2 + rey**2)
        emax = max(oe.max().item(), 1e-5)
        oe_np = (oe[0,0].numpy() / emax).clip(0, 1)
        re_np = (re[0,0].numpy() / emax).clip(0, 1)

        panels = []
        if in_chans == 1:
            panels.append(orig[b, 0])
            panels.append(masked_vis[b, 0])
            panels.append(recon[b, 0])
            panels.append(np.clip(np.abs(recon[b, 0] - orig[b, 0]) * 5, 0, 1))
        else:
            panels.append(np.transpose(orig[b], (1, 2, 0)))
            panels.append(np.transpose(masked_vis[b], (1, 2, 0)))
            panels.append(np.transpose(recon[b], (1, 2, 0)))
            err = np.clip(np.abs(np.transpose(recon[b]-orig[b], (1,2,0))) * 5, 0, 1)
            panels.append(err)
        panels.append(oe_np)
        panels.append(re_np)

        for col, panel in enumerate(panels):
            arr = np.clip(panel * 255, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                img_pil = Image.fromarray(arr, 'L').convert('RGB')
            else:
                img_pil = Image.fromarray(arr, 'RGB')
            img_pil = img_pil.resize((cell, cell), Image.NEAREST)
            x_off = label_w + pad + col * (cell + pad)
            canvas.paste(img_pil, (x_off, y_off))

    path = os.path.join(args.save_dir, f"recon{suffix}.png")
    canvas.save(path)
    print(f"  [可视化] {path}", flush=True)


def spectrum_analysis_simple(model, dataset, device, args):
    model.eval()
    torch.manual_seed(42)
    n = min(100, len(dataset))
    imgs = torch.stack([dataset[i][0] for i in range(n)]).to(device)
    with torch.no_grad():
        recon_imgs, _ = model.reconstruct(imgs, args.mask_ratio)
        recon_imgs = recon_imgs.clamp(0, 1)

    all_o, all_r = [], []
    for i in range(n):
        f, o = compute_radial_spectrum(imgs[i].cpu())
        _, r = compute_radial_spectrum(recon_imgs[i].cpu())
        all_o.append(o); all_r.append(r)
    avg_o = np.mean(all_o, axis=0)
    avg_r = np.mean(all_r, axis=0)
    nf = len(f)
    hi_o = avg_o[2*nf//3:].sum()
    hi_r = avg_r[2*nf//3:].sum()
    print(f"  高频恢复: {hi_r/(hi_o+1e-10)*100:.1f}%", flush=True)
    np.savez(os.path.join(args.save_dir, "spectrum_data.npz"),
             freqs=f, orig=avg_o, recon=avg_r)


# ============================================================
#  批量可视化已有模型
# ============================================================

def visualize_all_models(args):
    """加载 mae_hf_fashion 下所有模型, 批量生成多角度可视化"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_ds, img_size, in_chans, class_names = get_dataset(
        args.dataset, args.data_root, augment=False)

    hf_dir = f"mae_hf_{args.dataset}"
    if not os.path.exists(hf_dir):
        print(f"  找不到 {hf_dir}")
        return

    out_dir = os.path.join(hf_dir, "gallery")
    os.makedirs(out_dir, exist_ok=True)

    # 查找所有实验目录
    exp_dirs = sorted([d for d in os.listdir(hf_dir)
                       if os.path.isdir(os.path.join(hf_dir, d)) and d[0].isdigit()])

    print(f"  找到 {len(exp_dirs)} 个实验: {exp_dirs}")

    mcfg = MODEL_CONFIGS[args.model_size]

    # ---------- 1. 每个模型多 seed 多样本 ----------
    seeds = [42, 123, 456]
    n_per_seed = 8

    for exp_name in exp_dirs:
        ckpt_path = os.path.join(hf_dir, exp_name, 'best_mae.pt')
        if not os.path.exists(ckpt_path):
            print(f"  跳过 {exp_name} (无 checkpoint)")
            continue

        model = MAEPro(
            img_size=img_size, patch_size=args.patch_size, in_chans=in_chans,
            mask_ratio=args.mask_ratio, norm_pix_loss=False, **mcfg,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        model.eval()

        print(f"\n  ▸ {exp_name} (val={ckpt.get('loss', '?'):.5f})")

        for seed in seeds:
            _args = copy.deepcopy(args)
            _args.save_dir = out_dir
            _args.mask_ratio = args.mask_ratio
            visualize_recon(model, test_ds, device, _args, class_names, in_chans,
                            suffix=f"_{exp_name}_seed{seed}", n_samples=n_per_seed, seed=seed)

    # ---------- 2. 横向对比大图: 同样本、不同模型 ----------
    print(f"\n  生成横向对比图...")
    _make_cross_model_comparison(exp_dirs, hf_dir, test_ds, device, args,
                                  class_names, in_chans, mcfg, out_dir)

    # ---------- 3. 各类别专项对比 ----------
    print(f"\n  生成各类别专项图...")
    _make_per_class_comparison(exp_dirs, hf_dir, test_ds, device, args,
                                class_names, in_chans, mcfg, out_dir)

    print(f"\n  ✓ 全部可视化完成: {out_dir}/")


def _make_cross_model_comparison(exp_dirs, hf_dir, test_ds, device, args,
                                  class_names, in_chans, mcfg, out_dir):
    """同一组样本在不同模型下的重建效果对比"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    img_size = test_ds[0][0].shape[-1]
    scale = 3 if img_size <= 32 else 2
    cell = img_size * scale

    sample_indices = [0, 3, 5, 7, 12, 18, 25, 33]  # 8个样本
    n_samples = len(sample_indices)
    n_models = len(exp_dirs)

    torch.manual_seed(42)
    imgs = torch.stack([test_ds[i][0] for i in sample_indices]).to(device)
    labels = [test_ds[i][1] for i in sample_indices]

    # 每个模型做重建
    all_recons = {}
    for exp_name in exp_dirs:
        ckpt_path = os.path.join(hf_dir, exp_name, 'best_mae.pt')
        if not os.path.exists(ckpt_path):
            continue
        model = MAEPro(
            img_size=img_size, patch_size=args.patch_size, in_chans=in_chans,
            mask_ratio=args.mask_ratio, norm_pix_loss=False, **mcfg,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        model.eval()
        with torch.no_grad():
            recon, _ = model.reconstruct(imgs, args.mask_ratio)
            all_recons[exp_name] = recon.clamp(0, 1).cpu().numpy()

    # 绘制: 行=样本, 列=原图 + 各模型重建 + 各模型误差
    n_cols = 1 + n_models  # 原图 + 各模型
    pad = 2
    title_h = 40
    label_w = 80
    total_w = label_w + n_cols * (cell + pad) + pad
    total_h = title_h + n_samples * (cell + pad) + pad

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 列标题
    draw.text((label_w + cell // 2 - 10, 8), "原图", fill=(0, 0, 0))
    for ci, exp_name in enumerate(exp_dirs):
        if exp_name not in all_recons:
            continue
        short = exp_name.split('_', 1)[1] if '_' in exp_name else exp_name
        x = label_w + pad + (ci + 1) * (cell + pad)
        draw.text((x + 2, 8), short[:12], fill=(0, 0, 0))

    orig_np = imgs.cpu().numpy()

    for row, b in enumerate(range(n_samples)):
        y_off = title_h + pad + row * (cell + pad)
        draw.text((4, y_off + cell // 2 - 6), class_names[labels[b]], fill=(50, 50, 50))

        # 原图
        if in_chans == 1:
            arr = np.clip(orig_np[b, 0] * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, 'L').convert('RGB')
        else:
            arr = np.clip(np.transpose(orig_np[b], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, 'RGB')
        pil = pil.resize((cell, cell), Image.NEAREST)
        canvas.paste(pil, (label_w + pad, y_off))

        # 各模型重建
        for ci, exp_name in enumerate(exp_dirs):
            if exp_name not in all_recons:
                continue
            r = all_recons[exp_name][b]
            if in_chans == 1:
                arr = np.clip(r[0] * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'L').convert('RGB')
            else:
                arr = np.clip(np.transpose(r, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'RGB')
            pil = pil.resize((cell, cell), Image.NEAREST)
            x_off = label_w + pad + (ci + 1) * (cell + pad)
            canvas.paste(pil, (x_off, y_off))

    path = os.path.join(out_dir, "cross_model_recon.png")
    canvas.save(path)
    print(f"  [横向对比] {path}")

    # 同样做误差图
    canvas2 = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw2 = ImageDraw.Draw(canvas2)
    draw2.text((label_w + cell // 2 - 10, 8), "原图", fill=(0, 0, 0))
    for ci, exp_name in enumerate(exp_dirs):
        if exp_name not in all_recons:
            continue
        short = exp_name.split('_', 1)[1] if '_' in exp_name else exp_name
        x = label_w + pad + (ci + 1) * (cell + pad)
        draw2.text((x + 2, 8), short[:12], fill=(0, 0, 0))

    for row, b in enumerate(range(n_samples)):
        y_off = title_h + pad + row * (cell + pad)
        draw2.text((4, y_off + cell // 2 - 6), class_names[labels[b]], fill=(50, 50, 50))

        if in_chans == 1:
            arr = np.clip(orig_np[b, 0] * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, 'L').convert('RGB')
        else:
            arr = np.clip(np.transpose(orig_np[b], (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, 'RGB')
        pil = pil.resize((cell, cell), Image.NEAREST)
        canvas2.paste(pil, (label_w + pad, y_off))

        for ci, exp_name in enumerate(exp_dirs):
            if exp_name not in all_recons:
                continue
            r = all_recons[exp_name][b]
            if in_chans == 1:
                err = np.clip(np.abs(r[0] - orig_np[b, 0]) * 5, 0, 1)
                arr = np.clip(err * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'L').convert('RGB')
            else:
                err = np.clip(np.abs(np.transpose(r - orig_np[b], (1, 2, 0))) * 5, 0, 1)
                arr = np.clip(err * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'RGB')
            pil = pil.resize((cell, cell), Image.NEAREST)
            x_off = label_w + pad + (ci + 1) * (cell + pad)
            canvas2.paste(pil, (x_off, y_off))

    path2 = os.path.join(out_dir, "cross_model_error.png")
    canvas2.save(path2)
    print(f"  [误差对比] {path2}")


def _make_per_class_comparison(exp_dirs, hf_dir, test_ds, device, args,
                                class_names, in_chans, mcfg, out_dir):
    """每个类挑 3 个样本, 比 baseline vs combo"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    img_size = test_ds[0][0].shape[-1]
    scale = 3 if img_size <= 32 else 2
    cell = img_size * scale

    # 加载 baseline 和 combo_best 模型
    models_to_compare = ['1_mse_baseline', '4_sobel_edge', '7_combo_best']
    loaded = {}
    for name in models_to_compare:
        ckpt_path = os.path.join(hf_dir, name, 'best_mae.pt')
        if not os.path.exists(ckpt_path):
            continue
        m = MAEPro(img_size=img_size, patch_size=args.patch_size, in_chans=in_chans,
                   mask_ratio=args.mask_ratio, norm_pix_loss=False, **mcfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(ckpt['model'])
        m.eval()
        loaded[name] = m

    if len(loaded) < 2:
        return

    # 每类取 3 个样本
    class_samples = {c: [] for c in range(len(class_names))}
    for i in range(len(test_ds)):
        _, y = test_ds[i]
        if len(class_samples[y]) < 3:
            class_samples[y].append(i)

    for cls_id in range(len(class_names)):
        indices = class_samples[cls_id]
        if not indices:
            continue

        n_samples = len(indices)
        n_model_cols = len(loaded)
        # 列: 原图, model1_recon, model1_err, model2_recon, model2_err, ...
        n_cols = 1 + n_model_cols * 2
        pad = 2
        title_h = 30
        label_w = 10

        total_w = label_w + n_cols * (cell + pad) + pad
        total_h = title_h + n_samples * (cell + pad) + pad

        canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw.text((label_w + 2, 8), "原图", fill=(0, 0, 0))
        col_i = 1
        for mname in loaded:
            short = mname.split('_', 1)[1][:10]
            x = label_w + pad + col_i * (cell + pad)
            draw.text((x, 8), short, fill=(0, 0, 200))
            col_i += 1
            x = label_w + pad + col_i * (cell + pad)
            draw.text((x, 8), "err×5", fill=(200, 0, 0))
            col_i += 1

        imgs = torch.stack([test_ds[idx][0] for idx in indices]).to(device)

        for row in range(n_samples):
            y_off = title_h + pad + row * (cell + pad)

            # 原图
            o = imgs[row].cpu().numpy()
            if in_chans == 1:
                arr = np.clip(o[0] * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'L').convert('RGB')
            else:
                arr = np.clip(np.transpose(o, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                pil = Image.fromarray(arr, 'RGB')
            pil = pil.resize((cell, cell), Image.NEAREST)
            canvas.paste(pil, (label_w + pad, y_off))

            col_i = 1
            for mname, m in loaded.items():
                with torch.no_grad():
                    recon, _ = m.reconstruct(imgs[row:row+1], args.mask_ratio)
                    recon = recon.clamp(0, 1).cpu().numpy()[0]

                # 重建
                if in_chans == 1:
                    arr = np.clip(recon[0] * 255, 0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr, 'L').convert('RGB')
                else:
                    arr = np.clip(np.transpose(recon, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr, 'RGB')
                pil = pil.resize((cell, cell), Image.NEAREST)
                x_off = label_w + pad + col_i * (cell + pad)
                canvas.paste(pil, (x_off, y_off))
                col_i += 1

                # 误差
                if in_chans == 1:
                    err = np.clip(np.abs(recon[0] - o[0]) * 5, 0, 1)
                    arr = np.clip(err * 255, 0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr, 'L').convert('RGB')
                else:
                    err = np.clip(np.abs(np.transpose(recon - o, (1, 2, 0))) * 5, 0, 1)
                    arr = np.clip(err * 255, 0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr, 'RGB')
                pil = pil.resize((cell, cell), Image.NEAREST)
                x_off = label_w + pad + col_i * (cell + pad)
                canvas.paste(pil, (x_off, y_off))
                col_i += 1

        path = os.path.join(out_dir, f"class_{cls_id:02d}_{class_names[cls_id]}.png")
        canvas.save(path)

    print(f"  [类别对比] {out_dir}/class_*.png ({len(class_names)} 类)")


# ============================================================
#  动态加权 Sweep
# ============================================================

def run_dynamic_sweep(args):
    results = []
    base_dir = args.save_dir

    experiments = [
        ("1_fixed",       "fixed"),
        ("2_uncertainty", "uncertainty"),
        ("3_ema_ratio",   "ema"),
        ("4_dwa",         "dwa"),
        ("5_rlw",         "rlw"),
    ]

    # 1_fixed 本质是 MSE+FFT+Sobel combo，如果已有 7_combo_best 可复用
    hf_dir = f"mae_hf_{args.dataset}"
    combo_ckpt = os.path.join(hf_dir, '7_combo_best', 'best_mae.pt')
    fixed_dir = os.path.join(base_dir, '1_fixed')
    fixed_ckpt = os.path.join(fixed_dir, 'best_mae.pt')
    if os.path.exists(combo_ckpt) and not os.path.exists(os.path.join(fixed_dir, 'history.json')):
        os.makedirs(fixed_dir, exist_ok=True)
        import shutil
        shutil.copy2(combo_ckpt, fixed_ckpt)
        # 同时复制 history 以便断点续训
        combo_hist = os.path.join(hf_dir, '7_combo_best', 'history.json')
        if os.path.exists(combo_hist):
            shutil.copy2(combo_hist, os.path.join(fixed_dir, 'history.json'))
        print(f"  ✓ 复用 7_combo_best checkpoint → 1_fixed (80 epoch 已完成)")

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  动态 Loss 加权 Sweep: {len(experiments)} experiments   ║")
    print(f"╚══════════════════════════════════════════════╝")

    for i, (name, mode) in enumerate(experiments):
        print(f"\n{'='*62}")
        print(f"  [{i+1}/{len(experiments)}] {name}: {WEIGHT_MODES[mode]}")
        print(f"{'='*62}")

        exp_args = copy.deepcopy(args)
        exp_args.save_dir = os.path.join(base_dir, name)
        exp_args.weight_mode = mode
        exp_args.fresh = False  # 默认断点续训
        exp_args.epochs = args.sweep_epochs

        try:
            best = train_dynamic(exp_args)
            results.append({"name": name, "weight_mode": mode, "best_val_loss": best})
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"name": name, "weight_mode": mode, "error": str(e)})

    # 汇总
    print(f"\n{'='*62}")
    print(f"  动态加权 Sweep 结果")
    print(f"{'='*62}")
    print(f"  {'实验':<20} {'模式':<15} {'best_val':>12}")
    print(f"  {'-'*50}")
    for r in sorted(results, key=lambda x: x.get('best_val_loss', 999)):
        if 'best_val_loss' in r:
            print(f"  {r['name']:<20} {r['weight_mode']:<15} {r['best_val_loss']:>12.5f}")
        else:
            print(f"  {r['name']:<20} {r['weight_mode']:<15} {'FAILED':>12}")

    log_path = os.path.join(base_dir, "dynamic_sweep_results.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  保存: {log_path}")


# ============================================================
#  Main
# ============================================================

def main():
    p = argparse.ArgumentParser("MAE Dynamic Weight + Gallery")
    p.add_argument('--dataset', default='fashion', choices=['fashion', 'cifar10', 'mnist'])
    p.add_argument('--weight_mode', default='uncertainty', choices=list(WEIGHT_MODES.keys()))
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
    p.add_argument('--patience', type=int, default=15,
                   help='Early stopping patience (0=disabled)')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--fresh', action='store_true')
    # Modes
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--sweep_epochs', type=int, default=1000)
    p.add_argument('--visualize_all', action='store_true',
                   help='批量可视化已有模型 (不训练)')
    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = f"mae_dynamic_{args.dataset}"

    if args.visualize_all:
        visualize_all_models(args)
    elif args.sweep:
        run_dynamic_sweep(args)
    else:
        train_dynamic(args)


if __name__ == "__main__":
    main()
