"""
绘制所有 Earthformer 实验的 loss 随 epoch 变化曲线
- train_loss = MAE + MSE (per-step, 聚合为 per-epoch 平均)
- valid_loss = valid_frame_mae_epoch + valid_frame_mse_epoch (NOT valid_loss_epoch, 那个是 -CSI)
- 自动过滤 NaN 数据
- 跳过无效/临时实验
"""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "loss_curves.png")

# 只绘制正式实验，跳过 tmp/old/early run
LABEL_MAP = {
    "exp_earthformer_baseline": "20f baseline",
    "exp_earthformer_exp1_moe_flash": "20f exp1 (MoE+Flash)",
    "exp_earthformer_exp2_swiglu_moe_flash": "20f exp2 (SwiGLU+MoE)",
    "exp_earthformer_exp3_balanced_moe_flash": "20f exp3 (Balanced MoE)",
    "exp_earthformer_exp4_moe_rope_flash": "20f exp4 (MoE+RoPE)",
    "exp_earthformer_exp5_swiglu_moe_rope_flash": "20f exp5 (SwiGLU+MoE+RoPE)",
    "exp_earthformer_exp6_balanced_moe_rope_flash": "20f exp6 (Balanced+RoPE)",
    "exp_earthformer_49f_baseline": "49f baseline",
    "exp_earthformer_49f_exp1_moe_flash": "49f exp1 (MoE+Flash)",
    "exp_earthformer_49f_exp2_swiglu_moe_flash": "49f exp2 (SwiGLU+MoE)",
    "exp_earthformer_49f_exp3_balanced_moe_flash": "49f exp3 (Balanced MoE)",
    "exp_earthformer_49f_exp4_moe_rope_flash": "49f exp4 (MoE+RoPE)",
    "exp_earthformer_49f_exp5_swiglu_moe_rope_flash": "49f exp5 (SwiGLU+MoE+RoPE)",
    "exp_earthformer_49f_exp6_balanced_moe_rope_flash": "49f exp6 (Balanced+RoPE)",
}


def _safe_float(s):
    """安全转换为 float，NaN/Inf/空值返回 None"""
    if not s:
        return None
    try:
        v = float(s)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, OverflowError):
        return None


def find_best_metrics_csv(exp_dir):
    """找到数据量最大的 metrics.csv（通常是最完整的训练 run）"""
    logs_dir = os.path.join(exp_dir, "lightning_logs")
    if not os.path.isdir(logs_dir):
        return None

    best_file = None
    best_lines = 0
    for ver_name in os.listdir(logs_dir):
        csv_path = os.path.join(logs_dir, ver_name, "metrics.csv")
        if os.path.isfile(csv_path):
            nlines = sum(1 for _ in open(csv_path, encoding='utf-8'))
            if nlines > best_lines:
                best_lines = nlines
                best_file = csv_path
    return best_file


def parse_metrics(csv_path):
    """
    从 metrics.csv 提取:
      - train_loss: per-step → 聚合为 per-epoch 平均 (= MAE + MSE)
      - valid_mae + valid_mse: per-epoch (从 valid_frame_mae_epoch + valid_frame_mse_epoch)
    注意: valid_loss_epoch 实际是 -CSI，不是 MAE+MSE！
    """
    train_by_epoch = defaultdict(list)
    valid_mae_by_epoch = {}
    valid_mse_by_epoch = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        has_train = 'train_loss' in cols
        has_valid_mae = 'valid_frame_mae_epoch' in cols
        has_valid_mse = 'valid_frame_mse_epoch' in cols

        for row in reader:
            epoch_str = row.get('epoch', '').strip()
            if not epoch_str:
                continue
            try:
                ep = int(epoch_str)
            except ValueError:
                continue

            # train_loss per step
            if has_train:
                tl = _safe_float(row.get('train_loss', '').strip())
                if tl is not None:
                    train_by_epoch[ep].append(tl)

            # valid MAE / MSE per epoch
            if has_valid_mae:
                vmae = _safe_float(row.get('valid_frame_mae_epoch', '').strip())
                if vmae is not None:
                    valid_mae_by_epoch[ep] = vmae
            if has_valid_mse:
                vmse = _safe_float(row.get('valid_frame_mse_epoch', '').strip())
                if vmse is not None:
                    valid_mse_by_epoch[ep] = vmse

    # 聚合 train_loss → per-epoch 平均
    train_epochs = sorted(train_by_epoch.keys())
    train_loss = [np.mean(train_by_epoch[e]) for e in train_epochs]

    # valid: MAE + MSE 合成 (与 train_loss 口径一致)
    valid_epochs_set = set(valid_mae_by_epoch.keys()) & set(valid_mse_by_epoch.keys())
    valid_epochs = sorted(valid_epochs_set)
    valid_mae = [valid_mae_by_epoch[e] for e in valid_epochs]
    valid_mse = [valid_mse_by_epoch[e] for e in valid_epochs]
    valid_loss = [valid_mae_by_epoch[e] + valid_mse_by_epoch[e] for e in valid_epochs]

    # 过滤掉含 NaN 的 epoch（整个 epoch 丢弃）
    clean_train_ep, clean_train_loss = [], []
    for e, l in zip(train_epochs, train_loss):
        if not np.isnan(l):
            clean_train_ep.append(e)
            clean_train_loss.append(l)

    clean_valid_ep, clean_valid_loss, clean_valid_mae, clean_valid_mse = [], [], [], []
    for e, l, m, s in zip(valid_epochs, valid_loss, valid_mae, valid_mse):
        if not np.isnan(l):
            clean_valid_ep.append(e)
            clean_valid_loss.append(l)
            clean_valid_mae.append(m)
            clean_valid_mse.append(s)

    return {
        'train_epoch': clean_train_ep, 'train_loss': clean_train_loss,
        'valid_epoch': clean_valid_ep, 'valid_loss': clean_valid_loss,
        'valid_mae': clean_valid_mae, 'valid_mse': clean_valid_mse,
    }


def main():
    # 只扫描正式实验
    all_data = {}
    for exp_name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if exp_name not in LABEL_MAP:
            continue
        exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        csv_path = find_best_metrics_csv(exp_dir)
        if csv_path is None:
            print(f"[SKIP] {exp_name}: no metrics.csv")
            continue
        data = parse_metrics(csv_path)
        # 至少需要 2 个 epoch 数据才有画图意义
        if len(data['train_epoch']) < 2 and len(data['valid_epoch']) < 2:
            print(f"[SKIP] {exp_name}: <2 epochs data (train={len(data['train_epoch'])}, valid={len(data['valid_epoch'])})")
            continue
        label = LABEL_MAP[exp_name]
        all_data[label] = data
        print(f"[OK]   {label:<30} train_ep={len(data['train_epoch']):>3}, valid_ep={len(data['valid_epoch']):>3}, "
              f"final_train={data['train_loss'][-1]:.6f}" + (f", final_valid_mae+mse={data['valid_loss'][-1]:.6f}" if data['valid_loss'] else ""))

    if not all_data:
        print("No valid data found!")
        return

    # -------- 分组 --------
    data_20f = {k: v for k, v in all_data.items() if k.startswith("20f")}
    data_49f = {k: v for k, v in all_data.items() if k.startswith("49f")}

    groups = []
    if data_20f:
        groups.append(("20-Frame Experiments", data_20f))
    if data_49f:
        groups.append(("49-Frame Experiments", data_49f))

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 3, figsize=(20, 5.5 * n_groups), squeeze=False)
    fig.suptitle("Earthformer Loss Curves — Train: MAE+MSE | Valid: MAE+MSE",
                 fontsize=15, fontweight='bold', y=0.98)

    colors = plt.cm.tab10.colors

    for g_idx, (group_title, group_data) in enumerate(groups):
        ax_train = axes[g_idx, 0]
        ax_valid = axes[g_idx, 1]
        ax_detail = axes[g_idx, 2]

        for i, (label, data) in enumerate(sorted(group_data.items())):
            c = colors[i % len(colors)]
            short = label.split(' ', 1)[1]  # 去掉 "20f "/"49f " 前缀

            if data['train_epoch']:
                ax_train.plot(data['train_epoch'], data['train_loss'],
                              label=short, color=c, linewidth=1.8, marker='o', markersize=3)
            if data['valid_epoch']:
                ax_valid.plot(data['valid_epoch'], data['valid_loss'],
                              label=short, color=c, linewidth=1.8, marker='s', markersize=3)
                # 分开画 MAE 和 MSE
                ax_detail.plot(data['valid_epoch'], data['valid_mae'],
                               label=f"{short} MAE", color=c, linewidth=1.5, linestyle='-', marker='o', markersize=2)
                ax_detail.plot(data['valid_epoch'], data['valid_mse'],
                               label=f"{short} MSE", color=c, linewidth=1.2, linestyle='--', marker='x', markersize=2)

        ax_train.set_title(f"{group_title} — Train Loss (MAE+MSE, epoch avg)")
        ax_train.set_xlabel("Epoch")
        ax_train.set_ylabel("Loss")
        ax_train.legend(fontsize=8)
        ax_train.grid(True, alpha=0.3)

        ax_valid.set_title(f"{group_title} — Valid Loss (MAE+MSE)")
        ax_valid.set_xlabel("Epoch")
        ax_valid.set_ylabel("Loss")
        ax_valid.legend(fontsize=8)
        ax_valid.grid(True, alpha=0.3)

        ax_detail.set_title(f"{group_title} — Valid MAE & MSE (separate)")
        ax_detail.set_xlabel("Epoch")
        ax_detail.set_ylabel("Metric Value")
        ax_detail.legend(fontsize=7, ncol=2)
        ax_detail.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\n=== 图像已保存: {OUTPUT_PATH} ===")

    # 汇总表
    print("\n=== 各实验最终指标汇总 ===")
    print(f"{'实验':<35} {'Epochs':>7} {'Train MAE+MSE':>14} {'Valid MAE':>12} {'Valid MSE':>12} {'Valid Total':>12}")
    print("-" * 96)
    for label in sorted(all_data.keys()):
        d = all_data[label]
        n_ep = max(len(d['train_epoch']), len(d['valid_epoch']))
        tl = f"{d['train_loss'][-1]:.6f}" if d['train_loss'] else "N/A"
        vmae = f"{d['valid_mae'][-1]:.6f}" if d['valid_mae'] else "N/A"
        vmse = f"{d['valid_mse'][-1]:.6f}" if d['valid_mse'] else "N/A"
        vl = f"{d['valid_loss'][-1]:.6f}" if d['valid_loss'] else "N/A"
        print(f"{label:<35} {n_ep:>7} {tl:>14} {vmae:>12} {vmse:>12} {vl:>12}")


if __name__ == "__main__":
    main()
