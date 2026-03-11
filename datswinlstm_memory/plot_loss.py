"""
DATSwinLSTM-Memory 384x384 训练损失曲线
从 checkpoints 和已知终端日志数据中提取 train/val loss
"""
import torch
import os
import re
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import numpy as np

# ========== 从检查点提取 val_loss ==========
checkpoint_dir = './checkpoints/384x384'
checkpoint_val = {}

print("=" * 50)
print("从检查点提取 val_loss...")
print("=" * 50)

for fname in sorted(os.listdir(checkpoint_dir)):
    if fname.endswith('.pth'):
        fpath = os.path.join(checkpoint_dir, fname)
        ckpt = torch.load(fpath, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', None)
        val_loss = ckpt.get('val_loss', None)
        if epoch is not None and val_loss is not None:
            checkpoint_val[epoch] = val_loss
            print(f"  {fname}: epoch={epoch}, val_loss={val_loss:.6f}")

# ========== 已知的终端输出数据（从多次 run 观察到的 epoch summary） ==========
# 格式: Epoch [X/50] Train: XXXX Val: XXXX
# 这些数据来自多次 command_status 输出

known_data = {
    # epoch: (train_loss, val_loss)
    # 从终端日志中确认的最后一行
    50: (0.0057, 0.0046),
}

# 用 checkpoint val_loss 补充 val_losses
val_losses = {}
train_losses = {}

for epoch, val_loss in checkpoint_val.items():
    val_losses[epoch] = val_loss

for epoch, (train_loss, val_loss) in known_data.items():
    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss

print(f"\nVal loss 数据点: {len(val_losses)}")
print(f"Train loss 数据点: {len(train_losses)}")

# ========== 绘图 ==========
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# 深色主题
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

# 排序数据
epochs_val = sorted(val_losses.keys())
losses_val = [val_losses[e] for e in epochs_val]

# 绘制 Val Loss 曲线
ax.plot(epochs_val, losses_val, 
        color='#ff6b6b', linewidth=2.8, marker='D', markersize=8,
        label='Validation Loss (MSE)', alpha=0.95, zorder=4,
        markerfacecolor='#ff6b6b', markeredgecolor='white', markeredgewidth=1.5)

# 绘制 Train Loss 点（如果有）
if train_losses:
    epochs_train = sorted(train_losses.keys())
    losses_train = [train_losses[e] for e in epochs_train]
    ax.scatter(epochs_train, losses_train,
               color='#58a6ff', s=120, zorder=5, 
               edgecolors='white', linewidth=1.5,
               label='Train Loss (MSE)', alpha=0.95)

# 标注最佳点
if val_losses:
    best_epoch = min(val_losses, key=val_losses.get)
    best_val = val_losses[best_epoch]
    ax.scatter([best_epoch], [best_val], 
               color='#ffd93d', s=200, zorder=6, 
               edgecolors='white', linewidth=2.5, marker='*')
    ax.annotate(f'Best Val Loss: {best_val:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_val),
                xytext=(best_epoch - 8, best_val + 0.001),
                fontsize=12, color='#ffd93d', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', 
                         edgecolor='#ffd93d', alpha=0.8))

# 添加数值标注
for epoch, val in zip(epochs_val, losses_val):
    ax.annotate(f'{val:.4f}', 
                xy=(epoch, val), xytext=(0, 12),
                textcoords='offset points', ha='center',
                fontsize=9, color='#c9d1d9', alpha=0.85)

# 美化
ax.set_xlabel('Epoch', fontsize=15, color='#c9d1d9', fontweight='bold', labelpad=10)
ax.set_ylabel('Loss (MSE)', fontsize=15, color='#c9d1d9', fontweight='bold', labelpad=10)
ax.set_title('DATSwinLSTM-Memory 384×384  Training Loss Curve', 
             fontsize=18, color='white', fontweight='bold', pad=20)

# 副标题
ax.text(0.5, 1.02, 'RTX 5070 Laptop (8GB)  |  embed_dim=64  |  accum_steps=4  |  50 Epochs',
        transform=ax.transAxes, ha='center', fontsize=11, color='#8b949e',
        style='italic')

ax.legend(fontsize=12, loc='upper right', 
          fancybox=True, framealpha=0.7, 
          edgecolor='#30363d', facecolor='#21262d',
          labelcolor='#c9d1d9')

ax.grid(True, alpha=0.1, color='#c9d1d9', linestyle='--')
ax.tick_params(colors='#8b949e', labelsize=11)
ax.set_xlim(0, 52)

for spine in ax.spines.values():
    spine.set_color('#30363d')
    spine.set_linewidth(1.5)

plt.tight_layout()
save_path = './checkpoints/384x384/loss_curve.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nLoss curve saved to: {save_path}")

# 打印摘要
print("\n" + "=" * 50)
print("训练摘要")
print("=" * 50)
if val_losses:
    best_e = min(val_losses, key=val_losses.get)
    print(f"  最佳验证损失: {val_losses[best_e]:.6f} (Epoch {best_e})")
    print(f"  最终验证损失: {losses_val[-1]:.6f} (Epoch {epochs_val[-1]})")
if train_losses:
    print(f"  最终训练损失: {train_losses[max(train_losses.keys())]:.6f}")
print(f"  总 Epochs: {max(epochs_val)}")
print(f"  GPU 峰值显存: ~5.47 GB / 8.55 GB")
