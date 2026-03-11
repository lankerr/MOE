"""
集成运行脚本: 自动检测 exp7-12 完成后运行 baseline 对比
========================================================

功能:
1. 检测 exp7-12 是否都完成训练
2. 如果完成，自动运行 baseline (train_384) 使用相同的 L1+MSE loss
3. 推理对比 CSI/HSS/PDD 等指标

用法:
    # 仅检查状态
    python run_baseline_comparison.py --check_only

    # 运行 baseline (如果 exp7-12 已完成)
    python run_baseline_comparison.py --run_baseline

    # 运行推理对比
    python run_baseline_comparison.py --evaluate

    # 全流程: 检查 -> 训练 baseline -> 评估对比
    python run_baseline_comparison.py --all
"""

import os
import sys
import json
import argparse
import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from config import cfg
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset

# 加速设置
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


# ===================== 实验配置 =====================

FLASH_EXPERIMENTS = [
    "exp7_moe_flash",
    "exp8_swiglu_moe_flash",
    "exp9_balanced_moe_flash",
    "exp10_moe_rope_flash",
    "exp11_swiglu_moe_rope_flash",
    "exp12_balanced_moe_rope_flash",
]

BASELINE_NAME = "baseline_384_l1mse"  # 与 exp7-12 使用相同的 loss


# ===================== 状态检测 =====================

def check_experiment_status(ckpt_dir="./checkpoints"):
    """检查所有实验的完成状态"""
    status = {}

    for exp_name in FLASH_EXPERIMENTS + [BASELINE_NAME]:
        exp_dir = os.path.join(ckpt_dir, exp_name)
        log_path = os.path.join(exp_dir, "training_log.json")
        best_path = os.path.join(exp_dir, "best_model.pt")

        exp_status = {
            "has_checkpoint": os.path.exists(best_path),
            "has_log": os.path.exists(log_path),
            "epochs_done": 0,
            "best_val_loss": float('inf'),
            "is_complete": False,
        }

        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
                exp_status["epochs_done"] = len(log)
                if log:
                    exp_status["best_val_loss"] = min(e.get("val_loss", float('inf')) for e in log)
                    # 假设 100 epochs 为完成
                    exp_status["is_complete"] = exp_status["epochs_done"] >= 10  # 至少 10 epochs
            except:
                pass

        status[exp_name] = exp_status

    return status


def print_status_report(status):
    """打印状态报告"""
    print("\n" + "=" * 70)
    print("实验状态报告")
    print("=" * 70)

    print("\n[Flash 实验 exp7-12]")
    all_flash_complete = True
    for exp_name in FLASH_EXPERIMENTS:
        s = status[exp_name]
        complete_mark = "✓" if s["is_complete"] else "✗"
        print(f"  {complete_mark} {exp_name:35s} | Epochs: {s['epochs_done']:3d} | "
              f"Best Val: {s['best_val_loss']:.4f}")
        if not s["is_complete"]:
            all_flash_complete = False

    print(f"\n[Baseline: {BASELINE_NAME}]")
    s = status[BASELINE_NAME]
    complete_mark = "✓" if s["is_complete"] else "✗"
    print(f"  {complete_mark} {BASELINE_NAME:35s} | Epochs: {s['epochs_done']:3d} | "
          f"Best Val: {s['best_val_loss']:.4f}")

    print("\n" + "-" * 70)
    if all_flash_complete:
        print("所有 Flash 实验已完成!")
        if status[BASELINE_NAME]["is_complete"]:
            print("Baseline 也已完成，可以运行评估对比。")
        else:
            print("Baseline 尚未完成，建议运行 baseline 训练。")
    else:
        print("部分 Flash 实验未完成，请先完成 exp7-12 的训练。")
    print("=" * 70)

    return all_flash_complete


# ===================== Baseline 训练 (L1+MSE Loss) =====================

def train_baseline(args):
    """训练 baseline 模型，使用与 exp7-12 相同的 L1+MSE loss"""

    print("\n" + "#" * 70)
    print("# Baseline 训练 (train_384 配置 + L1+MSE Loss)")
    print("#" * 70)
    print(f"# 与 exp7-12 使用相同的 loss 函数: L1 + MSE")
    print("#" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据集
    print("\n加载数据集...")
    sevir_paths = cfg.get_sevir_paths()

    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 6, 13),
        end_date=datetime.datetime(2017, 8, 15),
        shuffle=True,
        verbose=True
    )

    val_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 8, 15),
        end_date=datetime.datetime(2017, 9, 15),
        shuffle=False,
        verbose=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # 模型 - 与 train_384.py 完全相同的配置
    model_args = argparse.Namespace(
        input_img_size=384,
        patch_size=4,
        input_channels=1,
        embed_dim=64,  # 与 train_384 相同
        depths_down=[2, 2],
        depths_up=[2, 2],
        heads_number=[4, 4],
        window_size=4,
        out_len=args.output_frames
    )

    model = Memory(
        model_args,
        memory_channel_size=256,  # 与 train_384 相同
        short_len=args.input_frames,
        long_len=args.seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP
    scaler = GradScaler('cuda') if args.use_amp else None

    # 保存目录
    save_dir = os.path.join(args.checkpoint_dir, BASELINE_NAME)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.json')

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    training_log = []

    latest_path = os.path.join(save_dir, 'latest_model.pt')
    if os.path.exists(latest_path):
        print(f"\n从 {latest_path} 恢复训练...")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))

    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                training_log = json.load(f)
        except:
            training_log = []

    # 训练循环
    print(f"\n开始训练 (Max {args.epochs} epochs)...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)

            x = batch[:, :args.input_frames]
            memory_seq = x.repeat(1, 3, 1, 1, 1)[:, :args.seq_len]
            target = batch[:, args.input_frames:args.input_frames + args.output_frames]

            with autocast('cuda', enabled=args.use_amp):
                output = model(x, memory_seq, phase=2)
                if isinstance(output, list):
                    output = torch.stack(output, dim=1)
                output = output[:, -args.output_frames:]

                # ===== 关键: 与 exp7-12 相同的 L1+MSE Loss =====
                loss = F.l1_loss(output, target) + F.mse_loss(output, target)
                loss = loss / args.accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * args.accumulation_steps

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item() * args.accumulation_steps:.4f}")

        train_loss /= len(train_loader)
        epoch_time = time.time() - t0

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                x = batch[:, :args.input_frames]
                memory_seq = x.repeat(1, 3, 1, 1, 1)[:, :args.seq_len]
                target = batch[:, args.input_frames:args.input_frames + args.output_frames]

                with autocast('cuda', enabled=args.use_amp):
                    output = model(x, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    output = output[:, -args.output_frames:]
                    loss = F.l1_loss(output, target) + F.mse_loss(output, target)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Time: {epoch_time:.0f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        training_log.append(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  ★ Best model! Val Loss: {val_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pt'))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(save_dir, 'latest_model.pt'))

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Baseline 训练完成! Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70)

    return best_val_loss


# ===================== 评估指标 =====================

def compute_csi(pred, target, threshold):
    """计算 CSI (Critical Success Index)"""
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    csi = tp / (tp + fp + fn + 1e-8)
    return csi.item()


def compute_hss(pred, target, threshold):
    """计算 HSS (Heidke Skill Score)"""
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()

    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-8

    hss = numerator / denominator
    return hss.item()


def compute_pod(pred, target, threshold):
    """计算 POD (Probability of Detection)"""
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    pod = tp / (tp + fn + 1e-8)
    return pod.item()


def compute_far(pred, target, threshold):
    """计算 FAR (False Alarm Rate)"""
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    far = fp / (tp + fp + 1e-8)
    return far.item()


# ===================== 评估对比 =====================

def evaluate_all_models(args):
    """评估所有模型并对比指标"""

    print("\n" + "=" * 70)
    print("模型评估对比")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载验证数据
    sevir_paths = cfg.get_sevir_paths()
    val_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 8, 15),
        end_date=datetime.datetime(2017, 9, 15),
        shuffle=False,
        verbose=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # 阈值 (雷达反射率 dBZ)
    thresholds = [16, 30, 40, 50]  # 对应不同降水强度

    # 评估所有模型
    all_results = {}
    models_to_eval = FLASH_EXPERIMENTS + [BASELINE_NAME]

    for exp_name in models_to_eval:
        ckpt_path = os.path.join(args.checkpoint_dir, exp_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"跳过 {exp_name}: 未找到检查点")
            continue

        print(f"\n评估 {exp_name}...")

        # 加载模型
        model_args = argparse.Namespace(
            input_img_size=384, patch_size=4, input_channels=1,
            embed_dim=64, depths_down=[2, 2], depths_up=[2, 2],
            heads_number=[4, 4], window_size=4, out_len=args.output_frames
        )

        model = Memory(model_args, memory_channel_size=256,
                       short_len=args.input_frames, long_len=args.seq_len)

        # 如果是 exp7-12，需要应用实验配置
        if exp_name in FLASH_EXPERIMENTS:
            from experiments.experiment_factory import EXPERIMENTS, apply_experiment
            exp_config = EXPERIMENTS[exp_name]
            model = apply_experiment(model, exp_config)

        model = model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # 评估
        metrics = {f'CSI_{t}': [] for t in thresholds}
        metrics.update({f'HSS_{t}': [] for t in thresholds})
        metrics.update({f'POD_{t}': [] for t in thresholds})
        metrics.update({f'FAR_{t}': [] for t in thresholds})
        metrics['MSE'] = []
        metrics['L1'] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                x = batch[:, :args.input_frames]
                memory_seq = x.repeat(1, 3, 1, 1, 1)[:, :args.seq_len]
                target = batch[:, args.input_frames:args.input_frames + args.output_frames]

                with autocast('cuda', enabled=args.use_amp):
                    output = model(x, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    output = output[:, -args.output_frames:]

                # 计算指标
                metrics['MSE'].append(F.mse_loss(output, target).item())
                metrics['L1'].append(F.l1_loss(output, target).item())

                for t in thresholds:
                    metrics[f'CSI_{t}'].append(compute_csi(output, target, t))
                    metrics[f'HSS_{t}'].append(compute_hss(output, target, t))
                    metrics[f'POD_{t}'].append(compute_pod(output, target, t))
                    metrics[f'FAR_{t}'].append(compute_far(output, target, t))

        # 汇总
        result = {k: np.mean(v) for k, v in metrics.items()}
        all_results[exp_name] = result

        print(f"  MSE: {result['MSE']:.4f} | L1: {result['L1']:.4f}")
        print(f"  CSI@30: {result['CSI_30']:.4f} | HSS@30: {result['HSS_30']:.4f}")

        del model
        torch.cuda.empty_cache()

    # 打印对比表
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)

    # 按平均 CSI 排序
    def avg_csi(exp_name):
        r = all_results[exp_name]
        return np.mean([r[f'CSI_{t}'] for t in [16, 30, 40, 50]])

    sorted_results = sorted(all_results.items(), key=lambda x: avg_csi(x[0]), reverse=True)

    print(f"\n{'模型':<35} | MSE    | L1     | CSI@16 | CSI@30 | CSI@40 | CSI@50")
    print("-" * 90)
    for exp_name, r in sorted_results:
        is_baseline = exp_name == BASELINE_NAME
        marker = "[B]" if is_baseline else "   "
        print(f"{marker} {exp_name:<32} | {r['MSE']:.4f} | {r['L1']:.4f} | "
              f"{r['CSI_16']:.4f} | {r['CSI_30']:.4f} | {r['CSI_40']:.4f} | {r['CSI_50']:.4f}")

    # 保存结果
    results_path = os.path.join(args.checkpoint_dir, "evaluation_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {results_path}")

    return all_results


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description='Baseline 对比集成脚本')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--check_only', action='store_true', help='仅检查状态')
    parser.add_argument('--run_baseline', action='store_true', help='运行 baseline 训练')
    parser.add_argument('--evaluate', action='store_true', help='运行评估对比')
    parser.add_argument('--all', action='store_true', help='全流程')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=8)
    parser.add_argument('--output_frames', type=int, default=12)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()

    print("=" * 70)
    print("DATSwinLSTM Baseline 对比集成脚本")
    print("=" * 70)

    # 检查状态
    status = check_experiment_status(args.checkpoint_dir)
    all_flash_complete = print_status_report(status)

    if args.check_only:
        return

    # 运行 baseline
    if args.run_baseline or args.all:
        if all_flash_complete:
            train_baseline(args)
        else:
            print("\n警告: 部分 Flash 实验未完成，仍可运行 baseline 进行对比")

    # 评估
    if args.evaluate or args.all:
        evaluate_all_models(args)


if __name__ == '__main__':
    main()
