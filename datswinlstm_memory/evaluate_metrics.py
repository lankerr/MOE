"""
完整评估脚本: 计算所有模型的 CSI/HSS/POD/FAR 指标
================================================

评估 exp7-12 (Flash 实验) 的降水预测性能
在 4 个阈值 (16/30/40/50 dBZ) 上计算指标

用法:
    python evaluate_metrics.py
    python evaluate_metrics.py --max_batches 50  # 快速测试
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
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np

from config import cfg
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset

# 加速设置
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


# ===================== 评估指标函数 =====================

def compute_csi(pred, target, threshold):
    """
    CSI (Critical Success Index) = TP / (TP + FP + FN)
    也称为 Threat Score (TS)
    衡量命中预测的准确率
    """
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    csi = tp / (tp + fp + fn + 1e-8)
    return csi.item()


def compute_hss(pred, target, threshold):
    """
    HSS (Heidke Skill Score)
    衡量相对于随机预测的改进程度
    HSS > 0 表示优于随机预测
    """
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
    """
    POD (Probability of Detection) = TP / (TP + FN)
    也称为 Hit Rate
    衡量实际发生的事件中被正确预测的比例
    """
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    pod = tp / (tp + fn + 1e-8)
    return pod.item()


def compute_far(pred, target, threshold):
    """
    FAR (False Alarm Rate) = FP / (TP + FP)
    衡量预测为发生但实际未发生的比例
    越低越好
    """
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    far = fp / (tp + fp + 1e-8)
    return far.item()


def compute_all_metrics(pred, target, thresholds, threshold_names):
    """计算所有指标"""
    metrics = {}
    for t, name in zip(thresholds, threshold_names):
        metrics[f'CSI_{name}'] = compute_csi(pred, target, t)
        metrics[f'HSS_{name}'] = compute_hss(pred, target, t)
        metrics[f'POD_{name}'] = compute_pod(pred, target, t)
        metrics[f'FAR_{name}'] = compute_far(pred, target, t)
    return metrics


# ===================== 模型加载 =====================

def load_model(exp_name, device, args):
    """加载指定实验的模型"""

    model_args = argparse.Namespace(
        input_img_size=384,
        patch_size=4,
        input_channels=1,
        embed_dim=64,
        depths_down=[2, 2],
        depths_up=[2, 2],
        heads_number=[4, 4],
        window_size=4,
        out_len=args.output_frames
    )

    model = Memory(
        model_args,
        memory_channel_size=256,
        short_len=args.input_frames,
        long_len=args.seq_len
    )

    # 应用实验配置 (如果是 exp7-12)
    if exp_name.startswith('exp'):
        from experiments.experiment_factory import EXPERIMENTS, apply_experiment
        exp_config = EXPERIMENTS[exp_name]
        model = apply_experiment(model, exp_config)

    model = model.to(device)

    # 加载权重
    ckpt_path = os.path.join(args.checkpoint_dir, exp_name, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  加载: {ckpt_path}")
    else:
        print(f"  警告: 未找到 {ckpt_path}")
        return None

    return model


# ===================== 主评估函数 =====================

def evaluate_model(model, val_loader, device, args, thresholds, threshold_names):
    """评估单个模型"""

    model.eval()

    # 累积指标
    all_metrics = {f'{m}_{name}': [] for m in ['CSI', 'HSS', 'POD', 'FAR'] for name in threshold_names}
    all_metrics['MSE'] = []
    all_metrics['L1'] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device, non_blocking=True)

            x = batch[:, :args.input_frames]
            memory_seq = x.repeat(1, 3, 1, 1, 1)[:, :args.seq_len]
            target = batch[:, args.input_frames:args.input_frames + args.output_frames]

            with autocast('cuda', dtype=torch.bfloat16):
                output = model(x, memory_seq, phase=2)
                if isinstance(output, list):
                    output = torch.stack(output, dim=1)
                output = output[:, -args.output_frames:]

            # 计算基础损失
            all_metrics['MSE'].append(F.mse_loss(output, target).item())
            all_metrics['L1'].append(F.l1_loss(output, target).item())

            # 计算各项指标
            metrics = compute_all_metrics(output, target, thresholds, threshold_names)
            for k, v in metrics.items():
                all_metrics[k].append(v)

            if (batch_idx + 1) % 20 == 0:
                print(f"    Batch {batch_idx+1}/{len(val_loader)}")

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break

    # 汇总
    result = {k: np.mean(v) for k, v in all_metrics.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description='评估 exp7-12 的降水预测指标')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--max_batches', type=int, default=0, help='0=全部评估')

    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=8)
    parser.add_argument('--output_frames', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--preset', type=str, default='20frame',
                        choices=['20frame', '36frame', '49frame'],
                        help='使用预设配置 (自动设置 seq_len/input_frames/output_frames)')

    args = parser.parse_args()

    # 应用预设配置
    if args.preset == '20frame':
        args.seq_len = 20
        args.input_frames = 8
        args.output_frames = 12
    elif args.preset == '36frame':
        args.seq_len = 36        # 3小时序列
        args.input_frames = 12   # 1小时输入
        args.output_frames = 12  # 1小时输出
    elif args.preset == '49frame':
        args.seq_len = 49        # 完整 SEVIR (4小时5分)
        args.input_frames = 12   # 1小时输入 (与原论文一致)
        args.output_frames = 12  # 1小时输出 (与原论文一致)

    # 阈值 - 数据已归一化到 0-1 范围 (data / 255.0)
    # SEVIR VIL 原始值 0-255 对应约 -10 到 60 dBZ
    # 转换: normalized = (dBZ + 10) / 70 * 255 / 255 = (dBZ + 10) / 70
    # 16 dBZ → 0.37, 30 dBZ → 0.57, 40 dBZ → 0.71, 50 dBZ → 0.86
    thresholds = [0.37, 0.57, 0.71, 0.86]  # 归一化后的阈值
    threshold_names = ["16dBZ", "30dBZ", "40dBZ", "50dBZ"]

    print("=" * 80)
    print("DATSwinLSTM 模型评估 - CSI/HSS/POD/FAR 指标")
    print("=" * 80)
    print(f"阈值 (归一化): {thresholds}")
    print(f"对应 dBZ: 16=毛毛雨, 30=小雨, 40=中雨, 50=暴雨")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载验证数据
    print("\n加载验证数据集...")
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"验证样本数: {len(val_dataset)}")

    # 评估所有 Flash 实验
    flash_experiments = [
        "exp7_moe_flash",
        "exp8_swiglu_moe_flash",
        "exp9_balanced_moe_flash",
        "exp10_moe_rope_flash",
        "exp11_swiglu_moe_rope_flash",
        "exp12_balanced_moe_rope_flash",
    ]

    all_results = {}

    for exp_name in flash_experiments:
        print(f"\n{'='*60}")
        print(f"评估: {exp_name}")
        print(f"{'='*60}")

        model = load_model(exp_name, device, args)
        if model is None:
            continue

        t0 = time.time()
        result = evaluate_model(model, val_loader, device, args, thresholds, threshold_names)
        elapsed = time.time() - t0

        all_results[exp_name] = result

        print(f"\n  结果 (耗时 {elapsed:.1f}s):")
        print(f"    MSE: {result['MSE']:.6f} | L1: {result['L1']:.6f}")
        for name in threshold_names:
            print(f"    @{name}: CSI={result[f'CSI_{name}']:.4f} HSS={result[f'HSS_{name}']:.4f} "
                  f"POD={result[f'POD_{name}']:.4f} FAR={result[f'FAR_{name}']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ===================== 打印汇总表格 =====================
    print("\n" + "=" * 100)
    print("评估结果汇总表")
    print("=" * 100)

    # 按 30dBZ 的 CSI 排序 (最常用的评估阈值)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['CSI_30dBZ'],
        reverse=True
    )

    # 表头
    print(f"\n{'模型':<30} | {'MSE':>8} | {'L1':>8} | "
          f"{'CSI@16':>8} | {'CSI@30':>8} | {'CSI@40':>8} | {'CSI@50':>8}")
    print("-" * 100)

    for exp_name, r in sorted_results:
        print(f"{exp_name:<30} | {r['MSE']:>8.4f} | {r['L1']:>8.4f} | "
              f"{r['CSI_16dBZ']:>8.4f} | {r['CSI_30dBZ']:>8.4f} | {r['CSI_40dBZ']:>8.4f} | {r['CSI_50dBZ']:>8.4f}")

    # HSS 表
    print(f"\n{'模型':<30} | "
          f"{'HSS@16':>8} | {'HSS@30':>8} | {'HSS@40':>8} | {'HSS@50':>8}")
    print("-" * 80)

    for exp_name, r in sorted_results:
        print(f"{exp_name:<30} | "
              f"{r['HSS_16dBZ']:>8.4f} | {r['HSS_30dBZ']:>8.4f} | {r['HSS_40dBZ']:>8.4f} | {r['HSS_50dBZ']:>8.4f}")

    # POD/FAR 表
    print(f"\n{'模型':<30} | "
          f"{'POD@30':>8} | {'FAR@30':>8} | {'POD@40':>8} | {'FAR@40':>8}")
    print("-" * 80)

    for exp_name, r in sorted_results:
        print(f"{exp_name:<30} | "
              f"{r['POD_30dBZ']:>8.4f} | {r['FAR_30dBZ']:>8.4f} | {r['POD_40dBZ']:>8.4f} | {r['FAR_40dBZ']:>8.4f}")

    # 保存结果
    results_path = os.path.join(args.checkpoint_dir, "evaluation_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {results_path}")
    print("=" * 100)

    return all_results


if __name__ == '__main__':
    main()
