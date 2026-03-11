"""
DATSwinLSTM-Memory 评估脚本
===========================
计算 CSI, HSS, POD, FAR 在 SEVIR 测试集上的指标

VIL 阈值 (论文 IV-B): 0.14, 0.70, 3.50, 6.90 kg/m²

用法:
    python evaluate.py --exp exp12_balanced_moe_rope_flash
    python evaluate.py --exp 384x384 --ckpt checkpoints/384x384/best_model.pth --baseline
    python evaluate.py --all
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset

# ===================== VIL 阈值转换 =====================
# SEVIR VIL: p∈[0,255], 归一化到 [0,1] (p/255)
# VIL (kg/m²) → p:
#   p = VIL*90.66 + 2       if VIL <= 0.1765
#   p = 38.9*ln(VIL) + 83.9 if VIL > 0.1765

def vil_to_normalized(vil_kgm2):
    """将 VIL (kg/m²) 转换为归一化 [0,1] 像素值"""
    import math
    if vil_kgm2 <= 0.1765:
        p = vil_kgm2 * 90.66 + 2
    else:
        p = 38.9 * math.log(vil_kgm2) + 83.9
    return p / 255.0

# 论文使用的4个阈值
THRESHOLDS_KGM2 = [0.14, 0.70, 3.50, 6.90]
THRESHOLDS_NORM = [vil_to_normalized(t) for t in THRESHOLDS_KGM2]


def compute_metrics(pred, target, threshold):
    """计算单个阈值下的 CSI, HSS, POD, FAR"""
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()

    hits = ((pred_bin == 1) & (tgt_bin == 1)).sum().item()
    misses = ((pred_bin == 0) & (tgt_bin == 1)).sum().item()
    false_alarms = ((pred_bin == 1) & (tgt_bin == 0)).sum().item()
    correct_neg = ((pred_bin == 0) & (tgt_bin == 0)).sum().item()

    # CSI = hits / (hits + misses + false_alarms)
    csi = hits / max(hits + misses + false_alarms, 1e-10)

    # POD = hits / (hits + misses)
    pod = hits / max(hits + misses, 1e-10)

    # FAR = false_alarms / (hits + false_alarms)
    far = false_alarms / max(hits + false_alarms, 1e-10)

    # HSS = 2*(hits*cn - misses*fa) / ((hits+misses)*(misses+cn) + (hits+fa)*(fa+cn))
    num = 2 * (hits * correct_neg - misses * false_alarms)
    den = ((hits + misses) * (misses + correct_neg) +
           (hits + false_alarms) * (false_alarms + correct_neg))
    hss = num / max(den, 1e-10)

    return {'CSI': csi, 'POD': pod, 'FAR': far, 'HSS': hss,
            'hits': hits, 'misses': misses, 'fa': false_alarms, 'cn': correct_neg}


def create_test_loader(args):
    """创建测试集 DataLoader"""
    sevir_paths = cfg.get_sevir_paths()
    test_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=1,
        start_date=datetime.datetime(2017, 9, 15),
        end_date=datetime.datetime(2017, 12, 31),
        shuffle=False, verbose=True
    )
    return DataLoader(test_dataset, batch_size=1, shuffle=False,
                      num_workers=0, pin_memory=True)


def load_model_exp(args, exp_name, device):
    """加载 exp7-12 实验模型"""
    from experiments.experiment_factory import EXPERIMENTS, apply_experiment

    config = EXPERIMENTS[exp_name]
    model_args = argparse.Namespace(
        input_img_size=384, patch_size=args.patch_size, input_channels=1,
        embed_dim=args.embed_dim, depths_down=[2, 2], depths_up=[2, 2],
        heads_number=[4, 4], window_size=args.window_size, out_len=args.output_frames
    )
    model = Memory(model_args, memory_channel_size=args.memory_channel_size,
                   short_len=args.input_frames, long_len=args.seq_len)
    model = apply_experiment(model, config)

    # 加载权重
    ckpt_path = os.path.join(args.checkpoint_dir, exp_name, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.checkpoint_dir, exp_name, 'latest_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt.get('epoch', '?'), ckpt.get('val_loss', '?')


def load_model_384(args, device):
    """加载 384 基线模型 (384x384_opt 用 L1+MSE, 或旧 384x384 用 MSE)"""
    model_args = argparse.Namespace(
        input_img_size=384, patch_size=4, input_channels=1,
        embed_dim=64, depths_down=[2, 2], depths_up=[2, 2],
        heads_number=[4, 4], window_size=4, out_len=12
    )
    # 与 train_384_opt.py 对齐: long_len=20
    model = Memory(model_args, memory_channel_size=256,
                   short_len=8, long_len=20)

    # 优先用 384x384_opt (同 L1+MSE loss), 其次用旧 384x384
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        opt_best = os.path.join(args.checkpoint_dir, '384x384_opt', 'best_model.pt')
        opt_latest = os.path.join(args.checkpoint_dir, '384x384_opt', 'latest_model.pt')
        old_best = os.path.join(args.checkpoint_dir, '384x384', 'best_model.pth')
        if os.path.exists(opt_best):
            ckpt_path = opt_best
        elif os.path.exists(opt_latest):
            ckpt_path = opt_latest
        elif os.path.exists(old_best):
            ckpt_path = old_best
            # 旧模型用 long_len=24, 需要重建
            model = Memory(model_args, memory_channel_size=256,
                           short_len=8, long_len=24)
        else:
            raise FileNotFoundError("找不到 384 基线检查点!")
    print(f"  加载基线: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt.get('epoch', '?'), ckpt.get('val_loss', '?')


@torch.no_grad()
def evaluate_model(model, test_loader, device, input_frames=8, output_frames=12,
                   seq_len=20, is_baseline=False):
    """在测试集上评估模型，返回各阈值的 CSI/HSS/POD/FAR"""
    # 累积统计
    accum = {t: {'hits': 0, 'misses': 0, 'fa': 0, 'cn': 0}
             for t in THRESHOLDS_NORM}
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0

    for batch_idx, data in enumerate(test_loader):
        data = data.to(device, non_blocking=True)
        x = data[:, :input_frames]
        target_seq = data[:, input_frames:input_frames + output_frames]

        # memory_seq: repeat 输入帧到 seq_len (与训练一致)
        repeat_factor = max(1, (seq_len + input_frames - 1) // input_frames)
        memory_seq = x.repeat(1, repeat_factor, 1, 1, 1)[:, :seq_len]

        with torch.amp.autocast('cuda'):
            output = model(x, memory_seq, phase=2)
            if isinstance(output, list):
                output = torch.stack(output, dim=1)

        min_t = min(output.shape[1], target_seq.shape[1])
        pred = output[:, :min_t].float().clamp(0, 1)
        tgt = target_seq[:, :min_t].float()

        total_mse += F.mse_loss(pred, tgt).item()
        total_mae += F.l1_loss(pred, tgt).item()
        num_samples += 1

        for thresh in THRESHOLDS_NORM:
            m = compute_metrics(pred, tgt, thresh)
            accum[thresh]['hits'] += m['hits']
            accum[thresh]['misses'] += m['misses']
            accum[thresh]['fa'] += m['fa']
            accum[thresh]['cn'] += m['cn']

        if (batch_idx + 1) % 100 == 0:
            print(f"  评估进度: {batch_idx+1}/{len(test_loader)}")

    # 计算最终指标
    results = {}
    for thresh, kgm2 in zip(THRESHOLDS_NORM, THRESHOLDS_KGM2):
        a = accum[thresh]
        h, m, f, c = a['hits'], a['misses'], a['fa'], a['cn']
        csi = h / max(h + m + f, 1e-10)
        pod = h / max(h + m, 1e-10)
        far = f / max(h + f, 1e-10)
        num = 2 * (h * c - m * f)
        den = (h + m) * (m + c) + (h + f) * (f + c)
        hss = num / max(den, 1e-10)
        results[f"{kgm2}"] = {'CSI': csi, 'POD': pod, 'FAR': far, 'HSS': hss}

    results['MSE'] = total_mse / max(num_samples, 1)
    results['MAE'] = total_mae / max(num_samples, 1)
    results['num_samples'] = num_samples
    return results


def print_results(name, results):
    """打印评估结果表格"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  MSE: {results['MSE']:.6f} | MAE: {results['MAE']:.6f}")
    print(f"{'='*70}")
    print(f"  {'Threshold':>12} | {'CSI':>8} | {'HSS':>8} | {'POD':>8} | {'FAR':>8}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for kgm2 in THRESHOLDS_KGM2:
        r = results[str(kgm2)]
        print(f"  {kgm2:>8.2f} kg | {r['CSI']:>8.4f} | {r['HSS']:>8.4f} | {r['POD']:>8.4f} | {r['FAR']:>8.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate DATSwinLSTM-Memory')
    parser.add_argument('--exp', type=str, default=None, help='实验名 (如 exp12_balanced_moe_rope_flash)')
    parser.add_argument('--baseline', action='store_true', help='评估384基线')
    parser.add_argument('--ckpt', type=str, default=None, help='自定义检查点路径')
    parser.add_argument('--all', action='store_true', help='评估所有实验+基线')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=8)
    parser.add_argument('--output_frames', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--memory_channel_size', type=int, default=256)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_file', type=str, default='./evaluation_results.json')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU 进行评估，不占用显存')
    return parser.parse_args()


def main():
    args = get_args()
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("加载测试集...")
    test_loader = create_test_loader(args)
    print(f"测试批次: {len(test_loader)}")

    all_results = {}

    if args.all:
        # 评估所有实验
        exps = [
            'exp7_moe_flash', 'exp8_swiglu_moe_flash', 'exp9_balanced_moe_flash',
            'exp10_moe_rope_flash', 'exp11_swiglu_moe_rope_flash', 'exp12_balanced_moe_rope_flash',
        ]
        # 先评估基线
        print("\n>>> 评估 384 基线...")
        try:
            model, ep, vl = load_model_384(args, device)
            results = evaluate_model(model, test_loader, device,
                                     input_frames=8, output_frames=12,
                                     seq_len=20, is_baseline=True)
            print_results("384x384 Baseline", results)
            all_results['384x384_baseline'] = results
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  384 基线评估失败: {e}")

        # 评估各实验
        for exp_name in exps:
            print(f"\n>>> 评估 {exp_name}...")
            try:
                model, ep, vl = load_model_exp(args, exp_name, device)
                results = evaluate_model(model, test_loader, device,
                                         input_frames=args.input_frames,
                                         output_frames=args.output_frames,
                                         seq_len=args.seq_len, is_baseline=False)
                print_results(exp_name, results)
                all_results[exp_name] = results
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {exp_name} 评估失败: {e}")

    elif args.baseline:
        model, ep, vl = load_model_384(args, device)
        results = evaluate_model(model, test_loader, device,
                                 input_frames=8, output_frames=12,
                                 seq_len=20, is_baseline=True)
        print_results("384x384 Baseline", results)
        all_results['384x384_baseline'] = results

    elif args.exp:
        model, ep, vl = load_model_exp(args, args.exp, device)
        results = evaluate_model(model, test_loader, device,
                                 input_frames=args.input_frames,
                                 output_frames=args.output_frames,
                                 seq_len=args.seq_len, is_baseline=False)
        print_results(args.exp, results)
        all_results[args.exp] = results

    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {args.output_file}")

    # 如果有多个结果，打印对比表
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print("  综合对比")
        print(f"{'='*90}")
        header = f"  {'Model':>35} | {'MSE':>8} | {'MAE':>8}"
        for kgm2 in THRESHOLDS_KGM2:
            header += f" | CSI@{kgm2}"
        print(header)
        print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}" + "-+---------" * len(THRESHOLDS_KGM2))
        for name, res in all_results.items():
            line = f"  {name:>35} | {res['MSE']:>8.5f} | {res['MAE']:>8.5f}"
            for kgm2 in THRESHOLDS_KGM2:
                line += f" | {res[str(kgm2)]['CSI']:>7.4f}"
            print(line)


if __name__ == '__main__':
    main()
