"""
实验对比分析脚本
================
对比 12 个实验的参数量、显存占用、训练结果
Exp1~6: 基础 MoE+RoPE
Exp7~12: + Flash Attention (SDPA)

用法:
    python compare_experiments.py               # 纯参数分析 (不需要GPU)
    python compare_experiments.py --gpu_test     # 包含GPU显存测试
    python compare_experiments.py --results      # 对比训练结果 (需要先训练)
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.amp import autocast


def analyze_parameters():
    """分析 6 个实验的参数量差异"""
    import argparse as ap
    from models.DATSwinLSTM_D_Memory import Memory
    from experiments.experiment_factory import EXPERIMENTS, apply_experiment
    from modules.moe_layer import MoELayer
    from modules.temporal_rope import TemporalRoPE2D
    
    model_args = ap.Namespace(
        input_img_size=384, patch_size=4, input_channels=1,
        embed_dim=128, depths_down=[3, 2], depths_up=[2, 3],
        heads_number=[4, 8], window_size=4, out_len=12
    )
    
    # Baseline
    baseline = Memory(model_args, memory_channel_size=512, short_len=12, long_len=36)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    del baseline
    
    print("=" * 90)
    print(f"{'实验':30s} | {'总参数':>12s} | {'MoE参数':>10s} | {'RoPE':>5s} | {'Flash':>5s} | {'vs Baseline':>12s}")
    print("-" * 90)
    print(f"{'Baseline (无改动)':30s} | {baseline_params:>12,} | {'--':>10s} | {'--':>5s} | {'--':>5s} | {'---':>12s}")
    
    results = {}
    
    for exp_name, config in EXPERIMENTS.items():
        model = Memory(model_args, memory_channel_size=512, short_len=12, long_len=36)
        model = apply_experiment(model, config)
        
        total_params = sum(p.numel() for p in model.parameters())
        moe_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.modules() if isinstance(m, MoELayer)
        )
        n_moe = sum(1 for m in model.modules() if isinstance(m, MoELayer))
        n_rope = sum(1 for m in model.modules() if isinstance(m, TemporalRoPE2D))
        
        diff = total_params - baseline_params
        diff_pct = 100 * diff / baseline_params
        
        rope_str = str(n_rope) if n_rope > 0 else "--"
        flash_str = "\u2713" if config.use_flash else "--"
        
        print(f"{config.name:30s} | {total_params:>12,} | {moe_params:>10,} | {rope_str:>5s} | {flash_str:>5s} | {diff:>+12,} ({diff_pct:>+.1f}%)")
        
        results[exp_name] = {
            'name': config.name,
            'total_params': total_params,
            'moe_params': moe_params,
            'n_moe_layers': n_moe,
            'n_rope_modules': n_rope,
            'diff_from_baseline': diff,
        }
        
        del model
    
    print("=" * 90)
    return results


def gpu_memory_test():
    """测试每个实验的 GPU 显存占用"""
    if not torch.cuda.is_available():
        print("No GPU available, skipping GPU test")
        return
    
    import argparse as ap
    from models.DATSwinLSTM_D_Memory import Memory
    from experiments.experiment_factory import EXPERIMENTS, apply_experiment, compute_total_loss
    
    model_args = ap.Namespace(
        input_img_size=384, patch_size=4, input_channels=1,
        embed_dim=128, depths_down=[3, 2], depths_up=[2, 3],
        heads_number=[4, 8], window_size=4, out_len=12
    )
    
    device = torch.device('cuda')
    
    print("\n" + "=" * 80)
    print(f"{'实验':30s} | {'模型显存':>10s} | {'前向峰值':>10s} | {'状态':>6s}")
    print("-" * 80)
    
    for exp_name, config in EXPERIMENTS.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # 创建模型
            model = Memory(model_args, memory_channel_size=512, short_len=12, long_len=36)
            model = apply_experiment(model, config)
            model = model.to(device)
            model.train()
            
            model_mem = torch.cuda.memory_allocated() / 1024**2
            
            # 模拟前向传播 (单batch)
            x = torch.randn(1, 12, 1, 384, 384, device=device)
            full = torch.randn(1, 36, 1, 384, 384, device=device)
            
            with autocast('cuda', enabled=True):
                y_hat = model(x, full, phase=1)
                if isinstance(y_hat, list):
                    y_hat = torch.stack(y_hat)
                loss = y_hat.mean()
                total_loss, _ = compute_total_loss(loss, model, config)
            
            total_loss.backward()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            status = "✓"
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                status = "OOM"
            else:
                peak_mem = 0
                status = "ERR"
        finally:
            del model
            if 'x' in locals():
                del x, full
            torch.cuda.empty_cache()
        
        print(f"{config.name:30s} | {model_mem:>8.1f}MB | {peak_mem:>8.1f}MB | {status:>6s}")
    
    print("=" * 80)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    print(f"GPU: {gpu_name} | Total: {gpu_total:.0f}MB")


def compare_results(checkpoint_dir='./checkpoints'):
    """对比训练结果"""
    from experiments.experiment_factory import EXPERIMENTS
    
    print("\n" + "=" * 80)
    print(f"{'实验':30s} | {'Best Val Loss':>14s} | {'Last Epoch':>10s} | {'总时间':>10s}")
    print("-" * 80)
    
    for exp_name in EXPERIMENTS:
        log_path = os.path.join(checkpoint_dir, exp_name, 'training_log.json')
        if not os.path.exists(log_path):
            print(f"{EXPERIMENTS[exp_name].name:30s} | {'未训练':>14s} |")
            continue
        
        with open(log_path) as f:
            log = json.load(f)
        
        if not log:
            print(f"{EXPERIMENTS[exp_name].name:30s} | {'空日志':>14s} |")
            continue
        
        best_val = min(entry['val_loss'] for entry in log)
        last_epoch = log[-1]['epoch']
        total_time = sum(entry.get('time', 0) for entry in log)
        
        print(f"{EXPERIMENTS[exp_name].name:30s} | {best_val:>14.4f} | {last_epoch:>10d} | {total_time/60:>8.1f}min")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_test', action='store_true', help='Run GPU memory test')
    parser.add_argument('--results', action='store_true', help='Compare training results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    # 总是显示参数分析
    results = analyze_parameters()
    
    if args.gpu_test:
        gpu_memory_test()
    
    if args.results:
        compare_results(args.checkpoint_dir)


if __name__ == '__main__':
    main()
