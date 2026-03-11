"""
12 实验冒烟测试: 真实数据 + 前向/反向传播验证
==============================================

对每个实验:
1. 创建模型 + 应用实验配置
2. 从真实 SEVIR 数据加载 1 batch
3. 前向传播 (Phase1 + Phase2) 
4. 反向传播 + 梯度检查
5. 记录显存占用
6. 确认进入训练 epoch 无 bug 后停止

用法:
    python smoke_test_all.py
"""

import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datetime import datetime

from config import cfg
from experiments.experiment_factory import (
    EXPERIMENTS, apply_experiment, compute_total_loss, ExperimentConfig
)
from modules.moe_layer import MoELayer
from modules.temporal_rope import TemporalRoPE2D
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset


# 当前阶段仅验证前 10 个实验
TARGET_EXPS = [
    "exp1_moe",
    "exp2_swiglu_moe",
    "exp3_balanced_moe",
    "exp4_moe_rope",
    "exp5_swiglu_moe_rope",
    "exp6_balanced_moe_rope",
    "exp7_moe_flash",
    "exp8_swiglu_moe_flash",
    "exp9_balanced_moe_flash",
    "exp10_moe_rope_flash",
]


def create_model(exp_config):
    """创建模型"""
    model_args = argparse.Namespace(
        input_img_size=384, patch_size=4, input_channels=1,
        embed_dim=128, depths_down=[3, 2], depths_up=[2, 3],
        heads_number=[4, 8], window_size=4, out_len=12
    )
    model = Memory(model_args, memory_channel_size=512, short_len=12, long_len=36)
    model = apply_experiment(model, exp_config)
    return model


def load_real_batch(device):
    """从真实 SEVIR 数据加载 1 个 batch"""
    paths = cfg.get_sevir_paths()
    dataset = SEVIRTorchDataset(
        sevir_catalog=paths['catalog_path'],
        sevir_data_dir=paths['data_dir'],
        seq_len=36, batch_size=1,
        start_date=datetime(2017, 6, 13),
        end_date=datetime(2017, 8, 15),
        shuffle=True, verbose=False
    )
    
    # 取第一个样本, 加 batch 维度
    sample = dataset[0].unsqueeze(0).to(device)  # (1, 36, 1, 384, 384)
    return sample


def smoke_test_one(exp_name, config, real_batch, device):
    """
    对单个实验进行完整冒烟测试
    
    Returns: dict with test results
    """
    result = {
        'name': config.name,
        'status': 'UNKNOWN',
        'model_params': 0,
        'model_mem_mb': 0,
        'peak_mem_mb': 0,
        'fwd_time': 0,
        'bwd_time': 0,
        'pred_loss': 0,
        'aux_loss': 0,
        'has_grad': False,
        'error': None,
    }
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # 1. 创建模型
        model = create_model(config)
        result['model_params'] = sum(p.numel() for p in model.parameters())
        model = model.to(device)
        model.train()
        
        result['model_mem_mb'] = torch.cuda.memory_allocated() / 1024**2
        
        # 2. 优化器 + Scaler
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scaler = GradScaler('cuda')
        
        # 3. 准备数据
        x = real_batch[:, :12, :, :, :]       # 输入帧 (1, 12, 1, 384, 384)
        full = real_batch                       # 全序列 (1, 36, 1, 384, 384)
        
        optimizer.zero_grad()
        
        # ===== Phase 1: 长期记忆 =====
        t0 = time.time()
        with autocast('cuda', enabled=True):
            y_hat = model(x, full, phase=1)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            y_target = full[:, 1:, :, :, :]
            
            # y_hat 和 y_target 可能形状不同，取公共部分
            min_t = min(y_hat.shape[1] if y_hat.dim() > 1 else y_hat.shape[0],
                       y_target.shape[1] if y_target.dim() > 1 else y_target.shape[0])
            pred_loss_p1 = F.l1_loss(y_hat[:, :min_t] if y_hat.dim() > 1 else y_hat[:min_t], 
                                      y_target[:, :min_t] if y_target.dim() > 1 else y_target[:min_t])
            
            total_loss_p1, loss_dict_p1 = compute_total_loss(pred_loss_p1, model, config)
            scaled_loss_p1 = total_loss_p1 / 4  # accumulation_steps=4
        
        scaler.scale(scaled_loss_p1).backward()
        fwd_time = time.time() - t0
        
        # ===== Phase 2: 短期预测 =====
        t1 = time.time()
        with autocast('cuda', enabled=True):
            y_hat2 = model(x, full, phase=2)
            if isinstance(y_hat2, list):
                y_hat2 = torch.stack(y_hat2)
            
            pred_loss_p2 = F.l1_loss(
                y_hat2[:, :min_t] if y_hat2.dim() > 1 else y_hat2[:min_t],
                y_target[:, :min_t] if y_target.dim() > 1 else y_target[:min_t]
            )
            total_loss_p2, loss_dict_p2 = compute_total_loss(pred_loss_p2, model, config)
            scaled_loss_p2 = total_loss_p2 / 4
        
        scaler.scale(scaled_loss_p2).backward()
        bwd_time = time.time() - t1
        
        # ===== 梯度更新 =====
        scaler.step(optimizer)
        scaler.update()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.parameters() if p.requires_grad)
        
        result.update({
            'status': 'PASS',
            'peak_mem_mb': peak_mem,
            'fwd_time': fwd_time,
            'bwd_time': bwd_time,
            'pred_loss': loss_dict_p1.get('pred_loss', 0),
            'aux_loss': loss_dict_p1.get('aux_loss', 0),
            'has_grad': has_grad,
        })
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            result['status'] = 'OOM'
            result['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            result['error'] = 'CUDA OOM'
        else:
            result['status'] = 'FAIL'
            result['error'] = str(e)
            traceback.print_exc()
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
        traceback.print_exc()
    finally:
        # 清理
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if 'scaler' in locals():
            del scaler
        torch.cuda.empty_cache()
    
    return result


def main():
    print("=" * 90)
    print("DATSwinLSTM-Memory 12 实验冒烟测试")
    print("=" * 90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
    print(f"GPU: {gpu_name} | 总显存: {gpu_total:.0f} MB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"SDPA: {'可用' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else '不可用'}")
    
    # 加载真实数据
    print("\n加载 SEVIR 真实数据...")
    real_batch = load_real_batch(device)
    print(f"数据 shape: {real_batch.shape}, range: [{real_batch.min():.4f}, {real_batch.max():.4f}]")
    
    # 运行所有实验
    results = {}
    
    print(f"\n{'='*90}")
    print(f"{'实验':35s} | {'状态':>6s} | {'参数量':>12s} | {'峰值显存':>10s} | {'前向':>7s} | {'Loss':>8s} | {'梯度':>4s}")
    print(f"{'-'*90}")
    
    print(f"\n[INFO] Target experiments ({len(TARGET_EXPS)}): {', '.join(TARGET_EXPS)}")

    for exp_name in TARGET_EXPS:
        if exp_name not in EXPERIMENTS:
            print(f"\n--- 测试 {exp_name} ---")
            results[exp_name] = {
                'name': exp_name,
                'status': 'FAIL',
                'model_params': 0,
                'model_mem_mb': 0,
                'peak_mem_mb': 0,
                'fwd_time': 0,
                'bwd_time': 0,
                'pred_loss': 0,
                'aux_loss': 0,
                'has_grad': False,
                'error': 'experiment not defined in EXPERIMENTS',
            }
            print(f"{exp_name:35s} | ✗ FAIL | {'0':>12s} | {'0.0MB':>10s} | {'0.0s':>7s} | {'0.0000':>8s} | {'✗':>4s}")
            continue

        config = EXPERIMENTS[exp_name]
        print(f"\n--- 测试 {exp_name} ---")
        
        result = smoke_test_one(exp_name, config, real_batch, device)
        results[exp_name] = result
        
        # 打印单行结果
        status_icon = {'PASS': '✓', 'FAIL': '✗', 'OOM': '⚠'}.get(result['status'], '?')
        grad_str = '✓' if result['has_grad'] else '✗'
        
        print(f"{config.name:35s} | {status_icon} {result['status']:>4s} | "
              f"{result['model_params']:>12,} | "
              f"{result['peak_mem_mb']:>8.1f}MB | "
              f"{result['fwd_time']:>5.1f}s | "
              f"{result['pred_loss']:>8.4f} | "
              f"{grad_str:>4s}")
        
        if result['error']:
            print(f"  └─ ERROR: {result['error'][:80]}")
    
    # 汇总报告
    print(f"\n{'='*90}")
    print("冒烟测试汇总报告")
    print(f"{'='*90}")
    
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    failed = sum(1 for r in results.values() if r['status'] == 'FAIL')
    oom = sum(1 for r in results.values() if r['status'] == 'OOM')
    
    print(f"\n总计: {len(results)} | 通过: {passed} | 失败: {failed} | OOM: {oom}")
    
    # Flash vs Non-Flash 显存对比
    print(f"\n--- Flash Attention 显存对比 ---")
    pairs = [
        ('exp1_moe', 'exp7_moe_flash'),
        ('exp2_swiglu_moe', 'exp8_swiglu_moe_flash'),
        ('exp3_balanced_moe', 'exp9_balanced_moe_flash'),
        ('exp4_moe_rope', 'exp10_moe_rope_flash'),
    ]
    
    for base, flash in pairs:
        if base in results and flash in results:
            base_mem = results[base]['peak_mem_mb']
            flash_mem = results[flash]['peak_mem_mb']
            saving = base_mem - flash_mem
            print(f"  {results[base]['name']:30s}  {base_mem:>7.1f}MB → {flash_mem:>7.1f}MB  (省 {saving:>+.1f}MB)")
    
    # 全部通过?
    if passed == len(results):
        print(f"\n✓ 全部 {len(results)} 个实验冒烟测试通过！可以开始正式训练。")
    else:
        print(f"\n⚠ 有 {failed + oom} 个实验未通过，请检查上方错误信息。")
    
    return results


if __name__ == '__main__':
    main()
