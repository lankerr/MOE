"""
DATSwinLSTM-Memory MoE+RoPE+Flash 实验训练脚本
================================================

统一训练脚本，支持 12 种实验变体:
  # 基础 MoE 实验 (手动 Attention)
  python train_experiment.py --exp exp1_moe
  python train_experiment.py --exp exp2_swiglu_moe
  python train_experiment.py --exp exp3_balanced_moe
  python train_experiment.py --exp exp4_moe_rope
  python train_experiment.py --exp exp5_swiglu_moe_rope
  python train_experiment.py --exp exp6_balanced_moe_rope
  
  # Flash Attention 实验 (SDPA)
  python train_experiment.py --exp exp7_moe_flash
  python train_experiment.py --exp exp8_swiglu_moe_flash
  python train_experiment.py --exp exp9_balanced_moe_flash
  python train_experiment.py --exp exp10_moe_rope_flash
  python train_experiment.py --exp exp11_swiglu_moe_rope_flash
  python train_experiment.py --exp exp12_balanced_moe_rope_flash
  
  # 跑所有实验
  python train_experiment.py --exp all

RTX 5070 8GB 显存优化:
  - 混合精度 (AMP bf16)
  - 梯度累积 (默认 4 步)
  - 稀疏 MoE (每 token 仅激活 top-2 专家)
  - Flash Attention (SDPA) 省 ~0.5GB 显存
"""

import os
import sys
import argparse
import datetime
import json
import time

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from config import cfg
from experiments.experiment_factory import (
    EXPERIMENTS, ExperimentConfig, 
    apply_experiment, compute_total_loss, get_experiment_expert_stats,
    collect_moe_aux_losses
)
from models.DATSwinLSTM_D_Memory import Memory


def get_args():
    parser = argparse.ArgumentParser(description='DATSwinLSTM-Memory MoE+RoPE Experiments')
    
    # 实验选择
    parser.add_argument('--exp', type=str, default='exp1_moe',
                        choices=list(EXPERIMENTS.keys()) + ['all'],
                        help='Experiment name')
    
    # 数据参数
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=8)
    parser.add_argument('--output_frames', type=int, default=12)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Debug mode: max batches per epoch (0 means all)')
    parser.add_argument('--enable_phase1', action='store_true', default=False,
                        help='Enable legacy phase-1 memory training (default: off for stability)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Enable AMP autocast')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable AMP and use full fp32')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='AMP dtype: bf16 (recommended on RTX50) or fp16')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--memory_channel_size', type=int, default=256)
    
    # 输出
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip_val', action='store_true', default=False,
                        help='Debug mode: skip validation phase')
    
    # MoE 覆盖参数 (可选, 覆盖预设)
    parser.add_argument('--num_experts', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    
    args = parser.parse_args()

    # 统一 dtype 映射，避免在训练循环里重复判断
    if args.amp_dtype == 'bf16':
        args.autocast_dtype = torch.bfloat16
    else:
        args.autocast_dtype = torch.float16

    return args


def create_model(args, exp_config: ExperimentConfig, device):
    """创建并配置模型"""
    model_args = argparse.Namespace(
        input_img_size=384,
        patch_size=args.patch_size,
        input_channels=1,
        embed_dim=args.embed_dim,
        depths_down=[2, 2],
        depths_up=[2, 2],
        heads_number=[4, 4],
        window_size=args.window_size,
        out_len=args.output_frames
    )
    
    # 创建基础模型
    model = Memory(
        model_args,
        memory_channel_size=args.memory_channel_size,
        short_len=args.input_frames,
        long_len=args.seq_len
    )
    
    # 应用实验配置 (MoE替换 + RoPE注入)
    model = apply_experiment(model, exp_config)
    
    return model.to(device)


def create_dataloaders(args):
    """创建数据加载器"""
    from sevir_torch_wrap import SEVIRTorchDataset
    
    sevir_paths = cfg.get_sevir_paths()
    
    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 6, 13),
        end_date=datetime.datetime(2017, 8, 15),
        shuffle=True, verbose=True
    )
    
    val_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 8, 15),
        end_date=datetime.datetime(2017, 9, 15),
        shuffle=False, verbose=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, exp_config):
    """训练一个epoch"""
    model.train()
    total_pred_loss = 0
    total_aux_loss = 0
    num_batches = 0
    t0 = time.time()
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        x = batch[:, :args.input_frames, :, :, :]
        repeat_factor = max(1, (args.seq_len + args.input_frames - 1) // args.input_frames)
        memory_seq = x.repeat(1, repeat_factor, 1, 1, 1)[:, :args.seq_len]
        
        # Phase 1: 长期记忆 (可选)
        if args.enable_phase1:
            model.set_memory_bank_requires_grad(True)

            with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
                y_hat = model(x, batch, phase=1)
                if isinstance(y_hat, list):
                    y_hat = torch.stack(y_hat, dim=1)
                y_target = batch[:, 1:, :, :, :]
                min_t = min(y_hat.shape[1], y_target.shape[1])
                pred_loss = F.l1_loss(y_target[:, :min_t], y_hat[:, :min_t])

                # MoE 辅助损失
                total_loss, loss_dict = compute_total_loss(pred_loss, model, exp_config)
                total_loss = total_loss / args.accumulation_steps

            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        else:
            loss_dict = {'pred_loss': 0.0, 'aux_loss': 0.0}
        
        # Phase 2: 短期预测
        model.set_memory_bank_requires_grad(False)
        
        with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
            y_hat2 = model(x, memory_seq, phase=2)
            if isinstance(y_hat2, list):
                y_hat2 = torch.stack(y_hat2, dim=1)
            y_target2 = batch[:, args.input_frames:args.input_frames + args.output_frames, :, :, :]
            # 截断到匹配长度
            min_len = min(y_hat2.shape[1], y_target2.shape[1])
            pred_loss2 = F.l1_loss(y_target2[:, :min_len], y_hat2[:, :min_len])
            
            total_loss2, loss_dict2 = compute_total_loss(pred_loss2, model, exp_config)
            total_loss2 = total_loss2 / args.accumulation_steps
        
        if scaler:
            scaler.scale(total_loss2).backward()
        else:
            total_loss2.backward()
        
        # 梯度累积
        if (batch_idx + 1) % args.accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_pred_loss += (loss_dict.get('pred_loss', 0) + loss_dict2.get('pred_loss', 0))
        total_aux_loss += (loss_dict.get('aux_loss', 0) + loss_dict2.get('aux_loss', 0))
        num_batches += 1
        
        # 日志
        if (batch_idx + 1) % args.log_interval == 0:
            avg_pred = total_pred_loss / num_batches
            avg_aux = total_aux_loss / num_batches
            elapsed = time.time() - t0
            
            log_msg = (f"  [{exp_config.name}] Epoch {epoch} | "
                      f"Batch {batch_idx+1}/{len(loader)} | "
                      f"Pred Loss: {avg_pred:.4f}")
            if avg_aux > 0:
                log_msg += f" | Aux Loss: {avg_aux:.6f}"
            log_msg += f" | Time: {elapsed:.1f}s"
            print(log_msg)
            
            # MoE 专家统计
            if exp_config.use_moe:
                stats = get_experiment_expert_stats(model)
                for layer_name, s in list(stats.items())[:2]:  # 只显示前2层
                    ratios = s.get('expert_ratios', [])
                    if ratios:
                        print(f"    {layer_name}: E_ratios={[f'{r:.2f}' for r in ratios]} "
                              f"balance={s.get('balance_score', 0):.3f}")

        # Debug: 每个 epoch 只跑前 max_batches 个 batch
        if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
            print(f"  [Debug] 达到 max_batches={args.max_batches}，提前结束本 epoch")
            break
    
    return {
        'pred_loss': total_pred_loss / max(num_batches, 1),
        'aux_loss': total_aux_loss / max(num_batches, 1),
        'time': time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, device, args, exp_config):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        x = batch[:, :args.input_frames, :, :, :]
        repeat_factor = max(1, (args.seq_len + args.input_frames - 1) // args.input_frames)
        memory_seq = x.repeat(1, repeat_factor, 1, 1, 1)[:, :args.seq_len]
        
        with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
            y_hat = model(x, memory_seq, phase=2)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat, dim=1)
            y_target = batch[:, args.input_frames:args.input_frames + args.output_frames, :, :, :]
            min_len = min(y_hat.shape[1], y_target.shape[1])
            loss = F.l1_loss(y_target[:, :min_len], y_hat[:, :min_len])
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def run_experiment(exp_name: str, args):
    """运行单个实验"""
    exp_config = EXPERIMENTS[exp_name]
    
    # 可选参数覆盖
    if args.num_experts is not None:
        exp_config.num_experts = args.num_experts
    if args.top_k is not None:
        exp_config.top_k = args.top_k
    
    print(f"\n{'#'*60}")
    print(f"# 实验: {exp_config.name}")
    print(f"# 配置: {exp_config}")
    print(f"{'#'*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = create_model(args, exp_config, device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # AMP scaler: 仅 fp16 需要 GradScaler；bf16 不需要（且在部分 nightly 上更稳定）
    scaler = GradScaler('cuda') if (args.use_amp and args.amp_dtype == 'fp16') else None
    print(f"AMP: {'ON' if args.use_amp else 'OFF'} | dtype={args.amp_dtype} | GradScaler={'ON' if scaler else 'OFF'}")
    
    # 数据加载
    try:
        train_loader, val_loader = create_dataloaders(args)
    except Exception as e:
        print(f"⚠ 数据加载失败: {e}")
        print("使用合成数据进行干跑测试...")
        train_loader = val_loader = None
    
    # Checkpoint 目录
    exp_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    
    # 训练日志
    log_path = os.path.join(exp_ckpt_dir, 'training_log.json')
    training_log = []
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"从 epoch {start_epoch} 恢复训练")
    
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if train_loader is None:
            # 干跑: 合成数据
            print(f"\nEpoch {epoch+1}/{args.epochs} (dry run)")
            batch_size = 1
            x_fake = torch.randn(batch_size, args.input_frames, 1, 384, 384, device=device)
            full_fake = torch.randn(batch_size, args.seq_len, 1, 384, 384, device=device)
            repeat_factor = max(1, (args.seq_len + args.input_frames - 1) // args.input_frames)
            memory_fake = x_fake.repeat(1, repeat_factor, 1, 1, 1)[:, :args.seq_len]
            
            model.train()
            with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
                try:
                    # 与 train_384 保持一致，主测 phase=2 路径
                    y_hat = model(x_fake, memory_fake, phase=2)
                    if isinstance(y_hat, list):
                        y_hat = torch.stack(y_hat, dim=1)
                    loss = y_hat.mean()
                    total_loss, loss_dict = compute_total_loss(loss, model, exp_config)
                    print(f"  Dry run loss: {loss_dict}")
                except Exception as e:
                    print(f"  ✗ 前向传播错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            
            if epoch >= 2:
                print("干跑测试完成 (3 epochs)")
                break
            continue
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch+1, args, exp_config
        )
        
        # 验证
        if args.skip_val:
            val_loss = train_metrics['pred_loss']
            print("  [Debug] 跳过验证阶段 (--skip_val)")
        else:
            val_loss = validate(model, val_loader, device, args, exp_config)
        
        scheduler.step()
        
        # 日志
        epoch_log = {
            'epoch': epoch + 1,
            'train_pred_loss': train_metrics['pred_loss'],
            'train_aux_loss': train_metrics['aux_loss'],
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': train_metrics['time'],
        }
        training_log.append(epoch_log)
        
        print(f"  Train Loss: {train_metrics['pred_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {train_metrics['time']:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': exp_config.__dict__,
            }, os.path.join(exp_ckpt_dir, 'best_model.pt'))
            print(f"  ★ 新最佳模型! Val Loss: {val_loss:.4f}")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(exp_ckpt_dir, f'epoch_{epoch+1}.pt'))
        
        # 保存日志
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # 清理 GPU
    del model
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"实验 {exp_name} 完成! Best Val Loss: {best_val_loss:.4f}")
    print(f"日志: {log_path}")
    print(f"{'='*60}")
    
    return best_val_loss


def main():
    args = get_args()
    
    if args.exp == 'all':
        # 运行所有实验
        results = {}
        for exp_name in EXPERIMENTS:
            try:
                val_loss = run_experiment(exp_name, args)
                results[exp_name] = val_loss
            except Exception as e:
                print(f"\n✗ 实验 {exp_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[exp_name] = float('inf')
        
        # 结果汇总
        print("\n" + "=" * 60)
        print("实验结果汇总")
        print("=" * 60)
        for exp_name, val_loss in sorted(results.items(), key=lambda x: x[1]):
            config = EXPERIMENTS[exp_name]
            print(f"  {exp_name:30s} | Val Loss: {val_loss:.4f} | {config.name}")
        
        # 保存汇总
        with open(os.path.join(args.checkpoint_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        run_experiment(args.exp, args)


if __name__ == '__main__':
    main()
