"""
DATSwinLSTM-Memory 加速版训练脚本 (Exp7-12 Flash实验)
=====================================================

加速措施:
  1. num_workers=2 + persistent_workers  — 数据加载与GPU计算重叠
  2. cudnn.benchmark = True              — cuDNN自动调优(固定384x384输入)
  3. torch.set_float32_matmul_precision  — TF32 矩阵乘(RTX50系列)
  4. torch.compile (可选)                — 算子融合加速
  5. 数据集 RAM 缓存                     — 避免反复读 HDF5

用法:
  python -u train_experiment_fast.py --exp exp12_balanced_moe_rope_flash --epochs 1
  python -u train_experiment_fast.py --exp all --epochs 100
"""

import os
import sys
import argparse
import datetime
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import numpy as np

from config import cfg
from experiments.experiment_factory import (
    EXPERIMENTS, ExperimentConfig,
    apply_experiment, compute_total_loss, get_experiment_expert_stats,
    collect_moe_aux_losses
)
from models.DATSwinLSTM_D_Memory import Memory

# ===================== 加速设置 =====================
# [加速1] cuDNN benchmark — 对固定输入尺寸自动选择最快卷积算法
torch.backends.cudnn.benchmark = True
# [加速2] TF32 矩阵乘 — RTX 30/40/50 系列硬件加速
torch.set_float32_matmul_precision('medium')


# ===================== 缓存数据集 =====================
class CachedSEVIRDataset(Dataset):
    """
    [加速3] 把所有样本预读进 RAM, 避免每个 batch 都打开 HDF5
    1738 samples × (20, 1, 384, 384) × float32 ≈ 8.5 GB RAM
    如果内存不够会自动回退到逐个读取
    """

    def __init__(self, base_dataset, cache_in_ram=True):
        self.base = base_dataset
        self.cache = None
        if cache_in_ram:
            try:
                print(f"  [Cache] 预加载 {len(base_dataset)} 个样本到 RAM...")
                t0 = time.time()
                self.cache = [base_dataset[i] for i in range(len(base_dataset))]
                elapsed = time.time() - t0
                # 估算内存占用
                mem_mb = sum(s.nelement() * s.element_size() for s in self.cache) / 1024**2
                print(f"  [Cache] 完成! {mem_mb:.0f} MB, 耗时 {elapsed:.1f}s")
            except MemoryError:
                print("  [Cache] 内存不足, 回退到逐个读取模式")
                self.cache = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        return self.base[idx]


def get_args():
    parser = argparse.ArgumentParser(description='DATSwinLSTM-Memory Fast Training')

    parser.add_argument('--exp', type=str, default='exp12_balanced_moe_rope_flash',
                        choices=list(EXPERIMENTS.keys()) + ['all'])

    # 数据
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=8)
    parser.add_argument('--output_frames', type=int, default=12)

    # 训练
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)          # 0=用RAM缓存加速, >0=多进程(禁用缓存)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--enable_phase1', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile (experimental, may slow 1st epoch)')
    parser.add_argument('--no_cache', action='store_true', default=True,
                        help='Disable RAM cache for datasets (default: disabled)')
    parser.add_argument('--cache', action='store_false', dest='no_cache',
                        help='Enable RAM cache (需要 16GB+ 空闲 RAM)')

    # 模型
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--memory_channel_size', type=int, default=256)
    parser.add_argument('--memory_slot_size', type=int, default=100)

    # 输出
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip_val', action='store_true', default=False)

    # MoE 覆盖
    parser.add_argument('--num_experts', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)

    args = parser.parse_args()
    args.autocast_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    return args


def create_model(args, exp_config, device):
    model_args = argparse.Namespace(
        input_img_size=384, patch_size=args.patch_size, input_channels=1,
        embed_dim=args.embed_dim, depths_down=[2, 2], depths_up=[2, 2],
        heads_number=[4, 4], window_size=args.window_size, out_len=args.output_frames
    )
    model = Memory(model_args, memory_channel_size=args.memory_channel_size,
                   memory_slot_size=args.memory_slot_size,
                   short_len=args.input_frames, long_len=args.seq_len)
    model = apply_experiment(model, exp_config)
    model = model.to(device)

    # [加速5] torch.compile — 算子融合 (PyTorch 2.x)
    if args.use_compile:
        try:
            print("[Compile] 正在编译模型 (首次前向会慢, 之后加速)...")
            model = torch.compile(model, mode='reduce-overhead')
            print("[Compile] 编译完成!")
        except Exception as e:
            print(f"[Compile] 编译失败, 使用 eager 模式: {e}")

    return model


def create_dataloaders(args):
    from sevir_torch_wrap import SEVIRTorchDataset

    sevir_paths = cfg.get_sevir_paths()

    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len, batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 6, 13),
        end_date=datetime.datetime(2017, 8, 15),
        shuffle=True, verbose=True
    )
    val_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len, batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 8, 15),
        end_date=datetime.datetime(2017, 9, 15),
        shuffle=False, verbose=True
    )

    # [加速3] 缓存数据到 RAM
    # 注意: num_workers>0 时 worker 会 fork/复制缓存, 导致 RAM OOM
    # 所以: workers=0 + cache=ON (推荐) 或 workers>0 + cache=OFF
    if args.num_workers > 0 and not args.no_cache:
        print("  [Cache] num_workers>0, 自动禁用 RAM 缓存 (避免 worker 复制内存)")
        use_cache = False
    else:
        use_cache = not args.no_cache
    train_dataset = CachedSEVIRDataset(train_dataset, cache_in_ram=use_cache)
    val_dataset = CachedSEVIRDataset(val_dataset, cache_in_ram=use_cache)

    # [加速4] num_workers + persistent_workers
    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, exp_config):
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

        # Phase 1 (可选)
        if args.enable_phase1:
            model.set_memory_bank_requires_grad(True)
            with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
                y_hat = model(x, batch, phase=1)
                if isinstance(y_hat, list):
                    y_hat = torch.stack(y_hat, dim=1)
                y_target = batch[:, 1:, :, :, :]
                min_t = min(y_hat.shape[1], y_target.shape[1])
                pred_loss = F.l1_loss(y_target[:, :min_t], y_hat[:, :min_t]) + F.mse_loss(y_target[:, :min_t], y_hat[:, :min_t])
                total_loss, loss_dict = compute_total_loss(pred_loss, model, exp_config)
                total_loss = total_loss / args.accumulation_steps
            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        else:
            loss_dict = {'pred_loss': 0.0, 'aux_loss': 0.0}

        # Phase 2
        model.set_memory_bank_requires_grad(False)
        with autocast('cuda', enabled=args.use_amp, dtype=args.autocast_dtype):
            y_hat2 = model(x, memory_seq, phase=2)
            if isinstance(y_hat2, list):
                y_hat2 = torch.stack(y_hat2, dim=1)
            y_target2 = batch[:, args.input_frames:args.input_frames + args.output_frames, :, :, :]
            min_len = min(y_hat2.shape[1], y_target2.shape[1])
            pred_loss2 = F.l1_loss(y_target2[:, :min_len], y_hat2[:, :min_len]) + F.mse_loss(y_target2[:, :min_len], y_hat2[:, :min_len])
            total_loss2, loss_dict2 = compute_total_loss(pred_loss2, model, exp_config)
            total_loss2 = total_loss2 / args.accumulation_steps

        if scaler:
            scaler.scale(total_loss2).backward()
        else:
            total_loss2.backward()

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

        if (batch_idx + 1) % args.log_interval == 0:
            avg_pred = total_pred_loss / num_batches
            avg_aux = total_aux_loss / num_batches
            elapsed = time.time() - t0
            speed = num_batches / elapsed

            log_msg = (f"  [{exp_config.name}] Epoch {epoch} | "
                      f"Batch {batch_idx+1}/{len(loader)} | "
                      f"Pred Loss: {avg_pred:.4f}")
            if avg_aux > 0:
                log_msg += f" | Aux: {avg_aux:.6f}"
            log_msg += f" | {speed:.1f} batch/s | {elapsed:.0f}s"
            print(log_msg)

            if exp_config.use_moe:
                stats = get_experiment_expert_stats(model)
                for layer_name, s in list(stats.items())[:2]:
                    ratios = s.get('expert_ratios', [])
                    if ratios:
                        print(f"    {layer_name}: E_ratios={[f'{r:.2f}' for r in ratios]} "
                              f"balance={s.get('balance_score', 0):.3f}")

        if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
            print(f"  [Debug] max_batches={args.max_batches} reached")
            break

    return {
        'pred_loss': total_pred_loss / max(num_batches, 1),
        'aux_loss': total_aux_loss / max(num_batches, 1),
        'time': time.time() - t0,
        'speed': num_batches / max(time.time() - t0, 1e-6),
    }


@torch.no_grad()
def validate(model, loader, device, args, exp_config):
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
            loss = F.l1_loss(y_target[:, :min_len], y_hat[:, :min_len]) + F.mse_loss(y_target[:, :min_len], y_hat[:, :min_len])

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_experiment(exp_name, args):
    exp_config = EXPERIMENTS[exp_name]

    if args.num_experts is not None:
        exp_config.num_experts = args.num_experts
    if args.top_k is not None:
        exp_config.top_k = args.top_k

    print(f"\n{'#'*60}")
    print(f"# 实验: {exp_config.name} (加速版)")
    print(f"# 配置: {exp_config}")
    print(f"# 加速: cudnn.benchmark=True | TF32 | "
          f"workers={args.num_workers} | cache={'ON' if not args.no_cache else 'OFF'}")
    print(f"{'#'*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(args, exp_config, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler('cuda') if (args.use_amp and args.amp_dtype == 'fp16') else None
    print(f"AMP: {'ON' if args.use_amp else 'OFF'} | dtype={args.amp_dtype} | GradScaler={'ON' if scaler else 'OFF'}")

    try:
        train_loader, val_loader = create_dataloaders(args)
    except Exception as e:
        print(f"⚠ 数据加载失败: {e}")
        return float('inf')

    exp_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    log_path = os.path.join(exp_ckpt_dir, 'training_log.json')
    training_log = []

    start_epoch = 0
    best_val_loss = float('inf')

    # 恢复训练: 优先用 latest_model.pt (最新), 其次用 --resume 指定的
    latest_path = os.path.join(exp_ckpt_dir, 'latest_model.pt')
    resume_path = args.resume
    if resume_path is None and os.path.exists(latest_path):
        resume_path = latest_path
        print(f"发现 latest_model.pt, 自动恢复")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"从 epoch {start_epoch} 恢复训练 (val_loss={best_val_loss:.4f})")

    # 恢复已有训练日志
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                training_log = json.load(f)
            print(f"加载已有训练日志: {len(training_log)} 条记录")
        except:
            training_log = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch+1, args, exp_config
        )

        if args.skip_val:
            val_loss = train_metrics['pred_loss']
            print("  [skip_val]")
        else:
            val_loss = validate(model, val_loader, device, args, exp_config)

        scheduler.step()

        epoch_log = {
            'epoch': epoch + 1,
            'train_pred_loss': train_metrics['pred_loss'],
            'train_aux_loss': train_metrics['aux_loss'],
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': train_metrics['time'],
            'speed': train_metrics['speed'],
        }
        training_log.append(epoch_log)

        print(f"  Train Loss: {train_metrics['pred_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Speed: {train_metrics['speed']:.1f} batch/s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {train_metrics['time']:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': exp_config.__dict__,
            }, os.path.join(exp_ckpt_dir, 'best_model.pt'))
            print(f"  ★ Best model! Val Loss: {val_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(exp_ckpt_dir, f'epoch_{epoch+1}.pt'))

        # 每个 epoch 都保存 latest_model.pt (断电恢复用)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(exp_ckpt_dir, 'latest_model.pt'))

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    del model
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"实验 {exp_name} 完成! Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    return best_val_loss


def main():
    args = get_args()

    print("=" * 60)
    print("DATSwinLSTM-Memory 加速版训练")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"float32_matmul_precision: medium (TF32)")
    print(f"num_workers: {args.num_workers} | RAM cache: {'ON' if not args.no_cache else 'OFF'}")

    if args.exp == 'all':
        # 只跑 Flash 实验 7-12 (倒序)
        exp_order = [
            "exp12_balanced_moe_rope_flash",
            "exp11_swiglu_moe_rope_flash",
            "exp10_moe_rope_flash",
            "exp9_balanced_moe_flash",
            "exp8_swiglu_moe_flash",
            "exp7_moe_flash",
        ]
        results = {}
        for i, exp_name in enumerate(exp_order, 1):
            print(f"\n>>> [{i}/{len(exp_order)}] {exp_name}")
            try:
                val_loss = run_experiment(exp_name, args)
                results[exp_name] = val_loss
            except Exception as e:
                print(f"\n✗ {exp_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[exp_name] = float('inf')

        print("\n" + "=" * 60)
        print("实验结果汇总")
        print("=" * 60)
        for exp_name, val_loss in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {exp_name:35s} | Val Loss: {val_loss:.4f}")

        with open(os.path.join(args.checkpoint_dir, 'experiment_summary_fast.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        run_experiment(args.exp, args)


if __name__ == '__main__':
    main()
