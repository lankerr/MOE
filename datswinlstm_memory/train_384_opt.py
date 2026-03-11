"""
DATSwinLSTM Memory 训练脚本 - 384x384 基线优化版
================================================
与 train_384.py 区别:
  - Loss: L1 + MSE (论文公式12, 与 exp7-12 统一)
  - 加速: cudnn.benchmark + TF32
  - 每 epoch 保存 latest_model.pt (断电恢复)
  - 训练日志 JSON (方便画图对比)
  - 自动检测 latest_model.pt 恢复训练
  - 修复 OOM: memory_seq 对齐为 seq_len=20 帧 (与 exp7-12 一致)

用法:
    conda activate rtx5070_cu128
    python -u train_384_opt.py --epochs 10
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset


def get_args():
    parser = argparse.ArgumentParser(description='Train DATSwinLSTM Memory 384x384 (L1+MSE)')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 数据参数 - 与 train_experiment_fast.py 对齐
    parser.add_argument('--input_img_size', type=int, default=384)
    parser.add_argument('--in_len', type=int, default=8)
    parser.add_argument('--out_len', type=int, default=12)
    parser.add_argument('--seq_len', type=int, default=20, help='总序列长度=in+out')

    # 模型参数 - 与 train_384.py / exp7-12 完全一致
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--depths_down', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--depths_up', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--heads_number', type=int, nargs='+', default=[4, 4])
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--memory_channel_size', type=int, default=256)
    parser.add_argument('--memory_slot_size', type=int, default=100,
                        help='Capacity of memory bank slots (100 for default, 1024 for 49-frame)')

    # 保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/384x384_opt')
    parser.add_argument('--resume', type=str, default=None)

    # 训练控制
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--accum_steps', type=int, default=4)

    # Early Stopping
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-5)

    return parser.parse_args()


def create_dataloaders(args):
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


def combined_loss(pred, target):
    """论文公式(12): Lpred = MAE + MSE (等权1:1)"""
    return F.l1_loss(pred, target) + F.mse_loss(pred, target)


def main():
    args = get_args()

    # ===== 加速优化 =====
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')

    print("=" * 70)
    print("DATSwinLSTM Memory 384x384 基线 (L1+MSE)")
    print("  Loss: L1 + MSE (论文公式12, 与 exp7-12 统一)")
    print("  无 Flash/MoE/RoPE (纯基线)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"显存: {mem:.2f} GB")
        torch.cuda.empty_cache()

    print(f"\n加载 SEVIR 数据...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 模型 — 与 train_experiment_fast.py 完全相同的基础架构
    model_args = argparse.Namespace(
        input_img_size=args.input_img_size,
        patch_size=args.patch_size,
        input_channels=args.input_channels,
        embed_dim=args.embed_dim,
        depths_down=args.depths_down,
        depths_up=args.depths_up,
        heads_number=args.heads_number,
        window_size=args.window_size,
        out_len=args.out_len
    )

    # 与 train_experiment_fast.py 对齐: short_len=in_len, long_len=seq_len
    model = Memory(
        model_args,
        memory_channel_size=args.memory_channel_size,
        memory_slot_size=args.memory_slot_size,
        short_len=args.in_len,
        long_len=args.seq_len  # 注意: 原先旧脚本传的是 args.memory_len, 新脚本传的是 seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'training_log.json')
    training_log = []

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    # ===== 自动恢复 =====
    latest_path = os.path.join(args.save_dir, 'latest_model.pt')
    resume_path = args.resume
    if resume_path is None and os.path.exists(latest_path):
        resume_path = latest_path
        print(f"发现 latest_model.pt, 自动恢复")

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"从 epoch {start_epoch} 恢复 (val_loss={best_val_loss:.4f})")

    # 恢复训练日志
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                training_log = json.load(f)
            print(f"加载已有训练日志: {len(training_log)} 条记录")
        except:
            training_log = []

    print(f"\n开始训练: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"Loss: L1 + MSE | AMP: {args.amp} | Accum: {args.accum_steps}")
    print(f"Memory: in={args.in_len} frames, seq={args.seq_len} frames")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs):
        # ===== 训练 =====
        model.train()
        train_loss_sum = 0.0
        num_train = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)

            # 与 train_experiment_fast.py 完全对齐 (line 216-242)
            x = data[:, :args.in_len]  # [B, 8, 1, 384, 384]
            y_target = data[:, args.in_len:args.in_len + args.out_len]  # [B, 12, 1, 384, 384]

            # memory_seq: 用输入帧 repeat 到 seq_len 帧 (20帧, 不是24!)
            repeat_factor = max(1, (args.seq_len + args.in_len - 1) // args.in_len)
            memory_seq = x.repeat(1, repeat_factor, 1, 1, 1)[:, :args.seq_len]

            if args.amp:
                with torch.amp.autocast('cuda'):
                    output = model(x, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    min_t = min(output.shape[1], y_target.shape[1])
                    loss = combined_loss(output[:, :min_t], y_target[:, :min_t])
                    loss_accum = loss / args.accum_steps

                scaler.scale(loss_accum).backward()
                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(x, memory_seq, phase=2)
                if isinstance(output, list):
                    output = torch.stack(output, dim=1)
                min_t = min(output.shape[1], y_target.shape[1])
                loss = combined_loss(output[:, :min_t], y_target[:, :min_t])
                loss_accum = loss / args.accum_steps
                loss_accum.backward()
                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss_sum += loss.item()
            num_train += 1

            if (batch_idx + 1) % args.print_freq == 0:
                speed = num_train / (time.time() - t0)
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | {speed:.2f} b/s")

        train_loss = train_loss_sum / max(num_train, 1)
        epoch_time = time.time() - t0

        # ===== 验证 =====
        model.eval()
        val_loss_sum = 0.0
        num_val = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                x = data[:, :args.in_len]
                y_target = data[:, args.in_len:args.in_len + args.out_len]
                repeat_factor = max(1, (args.seq_len + args.in_len - 1) // args.in_len)
                memory_seq = x.repeat(1, repeat_factor, 1, 1, 1)[:, :args.seq_len]

                if args.amp:
                    with torch.amp.autocast('cuda'):
                        output = model(x, memory_seq, phase=2)
                        if isinstance(output, list):
                            output = torch.stack(output, dim=1)
                        min_t = min(output.shape[1], y_target.shape[1])
                        loss = combined_loss(output[:, :min_t], y_target[:, :min_t])
                else:
                    output = model(x, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    min_t = min(output.shape[1], y_target.shape[1])
                    loss = combined_loss(output[:, :min_t], y_target[:, :min_t])

                val_loss_sum += loss.item()
                num_val += 1

        val_loss = val_loss_sum / max(num_val, 1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        speed = num_train / epoch_time

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {train_loss:.4f} Val: {val_loss:.4f} "
              f"LR: {current_lr:.2e} Time: {epoch_time:.0f}s ({speed:.2f} b/s)")

        # ===== 训练日志 =====
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "time": epoch_time,
            "speed": speed
        })
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        # ===== Early Stopping =====
        improvement = best_val_loss - val_loss
        if improvement > args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'patience_counter': patience_counter,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  [BEST] val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"  [WAIT] 无改善 ({patience_counter}/{args.patience})")

        # 每 epoch 保存 latest (断电恢复)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'patience_counter': patience_counter,
        }, latest_path)

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n[EARLY STOP] 连续 {args.patience} epoch 无改善!")
            break

    print("=" * 70)
    print(f"训练完成! 最佳 val_loss: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
