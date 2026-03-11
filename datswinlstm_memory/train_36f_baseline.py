"""
DATSwinLSTM Memory 训练脚本 - 36帧 Flash Attention 基线版本
=========================================================

与 20帧基线 (train_384.py) 的区别:
- seq_len:      20 → 36           (总序列长度)
- in_len:        8 → 12           (输入帧数)  
- out_len:      12 → 24           (预测帧数)
- memory_len:   24 → 36           (长期记忆帧数)
- Flash Attention (SDPA) 默认开启  (WindowAttention + Memory Attention)
- MotionEncoder2D 自动分块处理 36帧 (chunk_size=20)

网络结构与 20帧基线完全一致:
- embed_dim=64, depths=[2,2], heads=[4,4]
- memory_channel=256, patch_size=4, window_size=4
- 适用于 8GB 显卡 (RTX 5070 等)
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models.DATSwinLSTM_D_Memory import Memory
from sevir_torch_wrap import SEVIRTorchDataset


def get_args():
    parser = argparse.ArgumentParser(description='Train DATSwinLSTM Memory - 36帧 Flash Attention 基线')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # ===== 36帧数据参数 (核心区别) =====
    parser.add_argument('--input_img_size', type=int, default=384)
    parser.add_argument('--in_len', type=int, default=12, help='输入帧数 (20帧版=8)')
    parser.add_argument('--out_len', type=int, default=24, help='预测帧数 (20帧版=12)')
    parser.add_argument('--memory_len', type=int, default=36, help='长期记忆帧数 (20帧版=24)')
    parser.add_argument('--seq_len', type=int, default=36, help='总序列长度 (20帧版=20)')
    
    # 模型参数 - 与20帧基线完全一致
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--depths_down', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--depths_up', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--heads_number', type=int, nargs='+', default=[4, 4])
    parser.add_argument('--window_size', type=int, default=4)
    
    # Memory 模块参数
    parser.add_argument('--memory_channel_size', type=int, default=256)
    
    # 保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/baseline_36f_flash')
    parser.add_argument('--resume', type=str, default=None)
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='amp')
    parser.add_argument('--accum_steps', type=int, default=4)
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    
    return parser.parse_args()


def create_dataloaders(args):
    """创建数据加载器 (seq_len=36)"""
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


def main():
    args = get_args()
    
    print("=" * 70)
    print("DATSwinLSTM Memory 训练 - 36帧 Flash Attention 基线")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 数据集
    print(f"\n加载 SEVIR 数据 (seq_len={args.seq_len})...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 36帧参数汇总
    print(f"\n36帧配置:")
    print(f"  输入帧数: {args.in_len} (20帧版=8)")
    print(f"  预测帧数: {args.out_len} (20帧版=12)")
    print(f"  序列长度: {args.seq_len} (20帧版=20)")
    print(f"  记忆长度: {args.memory_len} (20帧版=24)")
    
    # 模型
    print(f"\n初始化模型...")
    print(f"  embed_dim: {args.embed_dim}, depths: {args.depths_down}, heads: {args.heads_number}")
    print(f"  Flash Attention (SDPA): ON (默认)")
    
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
    
    model = Memory(
        model_args,
        memory_channel_size=args.memory_channel_size,
        short_len=args.in_len,
        long_len=args.memory_len
    ).to(device)
    
    # Flash Attention 已默认开启 (WindowAttention use_flash=True, Attention use_flash=True)
    # 验证 Flash Attention 状态
    flash_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'use_flash') and module.use_flash:
            flash_count += 1
    print(f"  Flash Attention 模块: {flash_count} 个已启用")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # Loss: L1 + MSE (论文公式12, 与 train_384_opt.py / exp7-12 统一)
    def combined_loss(pred, target):
        return F.l1_loss(pred, target) + F.mse_loss(pred, target)
    criterion = combined_loss
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f"\n[*] 恢复训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"  从 Epoch {start_epoch+1} 恢复, best_val={best_val_loss:.6f}, patience={patience_counter}")
    
    # 训练循环
    print(f"\n开始训练...")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, AMP: {args.amp}")
    print(f"Early Stopping: patience={args.patience}, min_delta={args.min_delta}")
    print("-" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # ---- 训练阶段 ----
        model.train()
        train_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            
            # 36帧: 输入12帧, 预测24帧
            input_seq = data[:, :args.in_len]         # [B, 12, 1, H, W]
            target_seq = data[:, args.in_len:]        # [B, 24, 1, H, W]
            
            # 长期记忆 = 输入序列 repeat 到 memory_len
            repeat_factor = max(1, (args.memory_len + args.in_len - 1) // args.in_len)
            memory_seq = input_seq.repeat(1, repeat_factor, 1, 1, 1)[:, :args.memory_len]
            
            if args.amp:
                with torch.amp.autocast('cuda'):
                    output = model(input_seq, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    output = output[:, -args.out_len:]
                    loss = criterion(output, target_seq)
                    loss_accum = loss / args.accum_steps
                
                scaler.scale(loss_accum).backward()
                
                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(input_seq, memory_seq, phase=2)
                if isinstance(output, list):
                    output = torch.stack(output, dim=1)
                output = output[:, -args.out_len:]
                loss = criterion(output, target_seq)
                loss_accum = loss / args.accum_steps
                
                loss_accum.backward()
                
                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                cur_mem = torch.cuda.memory_allocated() / 1e9
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"显存: {cur_mem:.2f}/{peak_mem:.2f} GB")
        
        train_loss /= len(train_loader) + 1e-8
        epoch_time = time.time() - start_time
        
        # ---- 验证阶段 ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                input_seq = data[:, :args.in_len]
                target_seq = data[:, args.in_len:]
                repeat_factor = max(1, (args.memory_len + args.in_len - 1) // args.in_len)
                memory_seq = input_seq.repeat(1, repeat_factor, 1, 1, 1)[:, :args.memory_len]
                
                if args.amp:
                    with torch.amp.autocast('cuda'):
                        output = model(input_seq, memory_seq, phase=2)
                        if isinstance(output, list):
                            output = torch.stack(output, dim=1)
                        output = output[:, -args.out_len:]
                        loss = criterion(output, target_seq)
                else:
                    output = model(input_seq, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    output = output[:, -args.out_len:]
                    loss = criterion(output, target_seq)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader) + 1e-8
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {train_loss:.4f} Val: {val_loss:.4f} "
              f"LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
        
        # ---- Early Stopping ----
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
                'patience_counter': 0,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  [BEST] val_loss={val_loss:.6f} (改善: {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"  [WAIT] 无改善 (delta={improvement:.6f}), 耐心: {patience_counter}/{args.patience}")
        
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'patience_counter': patience_counter,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth'))
        
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n{'!'*70}")
            print(f"[EARLY STOP] 连续 {args.patience} epoch 无改善!")
            print(f"  最佳: {best_val_loss:.6f}, 停止于 Epoch {epoch+1}")
            print(f"{'!'*70}")
            break
    
    print("=" * 70)
    if args.patience > 0 and patience_counter >= args.patience:
        print(f"训练被 Early Stopping 停止")
    else:
        print(f"训练完成 ({args.epochs} epochs)")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
