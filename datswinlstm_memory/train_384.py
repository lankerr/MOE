"""
DATSwinLSTM Memory 训练脚本 - 384x384 全分辨率版本
适用于 8GB 显卡 (RTX 5070 等)
训练峰值显存: ~7.26 GB

与 train.py 区别:
- embed_dim: 64 (原 128)
- depths: [2,2] (原 [3,2])
- 可在 8GB 显卡运行
"""

import os
import sys
import torch
import torch.nn as nn
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
    parser = argparse.ArgumentParser(description='Train DATSwinLSTM Memory - 384x384 配置 (8GB 显卡)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='\u6700\u5927 epoch \u6570 (\u914d\u5408 early stopping \u4f7f\u7528)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # 数据参数 - 384x384 全分辨率
    parser.add_argument('--input_img_size', type=int, default=384)
    parser.add_argument('--in_len', type=int, default=8, help='输入帧数')
    parser.add_argument('--out_len', type=int, default=12, help='预测帧数')
    parser.add_argument('--memory_len', type=int, default=24, help='长期记忆帧数')
    parser.add_argument('--seq_len', type=int, default=20, help='总序列长度 (in_len + out_len)')
    
    # 模型参数 - 轻量结构保证 8GB 能跑
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=64, help='64 以适应 8GB (原 128)')
    parser.add_argument('--depths_down', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--depths_up', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--heads_number', type=int, nargs='+', default=[4, 4])
    parser.add_argument('--window_size', type=int, default=4)
    
    # Memory 模块参数
    parser.add_argument('--memory_channel_size', type=int, default=256)
    
    # 保存路径
    parser.add_argument('--save_dir', type=str, default='./checkpoints/384x384')
    parser.add_argument('--resume', type=str, default=None, help='断电恢复训练的检查点路径 (例如 ./checkpoints/384x384/best_model.pth)')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--accum_steps', type=int, default=4, help='梯度累加步数，模拟更大的 batch size')
    
    # Early Stopping 熔断停止机制
    parser.add_argument('--patience', type=int, default=15, help='连续多少个 epoch 没有改善就停止 (0=禁用)')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='最小改善阈值，低于此值视为无改善')
    
    return parser.parse_args()


def create_dataloaders(args):
    """创建数据加载器"""
    sevir_paths = cfg.get_sevir_paths()
    
    # 训练集: 2017/6/13 - 2017/8/15
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
    
    # 验证集: 2017/8/15 - 2017/9/15
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    args = get_args()
    
    print("=" * 70)
    print("DATSwinLSTM Memory 训练 - 384x384 (8GB 显卡版)")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 数据集
    print(f"\n加载 SEVIR 数据...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 模型
    print(f"\n初始化模型...")
    print(f"  分辨率: {args.input_img_size}x{args.input_img_size}")
    print(f"  embed_dim: {args.embed_dim} (原 128)")
    print(f"  depths: {args.depths_down} (原 [3,2])")
    print(f"  heads: {args.heads_number}")
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")
    print(f"  预估显存: ~7.26 GB")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0  # Early Stopping 计数器
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f"\n[*] 正在恢复训练...")
        print(f"  加载检查点: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"  成功恢复! 开始训练从 Epoch {start_epoch+1}")
        print(f"  当前最佳 val_loss: {best_val_loss:.6f}, 耐心计数: {patience_counter}/{args.patience}")
        
    # 训练循环
    print(f"\n开始训练...")
    print(f"Max Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"AMP: {args.amp}")
    if args.patience > 0:
        print(f"Early Stopping: patience={args.patience}, min_delta={args.min_delta}")
    else:
        print(f"Early Stopping: 已禁用")
    print("-" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(train_loader):
            # data: [B, T, C, H, W]
            data = data.to(device, non_blocking=True)
            
            # 分割输入和目标
            input_seq = data[:, :args.in_len]        # [B, 8, 1, H, W]
            target_seq = data[:, args.in_len:]       # [B, 12, 1, H, W]
            
            # 长期记忆 = 输入序列 + padding
            memory_seq = input_seq.repeat(1, 3, 1, 1, 1)  # [B, 24, 1, H, W]
            
            if args.amp:
                with torch.amp.autocast('cuda'):
                    output = model(input_seq, memory_seq, phase=2)
                    if isinstance(output, list):
                        output = torch.stack(output, dim=1)
                    # 只取最后 out_len 帧（预测帧）
                    output = output[:, -args.out_len:]
                    loss = criterion(output, target_seq)
                    loss_for_accum = loss / args.accum_steps
                
                scaler.scale(loss_for_accum).backward()
                
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
                # 只取最后 out_len 帧（预测帧）
                output = output[:, -args.out_len:]
                loss = criterion(output, target_seq)
                loss_for_accum = loss / args.accum_steps
                
                loss_for_accum.backward()
                
                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % args.print_freq == 0:
                current_mem = torch.cuda.memory_allocated() / 1e9
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"显存: {current_mem:.2f}/{peak_mem:.2f} GB")
        
        train_loss /= len(train_loader)+1e-8
        epoch_time = time.time() - start_time
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                input_seq = data[:, :args.in_len]
                target_seq = data[:, args.in_len:]
                memory_seq = input_seq.repeat(1, 3, 1, 1, 1)
                
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
        
        val_loss /= len(val_loader)+1e-8
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {train_loss:.4f} Val: {val_loss:.4f} "
              f"LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
        
        # ========== Early Stopping 熔断判定 ==========
        improvement = best_val_loss - val_loss
        
        if improvement > args.min_delta:
            # 有显著改善 -> 重置耐心计数器
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'patience_counter': patience_counter,
                'args': args
            }, save_path)
            print(f"  [BEST] 保存最佳模型: val_loss={val_loss:.6f} (改善: {improvement:.6f})")
        else:
            # 无改善 -> 计数 +1
            patience_counter += 1
            print(f"  [WAIT] 无显著改善 (delta={improvement:.6f} < min_delta={args.min_delta}), "
                  f"耐心: {patience_counter}/{args.patience}")
        
        # 定期保存检查点（包含 patience_counter 用于断点恢复）
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'patience_counter': patience_counter,
                'args': args
            }, save_path)
        
        # 到达耐心极限 -> 触发熔断停止
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n" + "!" * 70)
            print(f"[EARLY STOP] 连续 {args.patience} 个 epoch 无改善, 触发熔断停止!")
            print(f"  最佳验证损失: {best_val_loss:.6f}")
            print(f"  停止于 Epoch: {epoch + 1}")
            print("!" * 70)
            break
    
    print("=" * 70)
    if args.patience > 0 and patience_counter >= args.patience:
        print(f"训练被 Early Stopping 熔断停止 (连续 {args.patience} 轮无改善)")
    else:
        print(f"训练完成 (达到最大 epoch 数: {args.epochs})")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
