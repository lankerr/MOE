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
    parser.add_argument('--epochs', type=int, default=50)
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
        start_date=datetime.datetime(2017, 1, 1),
        end_date=datetime.datetime(2019, 6, 1),
        shuffle=True,
        verbose=True
    )
    
    # 验证集: 2017/8/15 - 2017/9/15
    val_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2019, 6, 1),
        end_date=datetime.datetime(2019, 12, 1),
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
        print(f"  成功恢复! 开始训练从 Epoch {start_epoch+1}")
        
    # 训练循环
    print(f"\n开始训练...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"AMP: {args.amp}")
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
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_path)
            print(f"  ✅ 保存最佳模型: {save_path}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_path)
    
    print("=" * 70)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
