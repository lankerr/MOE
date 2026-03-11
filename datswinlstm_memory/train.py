"""
DATSwinLSTM-Memory 简化训练脚本
适用于智算平台部署

使用方法:
    python train.py                    # 默认训练
    python train.py --batch_size 2     # 自定义批次大小
    python train.py --epochs 50        # 自定义训练轮数
"""

import os
import sys
import argparse
import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

# 本地模块
from config import cfg
from sevir_torch_wrap import SEVIRTorchDataset
from sevir import SEVIRSkillScore

# [FIX] Import path correction
try:
    from models.DATSwinLSTM_D_Memory import Memory
except ImportError:
    from models.DATSwinLSTM_D_Memory.DATSwinLSTM_D_Memory import Memory


def get_args():
    parser = argparse.ArgumentParser(description='DATSwinLSTM-Memory Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory (default: from config.py)')
    parser.add_argument('--seq_len', type=int, default=36,
                        help='Sequence length')
    parser.add_argument('--input_frames', type=int, default=12,
                        help='Number of input frames')
    parser.add_argument('--output_frames', type=int, default=24,
                        help='Number of output frames')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size')
    parser.add_argument('--window_size', type=int, default=4,
                        help='Window size')
    
    # 其他
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluation only')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training (AMP)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    
    return parser.parse_args()


def create_dataloaders(args):
    """创建数据加载器"""
    # 获取 SEVIR 路径
    sevir_paths = cfg.get_sevir_paths()
    
    # 2017 年数据从 6 月 13 日开始
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
    
    test_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        start_date=datetime.datetime(2017, 9, 15),
        end_date=datetime.datetime(2017, 11, 1),
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_model(args, device):
    """创建模型"""
    model_args = argparse.Namespace(
        input_img_size=384,
        patch_size=args.patch_size,
        input_channels=1,
        embed_dim=args.embed_dim,
        depths_down=[3, 2],
        depths_up=[2, 3],
        heads_number=[4, 8],
        window_size=args.window_size,
        out_len=args.output_frames
    )
    
    model = Memory(
        model_args,
        memory_channel_size=512,
        short_len=args.input_frames,
        long_len=args.seq_len
    )
    
    return model.to(device)


def train_epoch(model, loader, optimizer, device, epoch, scaler=None, use_amp=False):
    """训练一个 epoch（支持混合精度）"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        # 分割输入和输出
        x = batch[:, :12, :, :, :]  # 输入 12 帧
        y = batch[:, 12:, :, :, :]  # 预测后 24 帧
        
        optimizer.zero_grad()
        
        # Phase 1: 记忆存储
        model.set_memory_bank_requires_grad(True)
        
        with autocast('cuda', enabled=use_amp):
            y_hat = model(x, batch, phase=1)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            y_combined = batch[:, 1:, :, :, :]
            loss_phase1 = F.l1_loss(y_combined, y_hat)
        
        if scaler:
            scaler.scale(loss_phase1).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_phase1.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Phase 2: 预测
        model.set_memory_bank_requires_grad(False)
        
        with autocast('cuda', enabled=use_amp):
            y_hat = model(x, x, phase=2)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            loss_phase2 = F.l1_loss(y_combined, y_hat)
        
        if scaler:
            scaler.scale(loss_phase2).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_phase2.backward()
            optimizer.step()
        
        total_loss += (loss_phase1.item() + loss_phase2.item())
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {(loss_phase1.item() + loss_phase2.item()):.4f}')
        
        # 清理 GPU 缓存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


def validate(model, loader, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    scorer = SEVIRSkillScore(metrics_list=['csi', 'pod']).to(device)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            x = batch[:, :12, :, :, :]
            y = batch[:, 12:, :, :, :]
            
            y_hat = model(x, x, phase=2)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            
            y_combined = batch[:, 1:, :, :, :]
            loss = F.l1_loss(y_combined, y_hat)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新评估指标
            scorer.update(y_hat[:, -24:], y)
    
    metrics = scorer.compute()
    avg_loss = total_loss / max(num_batches, 1)
    
    return avg_loss, metrics


def main():
    torch.set_float32_matmul_precision('medium')
    args = get_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建 checkpoint 目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 创建数据加载器
    print('Creating dataloaders...')
    train_loader, val_loader, test_loader = create_dataloaders(args)
    print(f'Train: {len(train_loader)} batches')
    print(f'Val: {len(val_loader)} batches')
    print(f'Test: {len(test_loader)} batches')
    
    # 创建模型
    print('Creating model...')
    model = create_model(args, device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 恢复检查点
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # 评估模式
    if args.eval_only:
        val_loss, metrics = validate(model, test_loader, device)
        print(f'Test Loss: {val_loss:.4f}')
        print('Metrics:', metrics)
        return
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=20, min_lr=5e-5
    )
    
    # 混合精度训练 Scaler
    scaler = GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print('Using mixed precision training (AMP)')
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n=== Epoch {epoch + 1}/{args.epochs} ===')
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1, scaler, args.use_amp)
        print(f'Train Loss: {train_loss:.4f}')
        
        # 验证
        val_loss, metrics = validate(model, val_loader, device)
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val CSI: {metrics.get("csi_avg", 0):.4f}')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()
