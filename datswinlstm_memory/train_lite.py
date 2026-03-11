"""
DATSwinLSTM-Memory 轻量级训练脚本
适用于 8GB 显存 GPU

核心修改:
- 图像分辨率: 128x128 (原 384x384)
- embed_dim: 64 (原 128)
- 输入序列: 8 帧 (原 12)
- 输出序列: 12 帧 (原 24)
"""

import os
import sys
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from config import cfg
from sevir_torch_wrap import SEVIRTorchDataset
from models.DATSwinLSTM_D_Memory import Memory


def get_args():
    parser = argparse.ArgumentParser(description='DATSwinLSTM-Memory Lite Training')
    
    # 轻量级参数
    parser.add_argument('--img_size', type=int, default=128, help='Image size (default: 128)')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dim (default: 64)')
    parser.add_argument('--input_frames', type=int, default=8, help='Input frames')
    parser.add_argument('--output_frames', type=int, default=12, help='Output frames')
    parser.add_argument('--seq_len', type=int, default=24, help='Total sequence length')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_lite')
    
    return parser.parse_args()


def create_dataloaders(args):
    """创建数据加载器"""
    sevir_paths = cfg.get_sevir_paths()
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader


def resize_batch(batch, target_size):
    """将 batch 缩放到目标分辨率"""
    B, T, C, H, W = batch.shape
    if H == target_size and W == target_size:
        return batch
    
    # 逐帧缩放
    batch = batch.view(B * T, C, H, W)
    batch = F.interpolate(batch, size=(target_size, target_size), mode='bilinear', align_corners=False)
    batch = batch.view(B, T, C, target_size, target_size)
    return batch


def create_model(args, device):
    """创建轻量级模型"""
    model_args = argparse.Namespace(
        input_img_size=args.img_size,
        patch_size=4,
        input_channels=1,
        embed_dim=args.embed_dim,
        depths_down=[2, 2],  # 减少层数
        depths_up=[2, 2],
        heads_number=[4, 4],  # 减少 attention heads
        window_size=4,
        out_len=args.output_frames
    )
    
    model = Memory(
        model_args,
        memory_channel_size=256,  # 减少 memory 大小
        short_len=args.input_frames,
        long_len=args.seq_len
    )
    
    return model.to(device)


def train_epoch(model, loader, optimizer, scaler, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        # 缩放到目标分辨率
        batch = resize_batch(batch, args.img_size)
        
        x = batch[:, :args.input_frames]
        
        optimizer.zero_grad()
        
        # Phase 1: 记忆存储
        model.set_memory_bank_requires_grad(True)
        
        with autocast('cuda'):
            y_hat = model(x, batch, phase=1)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            y_combined = batch[:, 1:]
            # 匹配输出长度
            min_len = min(y_hat.shape[0], y_combined.shape[1])
            loss_phase1 = F.l1_loss(y_combined[:, :min_len], y_hat[:min_len].permute(1, 0, 2, 3, 4))
        
        scaler.scale(loss_phase1).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Phase 2: 预测
        model.set_memory_bank_requires_grad(False)
        
        with autocast('cuda'):
            y_hat = model(x, x, phase=2)
            if isinstance(y_hat, list):
                y_hat = torch.stack(y_hat)
            loss_phase2 = F.l1_loss(y_combined[:, :min_len], y_hat[:min_len].permute(1, 0, 2, 3, 4))
        
        scaler.scale(loss_phase2).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += (loss_phase1.item() + loss_phase2.item())
        num_batches += 1
        
        if batch_idx % 20 == 0:
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {(loss_phase1.item() + loss_phase2.item()):.4f}, '
                  f'GPU: {mem_used:.2f}GB')
        
        # 清理缓存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


def validate(model, loader, device, args):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            batch = resize_batch(batch, args.img_size)
            
            x = batch[:, :args.input_frames]
            
            with autocast('cuda'):
                y_hat = model(x, x, phase=2)
                if isinstance(y_hat, list):
                    y_hat = torch.stack(y_hat)
                y_combined = batch[:, 1:]
                min_len = min(y_hat.shape[0], y_combined.shape[1])
                loss = F.l1_loss(y_combined[:, :min_len], y_hat[:min_len].permute(1, 0, 2, 3, 4))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Image size: {args.img_size}x{args.img_size}')
    print(f'embed_dim: {args.embed_dim}')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print('Creating dataloaders...')
    train_loader, val_loader = create_dataloaders(args)
    print(f'Train: {len(train_loader)} batches, Val: {len(val_loader)} batches')
    
    print('Creating model...')
    model = create_model(args, device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {params:,}')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n=== Epoch {epoch}/{args.epochs} ===')
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        print(f'Train Loss: {train_loss:.4f}')
        
        val_loss = validate(model, val_loader, device, args)
        print(f'Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f'Saved best model (val_loss: {val_loss:.4f})')
    
    print('\n Training completed!')


if __name__ == '__main__':
    main()
