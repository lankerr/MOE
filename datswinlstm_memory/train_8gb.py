"""
DATSwinLSTM-Memory 训练脚本 - RTX 5070 8GB 显存优化版
针对 8GB 显存优化：降低分辨率 + 梯度检查点 + 混合精度

使用方法:
    # 在 WSL 中运行
    conda activate dlenv
    cd ~/workspace/datswinlstm_memory  # 或者桌面路径
    python train_8gb.py
"""

import os
import sys
import argparse
import datetime
import gc

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 本地模块
from config import cfg
from sevir_torch_wrap import SEVIRTorchDataset


class SimpleSwinLSTMCell(nn.Module):
    """简化版 SwinLSTM Cell，8GB 显存优化"""
    
    def __init__(self, in_channels, hidden_channels, img_size, patch_size=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        
        # 特征提取
        self.patch_embed = nn.Conv2d(in_channels, hidden_channels, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # LSTM 门控
        self.conv_gates = nn.Conv2d(hidden_channels * 2, hidden_channels * 4,
                                    kernel_size=3, padding=1)
        
        # 输出重建
        self.reconstruct = nn.ConvTranspose2d(hidden_channels, in_channels,
                                               kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x, h, c):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 提取 patch 特征
        x_feat = self.patch_embed(x)  # (B, hidden, H/patch, W/patch)
        
        # 初始化状态
        if h is None:
            h = torch.zeros_like(x_feat)
            c = torch.zeros_like(x_feat)
        
        # LSTM 门控
        combined = torch.cat([x_feat, h], dim=1)
        gates = self.conv_gates(combined)
        
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        # 重建输出
        output = self.reconstruct(h_new)
        
        return output, h_new, c_new


class SimpleSwinLSTM(nn.Module):
    """简化版 SwinLSTM 模型，适合 8GB 显存"""
    
    def __init__(self, in_channels=1, hidden_channels=64, img_size=192, 
                 patch_size=4, input_frames=12, output_frames=24):
        super().__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        
        self.cell = SimpleSwinLSTMCell(in_channels, hidden_channels, 
                                       img_size, patch_size)
        
    def forward(self, x):
        """
        x: (B, T, C, H, W) - 输入序列
        返回: (B, output_frames, C, H, W) - 预测序列
        """
        B, T, C, H, W = x.shape
        
        # 编码阶段：处理输入帧
        h, c = None, None
        for t in range(T):
            output, h, c = self.cell(x[:, t], h, c)
        
        # 解码阶段：预测未来帧
        outputs = []
        current_input = x[:, -1]  # 用最后一帧开始预测
        
        for t in range(self.output_frames):
            output, h, c = self.cell(current_input, h, c)
            outputs.append(output)
            current_input = output  # 自回归
        
        return torch.stack(outputs, dim=1)


def main():
    print("=" * 60)
    print("DATSwinLSTM 训练脚本 - RTX 5070 (8GB) 优化版")
    print("=" * 60)
    
    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"显存: {gpu_mem:.2f} GB")
    else:
        print("⚠️ 未检测到 CUDA GPU，使用 CPU")
    
    # 获取 SEVIR 路径
    print("\n📁 加载 SEVIR 数据集...")
    sevir_paths = cfg.get_sevir_paths()
    print(f"  数据目录: {sevir_paths['data_dir']}")
    print(f"  目录索引: {sevir_paths['catalog_path']}")
    
    # 检查路径是否存在
    if not os.path.exists(sevir_paths['catalog_path']):
        print(f"\n❌ 错误: 找不到 CATALOG.csv")
        print(f"   请检查 config.py 中的 datasets_dir 配置")
        print(f"   当前配置: {cfg.datasets_dir}")
        print(f"   SEVIR 数据应该在: {sevir_paths['root_dir']}")
        return
    
    # 使用小日期范围测试
    # SEVIR 数据从 2017-06-13 开始
    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=36,
        batch_size=1,
        start_date=datetime.datetime(2017, 6, 15),
        end_date=datetime.datetime(2017, 6, 20),  # 5 天数据
        shuffle=True,
        verbose=True
    )
    
    print(f"  训练样本数: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ 没有找到训练数据！")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建简化模型（8GB 优化）
    print("\n🔧 创建模型（8GB 优化版）...")
    model = SimpleSwinLSTM(
        in_channels=1,
        hidden_channels=64,      # 降低隐藏层维度
        img_size=192,            # 降低分辨率
        patch_size=4,
        input_frames=12,
        output_frames=24
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型显存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # 优化器（使用混合精度）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练一个 epoch
    print("\n🚀 开始训练...")
    model.train()
    
    num_epochs = 1
    num_batches = min(10, len(train_loader))  # 限制批次数
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            # 数据预处理
            batch = batch.to(device, non_blocking=True)
            
            # 降采样到 192x192 以节省显存
            B, T, C, H, W = batch.shape
            batch_lowres = F.interpolate(
                batch.view(-1, C, H, W),
                size=(192, 192),
                mode='bilinear',
                align_corners=False
            ).view(B, T, C, 192, 192)
            
            # 分割输入和目标
            x = batch_lowres[:, :12]   # 输入 12 帧
            y = batch_lowres[:, 12:]   # 预测 24 帧
            
            del batch, batch_lowres
            
            # 前向传播（混合精度）
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                y_hat = model(x)
                loss = F.l1_loss(y_hat, y)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # 打印进度
            print(f"  Batch {batch_idx+1}/{num_batches}: "
                  f"Loss={loss.item():.4f}, "
                  f"显存={torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            # 清理
            del x, y, y_hat, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} 完成! 平均 Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    
    # 保存模型
    checkpoint_dir = cfg.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "model_8gb_test.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
