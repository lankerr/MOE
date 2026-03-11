"""
EarthFormer 训练脚本 - RTX 5070 8GB 显存优化版
使用 SEVIR 数据集进行训练

使用方法:
    conda activate dlenv
    cd earth-forecasting-transformer
    python train_earthformer_8gb.py
"""

import os
import sys
import datetime
import gc

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加 datswinlstm_memory 路径以使用其 SEVIR 数据加载器
datswin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datswinlstm_memory")
sys.path.insert(0, datswin_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# EarthFormer 模型
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

# SEVIR 数据加载 (从 datswinlstm_memory)
from config import cfg
from sevir_torch_wrap import SEVIRTorchDataset


def main():
    print("=" * 60)
    print("EarthFormer 训练脚本 - RTX 5070 (8GB) 优化版")
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
        print("未检测到 CUDA GPU，使用 CPU")
    
    # 加载 SEVIR 数据
    print("\n加载 SEVIR 数据集...")
    sevir_paths = cfg.get_sevir_paths()
    print(f"  数据目录: {sevir_paths['data_dir']}")
    
    if not os.path.exists(sevir_paths['catalog_path']):
        print(f"错误: 找不到 CATALOG.csv")
        return
    
    # 使用小日期范围
    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_paths['catalog_path'],
        sevir_data_dir=sevir_paths['data_dir'],
        seq_len=36,
        batch_size=1,
        start_date=datetime.datetime(2017, 6, 15),
        end_date=datetime.datetime(2017, 6, 20),
        shuffle=True,
        verbose=True
    )
    
    print(f"  训练样本数: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("没有找到训练数据")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建 EarthFormer 模型（8GB 优化配置）
    # 使用较小的配置以适应 8GB 显存
    print("\n创建 EarthFormer 模型（8GB 优化版）...")
    
    # 降低分辨率到 128x128 以节省显存
    target_size = 128
    input_frames = 12
    output_frames = 24
    
    model = CuboidTransformerModel(
        input_shape=(input_frames, target_size, target_size, 1),  # (T, H, W, C)
        target_shape=(output_frames, target_size, target_size, 1),
        base_units=32,              # 降低基础单元数（原始 128）
        block_units=None,
        scale_alpha=1.0,
        num_heads=4,                # 注意力头数
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        downsample=2,
        downsample_type="patch_merge",
        upsample_type="upsample",
        upsample_kernel_size=3,
        enc_depth=[1, 1],           # 减少编码器深度
        enc_attn_patterns=None,
        enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        enc_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
        enc_shift_size=[(0, 0, 0), (0, 0, 0)],
        enc_use_inter_ffn=True,
        dec_depth=[1, 1],           # 减少解码器深度
        dec_cross_start=0,
        dec_self_attn_patterns=None,
        dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        dec_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
        dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
        dec_cross_attn_patterns=None,
        dec_cross_cuboid_hw=[(4, 4), (4, 4)],
        dec_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
        dec_cross_shift_hw=[(0, 0), (0, 0)],
        dec_cross_n_temporal=[1, 2],
        dec_cross_last_n_frames=None,
        dec_use_inter_ffn=True,
        dec_hierarchical_pos_embed=False,
        num_global_vectors=4,       # 全局向量数
        use_dec_self_global=True,
        dec_self_update_global=True,
        use_dec_cross_global=True,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        z_init_method="nearest_interp",
        initial_downsample_type="conv",
        initial_downsample_activation="leaky",
        initial_downsample_scale=1,
        initial_downsample_conv_layers=2,
        final_upsample_conv_layers=2,
        checkpoint_level=True,      # 使用梯度检查点节省显存
        pos_embed_type="t+hw",
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        dec_use_first_self_attn=False,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_up_linear_init_mode="0",
        norm_init_mode="0",
        ffn_activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        padding_type="ignore",
    )
    
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型显存: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练
    print("\n开始训练...")
    model.train()
    
    num_batches = min(10, len(train_loader))
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        # 数据预处理
        batch = batch.to(device, non_blocking=True)  # (B, T, C, H, W)
        
        # 降采样到目标分辨率
        B, T, C, H, W = batch.shape
        batch_lowres = F.interpolate(
            batch.view(-1, C, H, W),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).view(B, T, C, target_size, target_size)
        
        # 分割输入和目标
        # EarthFormer 需要 (B, T, H, W, C) 格式
        x = batch_lowres[:, :input_frames].permute(0, 1, 3, 4, 2)   # (B, T, H, W, C)
        y = batch_lowres[:, input_frames:].permute(0, 1, 3, 4, 2)   # (B, T, H, W, C)
        
        del batch, batch_lowres
        
        # 前向传播
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            y_hat = model(x)
            loss = F.l1_loss(y_hat, y)
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        print(f"  Batch {batch_idx+1}/{num_batches}: "
              f"Loss={loss.item():.4f}, "
              f"显存={torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # 清理
        del x, y, y_hat, loss
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    print(f"\nEpoch 完成! 平均 Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("EarthFormer 训练验证成功!")
    print("=" * 60)
    
    # 保存模型
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "earthformer_8gb_test.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
