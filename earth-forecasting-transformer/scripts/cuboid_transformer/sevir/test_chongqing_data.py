"""
重庆数据冒烟测试
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import Chongqing DataModule
import sys
sys.path.insert(0, os.path.dirname(__file__))
from chongqing_datamodule import ChongqingRadarDataset

print("="*60)
print("重庆数据冒烟测试")
print("="*60)

# Test 1: Dataset 初始化
print("\n[Test 1] Dataset 初始化")
data_dir = r"C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple"

try:
    train_dataset = ChongqingRadarDataset(
        data_dir=data_dir,
        mode='train',
        in_len=24,
        out_len=24,
        img_size=(384, 384),  # 使用实际尺寸
        stride=24,
    )
    print("[OK] Train dataset created")
    print(f"  Samples: {len(train_dataset)}")

    val_dataset = ChongqingRadarDataset(
        data_dir=data_dir,
        mode='val',
        in_len=24,
        out_len=24,
        img_size=(384, 384),
        stride=24,
    )
    print(f"[OK] Val dataset: {len(val_dataset)} samples")

    test_dataset = ChongqingRadarDataset(
        data_dir=data_dir,
        mode='test',
        in_len=24,
        out_len=24,
        img_size=(384, 384),
        stride=24,
    )
    print(f"[OK] Test dataset: {len(test_dataset)} samples")

except Exception as e:
    print(f"[X] Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: 数据加载
print("\n[Test 2] 数据加载")
try:
    x, y = train_dataset[0]
    print(f"[OK] Sample loaded")
    print(f"  Input shape: {x.shape}")  # (T, H, W, C)
    print(f"  Output shape: {y.shape}")
    print(f"  Input dtype: {x.dtype}")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")

except Exception as e:
    print(f"[X] Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: DataLoader
print("\n[Test 3] DataLoader")
try:
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(train_loader))
    print(f"[OK] Batch loaded")
    print(f"  Batch input shape: {batch_x.shape}")  # (B, T, H, W, C)
    print(f"  Batch output shape: {batch_y.shape}")

except Exception as e:
    print(f"[X] DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: 模型前向传播
print("\n[Test 4] 模型前向传播")
try:
    from earthformer.cuboid_transformer.cuboid_transformer_model import CuboidTransformerModel

    model_config = {
        'input_shape': [24, 384, 384, 1],
        'target_shape': [24, 384, 384, 1],
        'base_units': 64,
        'enc_depth': [2, 2],
        'dec_depth': [2, 2],
        'enc_use_inter_ffn': True,
        'dec_use_inter_ffn': True,
        'downsample': 2,
        'downsample_type': 'patch_merge',
        'upsample_type': 'upsample',
        'num_global_vectors': 8,
        'use_dec_self_global': False,
        'dec_self_update_global': True,
        'self_pattern': 'axial',
        'cross_self_pattern': 'axial',
        'cross_pattern': 'cross_1x1',
    }

    model = CuboidTransformerModel(**model_config)

    # Forward pass
    with torch.no_grad():
        output = model(batch_x)

    print(f"[OK] Forward pass successful")
    print(f"  Output shape: {output.shape}")

except Exception as e:
    print(f"[X] Model forward failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("冒烟测试完成！")
print("="*60)
print("\n下一步: 运行完整训练")
print("  python train_baseline_experiments.py --exp chongqing_2h_to_2h --test_run")
