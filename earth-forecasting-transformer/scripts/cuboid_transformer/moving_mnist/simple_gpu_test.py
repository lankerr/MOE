"""
Simple GPU test script for Earthformer on MovingMNIST (Fixed ver 2)
Smaller model and batch size for testing
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("Earthformer GPU Test - MovingMNIST (Small Config)")
print("=" * 60)

# Check GPU
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Clear cache
    torch.cuda.empty_cache()

# Import Earthformer components
print("\nLoading Earthformer model...")
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.moving_mnist.moving_mnist import MovingMNISTDataModule

# Smaller model config for testing
model_config = {
    'input_shape': (10, 64, 64, 1),
    'target_shape': (10, 64, 64, 1),
    'base_units': 32,  # Reduced from 64
    'block_units': None,
    'scale_alpha': 1.0,
    'enc_depth': [2, 2],  # Reduced from [4, 4]
    'dec_depth': [2, 2],  # Reduced from [4, 4]
    'enc_use_inter_ffn': True,
    'dec_use_inter_ffn': True,
    'dec_hierarchical_pos_embed': False,
    'downsample': 2,
    'downsample_type': 'patch_merge',
    'upsample_type': 'upsample',
    'enc_attn_patterns': ['axial', 'axial'],
    'dec_self_attn_patterns': ['axial', 'axial'],
    'dec_cross_attn_patterns': ['cross_1x1', 'cross_1x1'],
    'dec_cross_last_n_frames': None,
    'dec_use_first_self_attn': False,
    'num_heads': 4,
    'attn_drop': 0.0,  # Reduced for testing
    'proj_drop': 0.0,
    'ffn_drop': 0.0,
    'ffn_activation': 'gelu',
    'gated_ffn': False,
    'norm_layer': 'layer_norm',
    'num_global_vectors': 0,
    'use_dec_self_global': False,
    'dec_self_update_global': True,
    'use_dec_cross_global': False,
    'use_global_vector_ffn': False,
    'use_global_self_attn': False,
    'separate_global_qkv': False,
    'global_dim_ratio': 1,
    'initial_downsample_type': 'conv',
    'initial_downsample_activation': 'leaky',
    'initial_downsample_scale': 2,
    'initial_downsample_conv_layers': 2,
    'final_upsample_conv_layers': 1,
    'padding_type': 'zeros',
    'z_init_method': 'zeros',
    'checkpoint_level': 0,
    'pos_embed_type': 't+hw',
    'use_relative_pos': True,
    'self_attn_use_final_proj': True,
    'attn_linear_init_mode': '0',
    'ffn_linear_init_mode': '0',
    'conv_init_mode': '0',
    'down_up_linear_init_mode': '0',
    'norm_init_mode': '0',
}

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = CuboidTransformerModel(**model_config)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass first
print("\nTesting forward pass with dummy data...")
with torch.no_grad():
    dummy_input = torch.randn(1, 10, 64, 64, 1).to(device)  # batch_size=1
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print("Forward pass successful!")

if torch.cuda.is_available():
    print(f"GPU Memory after forward: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Load data
print("\nLoading MovingMNIST dataset...")
dm = MovingMNISTDataModule(batch_size=1)  # Smaller batch size
dm.prepare_data()
dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()

print(f"Train samples: {dm.num_train_samples}")
print(f"Val samples: {dm.num_val_samples}")
print(f"Test samples: {dm.num_test_samples}")

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
print("\n" + "=" * 60)
print("Starting Training...")
print("=" * 60)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # x, y shape: (B, T, H, W, C)
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            if torch.cuda.is_available():
                print(f"  GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Stop early for testing
        if batch_idx >= 50:
            break
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            val_loss += loss.item()
            val_batches += 1
            if batch_idx >= 10:
                break
    
    avg_val_loss = val_loss / val_batches
    print(f"Validation Loss: {avg_val_loss:.6f}")
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
