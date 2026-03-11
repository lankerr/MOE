"""
CPU-only test script for Earthformer to verify model works
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
import torch.nn.functional as F

print("=" * 60)
print("Earthformer CPU Test - MovingMNIST")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Force CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Import Earthformer components
print("\nLoading Earthformer model...")
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.moving_mnist.moving_mnist import MovingMNISTDataModule

# Small model config for testing
model_config = {
    'input_shape': (10, 64, 64, 1),
    'target_shape': (10, 64, 64, 1),
    'base_units': 16,  # Very small for CPU testing
    'block_units': None,
    'scale_alpha': 1.0,
    'enc_depth': [1, 1],  # Minimal depth
    'dec_depth': [1, 1],
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
    'num_heads': 2,  # Reduced heads
    'attn_drop': 0.0,
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

print("\nCreating model...")
model = CuboidTransformerModel(**model_config)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\nTesting forward pass with dummy data...")
with torch.no_grad():
    dummy_input = torch.randn(1, 10, 64, 64, 1).to(device)
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print("Forward pass successful!")

# Load real data
print("\nLoading MovingMNIST dataset...")
dm = MovingMNISTDataModule(batch_size=1)
dm.prepare_data()
dm.setup()

train_loader = dm.train_dataloader()

print(f"Train samples: {dm.num_train_samples}")
print(f"Val samples: {dm.num_val_samples}")

# Training loop - just a few iterations
print("\n" + "=" * 60)
print("Starting Training (CPU mode)...")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

model.train()
for batch_idx, (x, y) in enumerate(train_loader):
    x = x.to(device)
    y = y.to(device)
    
    optimizer.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    
    print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    if batch_idx >= 5:  # Only a few batches on CPU
        break

print("\n" + "=" * 60)
print("CPU Test completed successfully!")
print("Earthformer model is working correctly.")
print("=" * 60)
