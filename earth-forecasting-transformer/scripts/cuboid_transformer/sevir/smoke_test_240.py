"""
Smoke test script for physics attention experiments on RTX 5070.

Usage:
    conda activate rtx5070_CU128
    python smoke_test_240.py
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

print("=" * 60)
print("SMOKE TEST - Physics Attention on RTX 5070")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Environment check
print("Environment Check:")
print("-" * 40)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# Test 1: Module import check
print("Test 1: Module Imports")
print("-" * 40)
try:
    from earthformer.cuboid_transformer.physics_attention import (
        PhysicsGuidedSparseAttention,
        DensityProximityCrossBlockAttention,
        FLASH_ATTENTION_AVAILABLE,
    )
    print("  [OK] Physics attention modules imported")
    print(f"  [OK] Flash Attention available: {FLASH_ATTENTION_AVAILABLE}")
except ImportError as e:
    print(f"  [X] Import failed: {e}")
    sys.exit(1)

# Test 2: GMR modules
print()
print("Test 2: GMR Modules")
print("-" * 40)
try:
    gmr_paths = [
        "scripts.cuboid_transformer.sevir.gmr_layers",
        "scripts.cuboid_transformer.sevir.gmr_patch_embed",
    ]
    for path in gmr_paths:
        try:
            module = __import__(path, fromlist=[''])
            print(f"  [OK] {path}")
        except ImportError:
            print(f"  [!] {path} not found")
except Exception as e:
    print(f"  [X] GMR check failed: {e}")

# Test 3: Forward pass test
print()
print("Test 3: Forward Pass Test")
print("-" * 40)

def test_pgsa_forward():
    """Test PGSA forward pass"""
    B, T, H, W, C = 1, 4, 64, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(B, T, H, W, C, device=device)
    dbz = torch.rand(B, T, H, W, 1, device=device) * 75

    pgsa = PhysicsGuidedSparseAttention(
        dim=C,
        num_heads=4,
        dbz_threshold=15.0,
        masking_mode='hybrid',
    ).to(device)

    start = time.time()
    out = pgsa(x, dbz_values=dbz)
    elapsed = time.time() - start

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"  [OK] PGSA forward: {elapsed*1000:.2f}ms")
    print(f"      Output shape: {out.shape}")
    print(f"      Sparse ratio: {pgsa.sparse_ratio.item():.2%}")

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        print(f"      Peak memory: {mem_allocated:.2f} GB")

    return True

def test_dpcba_forward():
    """Test DPCBA forward pass"""
    B, T, H, W, C = 1, 4, 64, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(B, T, H, W, C, device=device)

    # Note: patch_size is physical patch size, cuboid_size is for forward
    dpcba = DensityProximityCrossBlockAttention(
        dim=C,
        num_heads=4,
        patch_size=(2, 8, 8),
        num_connections=4,
    ).to(device)

    start = time.time()
    out = dpcba(x, cuboid_size=(2, 8, 8))
    elapsed = time.time() - start

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"  [OK] DPCBA forward: {elapsed*1000:.2f}ms")

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        print(f"      Peak memory: {mem_allocated:.2f} GB")

    return True

# Run tests
try:
    test_pgsa_forward()
except Exception as e:
    print(f"  [X] PGSA test failed: {e}")
    import traceback
    traceback.print_exc()

try:
    test_dpcba_forward()
except Exception as e:
    print(f"  [X] DPCBA test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Learning rate scheduler
print()
print("Test 4: WSD Learning Rate Scheduler")
print("-" * 40)

class WSDScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup-Stable-Decay scheduler from 2024 research"""
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.stable_steps:
            return self.base_lrs
        else:
            progress = (self.last_epoch - self.warmup_steps - self.stable_steps) / self.decay_steps
            progress = min(progress, 1.0)
            return [base_lr * (0.01 + 0.99 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2)
                    for base_lr in self.base_lrs]

optimizer = torch.optim.SGD([torch.randn(10, 10)], lr=1e-3)
scheduler = WSDScheduler(optimizer, warmup_steps=10, stable_steps=50, decay_steps=40)

lrs = []
for _ in range(100):
    lrs.append(scheduler.get_last_lr()[0])
    scheduler.step()

print(f"  [OK] WSD scheduler implemented")
print(f"      Initial LR: {lrs[0]:.6f}")
print(f"      Peak LR: {max(lrs):.6f}")
print(f"      Final LR: {lrs[-1]:.6f}")
print(f"      Stable steps: ~{sum(1 for lr in lrs if lr > 0.9 * max(lrs))}")

# Test 5: End-to-end smoke test
print()
print("Test 5: End-to-End Smoke Test")
print("-" * 40)

def end_to_end_smoke_test():
    """Minimal end-to-end test"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=4),
                nn.Conv2d(32, 64, 3, stride=3),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=3),
                nn.ConvTranspose2d(32, 1, 4, stride=4),
            )

        def forward(self, x):
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, C, H, W)
            feat = self.encoder(x)
            out = self.decoder(feat)
            out = out.reshape(B, T, H, W, C)
            return out

    model = DummyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = WSDScheduler(optimizer, warmup_steps=5, stable_steps=10, decay_steps=5)

    losses = []
    for step in range(20):
        x = torch.randn(1, 4, 384, 384, 1, device=device)
        target = torch.randn(1, 4, 384, 384, 1, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 5 == 0:
            print(f"    Step {step:2d} | Loss: {loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"  [OK] End-to-end test passed")
    print(f"      Final loss: {losses[-1]:.6f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True

try:
    end_to_end_smoke_test()
except Exception as e:
    print(f"  [X] End-to-end test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print()
print("=" * 60)
print("SMOKE TEST SUMMARY")
print("=" * 60)

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

print("Next Steps:")
print("  1. Run full baseline experiment: python train_49f_gmr_patch.py")
print("  2. Test sparse mask: python train_physics_attention.py --variant pgsa")
print("  3. Run ablation: python train_physics_attention.py --variant all")
print()
