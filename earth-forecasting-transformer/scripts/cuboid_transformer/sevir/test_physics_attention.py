"""
Simple test script for physics_attention modules.

This script validates the basic functionality without requiring actual data.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

print("Testing Physics-Guided Attention Modules\n")

# Test 1: PGSA Layer
print("=" * 50)
print("Test 1: PhysicsGuidedSparseAttention")
print("=" * 50)

try:
    from earthformer.cuboid_transformer.physics_attention import PhysicsGuidedSparseAttention

    # Create dummy input
    B, T, H, W, C = 2, 4, 32, 32, 64
    x = torch.randn(B, T, H, W, C)
    dbz = torch.rand(B, T, H, W, 1) * 75  # dBZ values 0-75

    # Test each masking mode
    for mode in ['token_drop', 'attention_mask', 'hybrid']:
        pgsa = PhysicsGuidedSparseAttention(
            dim=C,
            num_heads=4,
            dbz_threshold=15.0,
            masking_mode=mode,
        )
        out = pgsa(x, dbz_values=dbz)
        print(f"  ✓ {mode:20s} | Output shape: {out.shape}")

    print("Test 1 PASSED\n")

except Exception as e:
    print(f"Test 1 FAILED: {e}\n")

# Test 2: DPCBA Layer
print("=" * 50)
print("Test 2: DensityProximityCrossBlockAttention")
print("=" * 50)

try:
    from earthformer.cuboid_transformer.physics_attention import DensityProximityScorer

    # Create dummy patches
    B, N, V, C = 2, 16, 8, 64
    patches = torch.randn(B, N, V, C)
    positions = torch.stack(torch.meshgrid(
        torch.arange(4),
        torch.arange(4),
        torch.arange(1),
        indexing='ij'
    ), dim=-1).reshape(-1, 3)

    scorer = DensityProximityScorer(
        density_weight=1.0,
        proximity_weight=1.0,
    )

    density_scores = scorer.compute_density_scores(patches)
    proximity_scores = scorer.compute_proximity_scores(positions)

    print(f"  ✓ Density scores shape:  {density_scores.shape}")
    print(f"  ✓ Proximity scores shape: {proximity_scores.shape}")
    print("Test 2 PASSED\n")

except Exception as e:
    print(f"Test 2 FAILED: {e}\n")

# Test 3: dBZ Computation Utilities
print("=" * 50)
print("Test 3: dBZ Computation Utilities")
print("=" * 50)

try:
    from earthformer.cuboid_transformer.physics_attention import (
        compute_dbz_from_vil,
        compute_dbz_from_features,
    )

    # VIL to dBZ
    vil = torch.rand(2, 4, 32, 32, 1) * 50  # VIL values
    dbz_from_vil = compute_dbz_from_vil(vil)
    print(f"  ✓ VIL→dBZ shape: {dbz_from_vil.shape}")

    # Features to dBZ
    features = torch.randn(2, 4, 32, 32, 64)
    for method in ['norm', 'max', 'mean']:
        dbz_from_feat = compute_dbz_from_features(features, method=method)
        print(f"  ✓ Features→dBZ ({method:4s}): {dbz_from_feat.shape}")

    print("Test 3 PASSED\n")

except Exception as e:
    print(f"Test 3 FAILED: {e}\n")

# Test 4: Flash Attention Compatibility
print("=" * 50)
print("Test 4: Flash Attention Compatibility")
print("=" * 50)

try:
    from earthformer.cuboid_transformer.physics_attention import FLASH_ATTENTION_AVAILABLE

    print(f"  Flash Attention available: {FLASH_ATTENTION_AVAILABLE}")

    if FLASH_ATTENTION_AVAILABLE:
        from earthformer.cuboid_transformer.physics_attention import (
            FlashAttentionCompatibleSparseAttention,
        )

        attn = FlashAttentionCompatibleSparseAttention(
            dim=64,
            num_heads=4,
            block_size=32,
        )

        N = 128
        x = torch.randn(2, N, 64)
        attn_mask = torch.ones(N, N, dtype=torch.bool)

        out = attn(x, attn_mask)
        print(f"  ✓ Output shape: {out.shape}")
    else:
        print("  ⚠ Flash Attention not installed (optional)")

    print("Test 4 PASSED\n")

except Exception as e:
    print(f"Test 4 FAILED: {e}\n")

# Test 5: Integration with EarthFormer
print("=" * 50)
print("Test 5: Integration Test")
print("=" * 50)

try:
    from earthformer.cuboid_transformer.physics_attention import (
        create_pgsa_cuboid_attention,
        create_dpcba_cuboid_attention,
    )

    # Test factory functions
    pgsa_layer = create_pgsa_cuboid_attention(
        dim=64,
        num_heads=4,
        cuboid_size=(2, 7, 7),
        dbz_threshold=15.0,
    )
    print(f"  ✓ PGSA layer created: {type(pgsa_layer).__name__}")

    dpcba_layer = create_dpcba_cuboid_attention(
        dim=64,
        num_heads=4,
        cuboid_size=(2, 7, 7),
        num_connections=4,
    )
    print(f"  ✓ DPCBA layer created: {type(dpcba_layer).__name__}")

    print("Test 5 PASSED\n")

except Exception as e:
    print(f"Test 5 FAILED: {e}\n")

print("=" * 50)
print("All tests completed!")
print("=" * 50)
