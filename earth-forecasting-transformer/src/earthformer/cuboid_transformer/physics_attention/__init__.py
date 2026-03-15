"""
Physics-Guided Attention Modules for EarthFormer

This package implements physics-driven sparse attention mechanisms for
meteorological forecasting transformers.

Modules:
- pgsa_layer: Physics-Guided Sparse Attention (15dBZ threshold masking)
- dpcba_layer: Density-Proximity Cross-Block Attention
- flash_attention_compat: Flash Attention integration

Example usage:
    >>> from physics_attention import PhysicsGuidedSparseAttention
    >>> pgsa = PhysicsGuidedSparseAttention(dim=128, num_heads=4)
    >>> output = pgsa(features, dbz_values)
"""

from .pgsa_layer import (
    PhysicsGuidedSparseAttention,
    PGSAWrapper,
    create_pgsa_cuboid_attention,
    compute_dbz_from_vil,
    compute_dbz_from_features,
)

from .dpcba_layer import (
    DensityProximityScorer,
    DensityProximityCrossBlockAttention,
    DPCBAWrapper,
    create_dpcba_cuboid_attention,
    create_flash_attention_block_mask,
)

from .flash_attention_compat import (
    FlashAttentionCompatibleSparseAttention,
    PhysicsGuidedFlashAttention,
    create_physics_guided_flash_attention,
    FLASH_ATTENTION_AVAILABLE,
)

__all__ = [
    # PGSA
    'PhysicsGuidedSparseAttention',
    'PGSAWrapper',
    'create_pgsa_cuboid_attention',
    'compute_dbz_from_vil',
    'compute_dbz_from_features',

    # DPCBA
    'DensityProximityScorer',
    'DensityProximityCrossBlockAttention',
    'DPCBAWrapper',
    'create_dpcba_cuboid_attention',
    'create_flash_attention_block_mask',

    # Flash Attention
    'FlashAttentionCompatibleSparseAttention',
    'PhysicsGuidedFlashAttention',
    'create_physics_guided_flash_attention',
    'FLASH_ATTENTION_AVAILABLE',
]

__version__ = '0.1.0'
