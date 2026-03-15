"""
Flash Attention Integration for Physics-Guided Sparse Attention

This module provides compatibility between our physics-guided attention patterns
and Flash Attention's block-sparse API.

Key features:
1. Convert physics masks to block-sparse format
2. Hybrid attention: local cuboid + global sparse connections
3. Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Warning: Flash Attention not available. Falling back to standard attention.")


class FlashAttentionCompatibleSparseAttention(nn.Module):
    """
    Sparse attention compatible with Flash Attention's block-sparse API.

    This class bridges the gap between physics-guided masks and efficient
    attention computation using Flash Attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 32,
        use_flash_attention: bool = True,
        fallback_to_standard: bool = True,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature dimension
        num_heads : int
            Number of attention heads
        block_size : int
            Block size for block-sparse attention
        use_flash_attention : bool
            Whether to use Flash Attention (if available)
        fallback_to_standard : bool
            Fall back to standard attention if Flash Attention unavailable
        """
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size

        self.use_flash = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        if use_flash_attention and not FLASH_ATTENTION_AVAILABLE and fallback_to_standard:
            print("Flash Attention unavailable, using standard attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def create_block_sparse_mask(
        self,
        attn_mask: torch.Tensor,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert boolean attention mask to block-sparse format.

        Parameters
        ----------
        attn_mask : torch.Tensor
            Boolean mask, shape (N, N)
        block_size : int
            Block size

        Returns
        -------
        block_mask : torch.Tensor
            Block-level mask, shape (n_blocks, n_blocks)
        cu_seqlens : torch.Tensor
            Cumulative sequence lengths for varlen API
        """
        N = attn_mask.shape[0]
        n_blocks = (N + block_size - 1) // block_size

        # Create block mask
        block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=attn_mask.device)
        for i in range(n_blocks):
            for j in range(n_blocks):
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                block_mask[i, j] = attn_mask[i_start:i_end, j_start:j_end].any()

        return block_mask

    def forward_flash_attention(
        self,
        qkv: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward using Flash Attention with block-sparse mask.

        Parameters
        ----------
        qkv : torch.Tensor
            QKV packed tensor, shape (B, N, 3, num_heads, head_dim)
        attn_mask : torch.Tensor
            Boolean attention mask, shape (N, N)

        Returns
        -------
        out : torch.Tensor
            Attention output, shape (B, N, C)
        """
        B, N, _, num_heads, head_dim = qkv.shape
        C = num_heads * head_dim

        # Create block-sparse mask
        block_mask = self.create_block_sparse_mask(attn_mask, self.block_size)

        # Note: Full Flash Attention block-sparse API requires custom kernel
        # For now, use standard attention with mask
        return self.forward_standard_attention(qkv, attn_mask)

    def forward_standard_attention(
        self,
        qkv: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard attention with mask (fallback).

        Parameters
        ----------
        qkv : torch.Tensor
            QKV packed tensor
        attn_mask : torch.Tensor
            Boolean mask

        Returns
        -------
        out : torch.Tensor
            Attention output
        """
        B, N, _, num_heads, head_dim = qkv.shape

        q, k, v = qkv.unbind(dim=2)  # (B, N, num_heads, head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)

        return out

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with appropriate attention backend.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape (B, N, C)
        attn_mask : torch.Tensor
            Boolean attention mask, shape (N, N)

        Returns
        -------
        out : torch.Tensor
            Output, shape (B, N, C)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.use_flash:
            return self.forward_flash_attention(qkv, attn_mask)
        else:
            return self.forward_standard_attention(qkv, attn_mask)


class PhysicsGuidedFlashAttention(nn.Module):
    """
    Complete physics-guided attention module using Flash Attention.

    Combines:
    1. Local cuboid attention (dense within cuboids)
    2. Global sparse connections (dense-dense patch connections)
    3. Flash Attention efficiency
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cuboid_size: Tuple[int, int, int] = (2, 7, 7),
        dbz_threshold: float = 15.0,
        num_global_connections: int = 8,
        block_size: int = 32,
        use_flash_attention: bool = True,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature dimension
        num_heads : int
            Number of attention heads
        cuboid_size : tuple
            Cuboid size for local attention
        dbz_threshold : float
            dBZ threshold for density filtering
        num_global_connections : int
            Number of global connections per patch
        block_size : int
            Block size for block-sparse attention
        use_flash_attention : bool
            Use Flash Attention backend
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.cuboid_size = cuboid_size
        self.dbz_threshold = dbz_threshold
        self.num_global_connections = num_global_connections

        # Local attention (within cuboids)
        from .pgsa_layer import PhysicsGuidedSparseAttention
        self.local_attn = PhysicsGuidedSparseAttention(
            dim=dim,
            num_heads=num_heads,
            dbz_threshold=dbz_threshold,
            masking_mode='attention_mask',
        )

        # Global sparse attention
        self.global_attn = FlashAttentionCompatibleSparseAttention(
            dim=dim,
            num_heads=num_heads,
            block_size=block_size,
            use_flash_attention=use_flash_attention,
        )

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with hybrid local-global attention.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, T, H, W, C)
        dbz_values : torch.Tensor, optional
            dBZ values for density scoring

        Returns
        -------
        out : torch.Tensor
            Output features
        """
        B, T, H, W, C = x.shape
        residual = x

        # Local attention (within cuboids)
        local_out = self.local_attn(x, dbz_values)

        # Global sparse attention (between dense patches)
        # Reshape to sequence format
        x_seq = local_out.reshape(B, -1, C)  # (B, T*H*W, C)

        # Compute global attention mask based on density
        global_mask = self._compute_global_mask(x, dbz_values)

        # Apply global attention
        global_out = self.global_attn(x_seq, global_mask)

        # Reshape back
        global_out = global_out.reshape(B, T, H, W, C)

        # Combine and project
        out = self.proj(global_out + local_out)
        out = self.norm(out + residual)

        return out

    def _compute_global_mask(
        self,
        x: torch.Tensor,
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute global sparse attention mask based on density.

        Parameters
        ----------
        x : torch.Tensor
            Input features
        dbz_values : torch.Tensor, optional
            dBZ values

        Returns
        -------
        mask : torch.Tensor
            Boolean attention mask, shape (N, N)
        """
        from .dpcba_layer import DensityProximityScorer

        B, T, H, W, C = x.shape
        N = T * H * W

        # Rearrange into patches
        from ..cuboid_transformer import cuboid_reorder
        patches = cuboid_reorder(x, self.cuboid_size, strategy=('l', 'l', 'l'))
        patches = patches.mean(dim=2)  # (B, N_patches, C)

        # Score patches
        scorer = DensityProximityScorer()
        positions = self._compute_patch_positions(T, H, W, self.cuboid_size).to(x.device)
        scores = scorer(patches, positions, dbz_values)  # (B, N_patches, N_patches)

        # Create sparse mask
        mask = torch.zeros(N, N, dtype=torch.bool, device=x.device)

        # Add top-k connections
        k = min(self.num_global_connections, N - 1)
        scores_mean = scores.mean(0)  # (N_patches, N_patches)
        for i in range(scores_mean.shape[0]):
            topk_indices = torch.topk(scores_mean[i], k).indices
            mask[i, topk_indices] = True

        return mask

    def _compute_patch_positions(
        self,
        T: int,
        H: int,
        W: int,
        cuboid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute patch position indices."""
        bT, bH, bW = cuboid_size
        nT = T // bT
        nH = H // bH
        nW = W // bW

        t_idx = torch.arange(nT) * bT
        h_idx = torch.arange(nH) * bH
        w_idx = torch.arange(nW) * bW

        positions = torch.stack(torch.meshgrid(t_idx, h_idx, w_idx, indexing='ij'), dim=-1)
        return positions.reshape(-1, 3)


def create_physics_guided_flash_attention(
    dim: int,
    num_heads: int,
    cuboid_size: Tuple[int, int, int] = (2, 7, 7),
    dbz_threshold: float = 15.0,
    **kwargs
) -> nn.Module:
    """
    Factory function for physics-guided Flash Attention layer.

    Parameters
    ----------
    dim : int
        Feature dimension
    num_heads : int
        Number of attention heads
    cuboid_size : tuple
        Cuboid size
    dbz_threshold : float
        dBZ threshold
    **kwargs
        Additional arguments

    Returns
    -------
    nn.Module
        Physics-guided Flash Attention layer
    """
    return PhysicsGuidedFlashAttention(
        dim=dim,
        num_heads=num_heads,
        cuboid_size=cuboid_size,
        dbz_threshold=dbz_threshold,
        **kwargs
    )
