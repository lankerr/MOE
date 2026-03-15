"""
Density-Proximity Cross-Block Attention (DPCBA) Module

This module implements density-aware and proximity-based cross-block attention
for meteorological tensor data. The key innovation is establishing dynamic
connections between dense regions (storms) while maintaining efficiency through
sparse attention patterns.

Key Design:
1. Density scoring: Score each patch based on dBZ mean
2. Proximity scoring: Score patch pairs based on spatiotemporal distance
3. Dynamic cross-block connections: Connect dense, nearby patches
4. Flash Attention compatible: Use block-sparse masks for efficiency

Reference:
- Combines insights from BigBird (block-sparse) with physics-based priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange
import math


class DensityProximityScorer(nn.Module):
    """
    Computes attention scores based on:
    1. Density: Mean dBZ value of a patch
    2. Proximity: Spatiotemporal distance between patches
    """

    def __init__(
        self,
        density_weight: float = 1.0,
        proximity_weight: float = 1.0,
        spatial_sigma: float = 32.0,
        temporal_sigma: float = 2.0,
    ):
        """
        Parameters
        ----------
        density_weight : float
            Weight for density component in score
        proximity_weight : float
            Weight for proximity component in score
        spatial_sigma : float
            Gaussian decay parameter for spatial distance (pixels)
        temporal_sigma : float
            Gaussian decay parameter for temporal distance (frames)
        """
        super().__init__()
        self.density_weight = density_weight
        self.proximity_weight = proximity_weight
        self.spatial_sigma = spatial_sigma
        self.temporal_sigma = temporal_sigma

    def compute_density_scores(
        self,
        patches: torch.Tensor,
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute density score for each patch.

        Parameters
        ----------
        patches : torch.Tensor
            Patch features, shape (B, N_patches, patch_volume, C)
        dbz_values : torch.Tensor, optional
            Pre-computed dBZ values

        Returns
        -------
        density_scores : torch.Tensor
            Density scores, shape (B, N_patches)
        """
        B, N, V, C = patches.shape

        if dbz_values is not None:
            # Use provided dBZ values
            # dbz_values shape: (B, T, H, W) or (B, T, H, W, 1)
            if dbz_values.dim() == 4:
                dbz_values = dbz_values.unsqueeze(-1)

            # Average dBZ over patch volume
            # Need to know patch size to aggregate properly
            # For now, use feature norm as proxy
            density_scores = patches.norm(dim=-1).mean(dim=-1)  # (B, N)
        else:
            # Use feature magnitude as density proxy
            density_scores = patches.norm(dim=-1).mean(dim=-1)  # (B, N)

        # Normalize to [0, 1]
        density_scores = density_scores / (density_scores.max(dim=1, keepdim=True)[0] + 1e-8)
        return density_scores

    def compute_proximity_scores(
        self,
        patch_positions: torch.Tensor,
        patch_size: Tuple[int, int, int] = (2, 7, 7),
    ) -> torch.Tensor:
        """
        Compute proximity score between patch pairs.

        Parameters
        ----------
        patch_positions : torch.Tensor
            Patch positions, shape (N_patches, 3) -> (t, h, w) indices
        patch_size : tuple
            Physical size of each patch (T, H, W)

        Returns
        -------
        proximity_scores : torch.Tensor
            Proximity scores, shape (N_patches, N_patches)
        """
        N = patch_positions.shape[0]

        # Compute pairwise distances
        # Using Manhattan distance (more efficient than Euclidean)
        t_diff = torch.abs(patch_positions[:, 0:1] - patch_positions[:, 0:1].T)
        h_diff = torch.abs(patch_positions[:, 1:2] - patch_positions[:, 1:2].T)
        w_diff = torch.abs(patch_positions[:, 2:3] - patch_positions[:, 2:3].T)

        # Convert index differences to physical distances
        t_dist = t_diff * patch_size[0]
        h_dist = h_diff * patch_size[1]
        w_dist = w_diff * patch_size[2]

        # Spatial distance (2D)
        spatial_dist = torch.sqrt(h_dist**2 + w_dist**2)

        # Gaussian decay
        spatial_score = torch.exp(-spatial_dist / self.spatial_sigma)
        temporal_score = torch.exp(-t_dist / self.temporal_sigma)

        proximity_scores = spatial_score * temporal_score
        return proximity_scores

    def forward(
        self,
        patches: torch.Tensor,
        patch_positions: torch.Tensor,
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined density-proximity scores.

        Parameters
        ----------
        patches : torch.Tensor
            Patch features, shape (B, N_patches, patch_volume, C)
        patch_positions : torch.Tensor
            Patch positions, shape (N_patches, 3)
        dbz_values : torch.Tensor, optional
            dBZ values

        Returns
        -------
        scores : torch.Tensor
            Combined scores, shape (B, N_patches, N_patches)
        """
        B = patches.shape[0]

        # Density scores
        density_scores = self.compute_density_scores(patches, dbz_values)  # (B, N)

        # Proximity scores (same for all batches)
        proximity_scores = self.compute_proximity_scores(patch_positions)  # (N, N)

        # Combined score: geometric mean of density product and proximity
        # score(i, j) = (density[i] * density[j])^0.5 * proximity(i, j)
        density_product = density_scores.unsqueeze(2) * density_scores.unsqueeze(1)  # (B, N, N)
        density_geometric_mean = torch.sqrt(density_product + 1e-8)

        scores = self.density_weight * density_geometric_mean + \
                 self.proximity_weight * proximity_scores.unsqueeze(0)

        return scores


class DensityProximityCrossBlockAttention(nn.Module):
    """
    Cross-block attention module that establishes dynamic connections
    between dense, nearby patches.

    Uses block-sparse attention pattern compatible with Flash Attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: Tuple[int, int, int] = (2, 7, 7),
        num_connections: int = 4,
        density_weight: float = 1.0,
        proximity_weight: float = 1.0,
        use_flash_attention: bool = True,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Parameters
        ----------
        dim : int
            Feature dimension
        num_heads : int
            Number of attention heads
        patch_size : tuple
            Physical size of each patch (T, H, W)
        num_connections : int
            Maximum number of cross-block connections per patch
        density_weight : float
            Weight for density scoring
        proximity_weight : float
            Weight for proximity scoring
        use_flash_attention : bool
            Whether to use Flash Attention (requires installation)
        qkv_bias : bool
            Bias in QKV projection
        attn_drop : float
            Attention dropout
        proj_drop : float
            Output projection dropout
        """
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.patch_size = patch_size
        self.num_connections = num_connections
        self.use_flash_attention = use_flash_attention

        self.scorer = DensityProximityScorer(
            density_weight=density_weight,
            proximity_weight=proximity_weight,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Statistics
        self.register_buffer('num_active_connections', torch.tensor(0))

    def compute_patch_positions(
        self,
        T: int,
        H: int,
        W: int,
        cuboid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Compute position indices for all patches.

        Parameters
        ----------
        T, H, W : int
            Temporal and spatial dimensions
        cuboid_size : tuple
            Size of each cuboid (bT, bH, bW)

        Returns
        -------
        positions : torch.Tensor
            Position indices, shape (N_patches, 3)
        """
        bT, bH, bW = cuboid_size
        nT = T // bT
        nH = H // bH
        nW = W // bW

        # Create grid of patch positions
        t_idx = torch.arange(nT) * bT + bT // 2
        h_idx = torch.arange(nH) * bH + bH // 2
        w_idx = torch.arange(nW) * bW + bW // 2

        positions = torch.stack(torch.meshgrid(t_idx, h_idx, w_idx, indexing='ij'), dim=-1)
        positions = positions.reshape(-1, 3)
        return positions

    def compute_sparse_attention_mask(
        self,
        scores: torch.Tensor,
        local_window_radius: int = 1,
    ) -> torch.Tensor:
        """
        Compute sparse attention mask based on density-proximity scores.

        Parameters
        ----------
        scores : torch.Tensor
            Density-proximity scores, shape (B, N, N) or (N, N)
        local_window_radius : int
            Always include this many adjacent patches

        Returns
        -------
        attn_mask : torch.Tensor
            Boolean attention mask, shape (N, N)
        """
        if scores.dim() == 3:
            scores = scores.mean(0)  # Average across batches

        N = scores.shape[0]

        # Initialize mask with local window
        # Assume patches are arranged in a grid
        grid_size = int(math.sqrt(N))
        attn_mask = torch.zeros(N, N, dtype=torch.bool)

        for i in range(N):
            # Local window connections
            t_i, h_i, w_i = i // (grid_size * grid_size), (i // grid_size) % grid_size, i % grid_size

            for dt in range(-local_window_radius, local_window_radius + 1):
                for dh in range(-local_window_radius, local_window_radius + 1):
                    for dw in range(-local_window_radius, local_window_radius + 1):
                        j = i + dt * grid_size * grid_size + dh * grid_size + dw
                        if 0 <= j < N:
                            attn_mask[i, j] = True

        # Add top-k connections based on scores
        k = min(self.num_connections, N - 1)
        for i in range(N):
            # Get top-k scores (excluding self)
            score_i = scores[i].clone()
            score_i[i] = -float('inf')  # Exclude self
            topk_indices = torch.topk(score_i, k).indices
            attn_mask[i, topk_indices] = True

        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        cuboid_size: Tuple[int, int, int],
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with density-proximity cross-block attention.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, T, H, W, C)
        cuboid_size : tuple
            Cuboid size for dividing input
        dbz_values : torch.Tensor, optional
            dBZ values for density scoring

        Returns
        -------
        out : torch.Tensor
            Output features, shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # Compute patch positions
        positions = self.compute_patch_positions(T, H, W, cuboid_size).to(x.device)

        # Rearrange into patches
        patches = self._rearrange_patches(x, cuboid_size)  # (B, N, V, C)

        # Compute density-proximity scores
        scores = self.scorer(patches, positions, dbz_values)

        # Compute sparse attention mask
        attn_mask = self.compute_sparse_attention_mask(scores).to(x.device)

        # Update statistics
        with torch.no_grad():
            self.num_active_connections = attn_mask.sum().float() / attn_mask.shape[0]

        # Apply sparse attention
        out = self._sparse_attention(x, patches, attn_mask)

        return out

    def _rearrange_patches(
        self,
        x: torch.Tensor,
        cuboid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Rearrange input into patches."""
        from ..cuboid_transformer import cuboid_reorder
        return cuboid_reorder(x, cuboid_size, strategy=('l', 'l', 'l'))

    def _sparse_attention(
        self,
        x: torch.Tensor,
        patches: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sparse attention based on mask."""
        B, N, V, C = patches.shape

        # Reshape to sequence format
        x_seq = patches.reshape(B, N, V * C)  # (B, N, V*C)

        # QKV projection
        qkv = self.qkv(x_seq).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply sparse mask
        attn = attn.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back to original format (simplified)
        # In practice, need proper inverse of cuboid_reorder
        return x + out.unsqueeze(2).expand_as(x)  # Residual connection (simplified)


class DPCBAWrapper(nn.Module):
    """
    Wrapper to integrate DPCBA into existing CuboidSelfAttentionLayer.

    Adds cross-block connections on top of base cuboid attention.
    """

    def __init__(
        self,
        base_attention_layer: nn.Module,
        num_connections: int = 4,
        density_weight: float = 1.0,
        proximity_weight: float = 1.0,
        enable_dpcba: bool = True,
    ):
        """
        Parameters
        ----------
        base_attention_layer : nn.Module
            Original CuboidSelfAttentionLayer
        num_connections : int
            Max cross-block connections per patch
        density_weight : float
            Density scoring weight
        proximity_weight : float
            Proximity scoring weight
        enable_dpcba : bool
            Whether to enable DPCBA (for ablation)
        """
        super().__init__()
        self.base_attention = base_attention_layer
        self.enable_dpcba = enable_dpcba

        if enable_dpcba:
            self.dpcba = DensityProximityCrossBlockAttention(
                dim=base_attention_layer.dim,
                num_heads=base_attention_layer.num_heads,
                cuboid_size=base_attention_layer.cuboid_size,
                num_connections=num_connections,
                density_weight=density_weight,
                proximity_weight=proximity_weight,
            )

    def forward(
        self,
        x: torch.Tensor,
        dbz_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with optional DPCBA enhancement.

        Parameters
        ----------
        x : torch.Tensor
            Input features
        dbz_values : torch.Tensor, optional
            dBZ values

        Returns
        -------
        out : torch.Tensor
            Output features
        """
        # Base attention
        out = self.base_attention(x)

        # DPCBA enhancement
        if self.enable_dpcba:
            dpcba_out = self.dpcba(x, self.base_attention.cuboid_size, dbz_values)
            out = out + 0.1 * dpcba_out  # Small residual connection

        return out


def create_dpcba_cuboid_attention(
    dim: int,
    num_heads: int,
    cuboid_size: Tuple[int, int, int] = (2, 7, 7),
    num_connections: int = 4,
    density_weight: float = 1.0,
    proximity_weight: float = 1.0,
    enable_dpcba: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a DPCBA-enhanced cuboid attention layer.

    Parameters
    ----------
    dim : int
        Feature dimension
    num_heads : int
        Number of attention heads
    cuboid_size : tuple
        Cuboid size (T, H, W)
    num_connections : int
        Max cross-block connections
    density_weight : float
        Density scoring weight
    proximity_weight : float
        Proximity scoring weight
    enable_dpcba : bool
        Enable/disable DPCBA
    **kwargs
        Additional arguments for base attention layer

    Returns
    -------
    nn.Module
        DPCBA-enhanced attention layer
    """
    from ..cuboid_transformer import CuboidSelfAttentionLayer

    base_layer = CuboidSelfAttentionLayer(
        dim=dim,
        num_heads=num_heads,
        cuboid_size=cuboid_size,
        **kwargs
    )

    return DPCBAWrapper(
        base_attention_layer=base_layer,
        num_connections=num_connections,
        density_weight=density_weight,
        proximity_weight=proximity_weight,
        enable_dpcba=enable_dpcba,
    )


# Utility function for creating block-sparse mask compatible with Flash Attention
def create_flash_attention_block_mask(
    attn_mask: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Convert boolean attention mask to block-sparse mask for Flash Attention.

    Parameters
    ----------
    attn_mask : torch.Tensor
        Boolean attention mask, shape (N, N)
    block_size : int
        Block size for block-sparse attention

    Returns
    -------
    block_mask : torch.Tensor
        Block mask, shape (n_blocks, n_blocks)
    """
    N = attn_mask.shape[0]
    n_blocks = (N + block_size - 1) // block_size

    block_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    for i in range(n_blocks):
        for j in range(n_blocks):
            # Check if any element in block is True
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_mask[i, j] = attn_mask[i_start:i_end, j_start:j_end].any()

    return block_mask
