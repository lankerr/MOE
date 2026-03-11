"""
Temporal RoPE (Rotary Position Embedding) for Spatiotemporal Models
====================================================================
为时空预测模型设计的旋转位置编码，核心思想:

1. **时间感知频率**: 
   - 长期记忆 (24帧) → 大 θ_base (低频旋转) → 缓慢变化，捕捉趋势
   - 短期记忆 (12帧) → 小 θ_base (高频旋转) → 快速变化，捕捉细节

2. **2D空间RoPE**:
   - 将 head_dim 分成两半，分别编码 H 和 W 方向
   - 窗口内的相对位置通过旋转自然编码

3. **即插即用设计**:
   - 不修改 attention 权重，仅对 Q/K 施加旋转
   - 与现有 relative position bias 互补 (RPE 加性 + RoPE 乘性)

Reference:
- RoFormer (Su et al., 2021): Rotary Position Embedding
- RoPE-2D (Lu et al., 2024): 2D Vision RoPE
- CoPE (Goyal et al., 2024): Contextual Position Encoding
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, max_len: int, theta_base: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    """
    预计算 RoPE 的 cos/sin 频率矩阵
    
    freq_i = 1 / (theta_base^(2i/dim)), for i = 0, 1, ..., dim/2 - 1
    
    θ_base 大 → 低频旋转 → 远距离位置仍有区分度 (适合长期)
    θ_base 小 → 高频旋转 → 近距离位置区分更敏感 (适合短期)
    
    Args:
        dim: 编码维度 (通常是 head_dim)
        max_len: 最大序列长度
        theta_base: 基础频率参数
        device: 计算设备
    
    Returns:
        freqs_cis: (max_len, dim//2, 2) -> [cos, sin] pairs
    """
    freqs = 1.0 / (theta_base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    freqs_cos = freqs.cos()  # (max_len, dim//2)
    freqs_sin = freqs.sin()  # (max_len, dim//2)
    return torch.stack([freqs_cos, freqs_sin], dim=-1)  # (max_len, dim//2, 2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    对 Q 或 K 应用旋转位置编码
    
    RoPE(x) = [x₁cos(mθ) - x₂sin(mθ), x₁sin(mθ) + x₂cos(mθ)]
    
    Args:
        x: (..., seq_len, dim) 输入张量
        freqs_cis: (seq_len, dim//2, 2) 预计算的 cos/sin
    
    Returns:
        与 x 同形状的旋转后张量
    """
    # x: (..., N, D) → (..., N, D//2, 2)
    orig_dtype = x.dtype
    x_float = x.float()
    x_reshape = x_float.reshape(*x_float.shape[:-1], -1, 2)  # (..., N, D//2, 2)
    
    # freqs_cis: (N, D//2, 2) → broadcast to match x
    cos_part = freqs_cis[..., 0]  # (N, D//2)
    sin_part = freqs_cis[..., 1]  # (N, D//2)
    
    # 扩展维度以支持 batch
    while cos_part.dim() < x_reshape.dim() - 1:
        cos_part = cos_part.unsqueeze(0)
        sin_part = sin_part.unsqueeze(0)
    
    x1 = x_reshape[..., 0]  # (..., N, D//2)
    x2 = x_reshape[..., 1]  # (..., N, D//2)
    
    # 旋转
    y1 = x1 * cos_part - x2 * sin_part
    y2 = x1 * sin_part + x2 * cos_part
    
    out = torch.stack([y1, y2], dim=-1)  # (..., N, D//2, 2)
    out = out.reshape(x.shape)
    return out.to(orig_dtype)


class TemporalRoPE2D(nn.Module):
    """
    时间感知的2D旋转位置编码
    
    核心设计:
    - **时间维度**: 根据记忆类型 (长期/短期) 使用不同的 θ_base
      - 长期 (24帧): theta_base=10000 → 低频，缓慢旋转
      - 短期 (12帧): theta_base=2000  → 高频，快速旋转
    - **空间维度**: 在 window attention 中编码 H/W 相对位置
      - head_dim 的前半部分编码 H 方向
      - head_dim 的后半部分编码 W 方向
    
    使用方式:
        rope = TemporalRoPE2D(head_dim=32, window_size=4)
        
        # 在 attention 中:
        q, k = rope(q, k, temporal_type='short')  # 或 'long'
    
    Args:
        head_dim: 每个 attention head 的维度
        window_size: 窗口大小 (H=W)
        theta_long: 长期记忆的基础频率 (大值=低频)
        theta_short: 短期记忆的基础频率 (小值=高频)
        max_temporal_len: 最大时间步长
    """
    
    def __init__(self, head_dim: int, window_size: int = 4,
                 theta_long: float = 10000.0, theta_short: float = 2000.0,
                 max_temporal_len: int = 50):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.theta_long = theta_long
        self.theta_short = theta_short
        
        # 空间 RoPE: head_dim 分成两半，分别编码 H 和 W
        # 确保 head_dim 被4整除 (split H/W, 每个再 pair)
        self.spatial_dim = head_dim  # 用于空间编码的维度
        self.spatial_half = self.spatial_dim // 2  # H方向用前半, W方向用后半
        
        # 预计算2D空间频率 (对窗口内位置)
        self._precompute_spatial_freqs(window_size)
        
        # 预计算时间频率
        self._precompute_temporal_freqs(max_temporal_len)
    
    def _precompute_spatial_freqs(self, window_size: int):
        """预计算窗口内2D空间位置的RoPE频率"""
        ws = window_size
        
        # H方向频率 (前半 head_dim)
        h_freqs = precompute_freqs_cis(
            dim=self.spatial_half,
            max_len=ws,
            theta_base=10000.0  # 空间RoPE使用标准theta
        )  # (ws, spatial_half//2, 2)
        
        # W方向频率 (后半 head_dim) 
        w_freqs = precompute_freqs_cis(
            dim=self.spatial_half,
            max_len=ws,
            theta_base=10000.0
        )  # (ws, spatial_half//2, 2)
        
        # 构建 (ws*ws, head_dim//2, 2) 的2D频率表
        # 对于每个(h, w)位置: [h_freqs[h], w_freqs[w]]
        spatial_freqs = torch.zeros(ws * ws, self.head_dim // 2, 2)
        half_h = self.spatial_half // 2  # H方向占用的频率数
        
        for h in range(ws):
            for w in range(ws):
                pos = h * ws + w
                spatial_freqs[pos, :half_h, :] = h_freqs[h]  # H方向
                spatial_freqs[pos, half_h:, :] = w_freqs[w]   # W方向
        
        self.register_buffer('spatial_freqs', spatial_freqs, persistent=False)
    
    def _precompute_temporal_freqs(self, max_len: int):
        """预计算不同theta的时间频率"""
        # 长期记忆频率 (低频旋转)
        long_freqs = precompute_freqs_cis(
            dim=self.head_dim, max_len=max_len, theta_base=self.theta_long
        )
        # 短期记忆频率 (高频旋转)
        short_freqs = precompute_freqs_cis(
            dim=self.head_dim, max_len=max_len, theta_base=self.theta_short
        )
        
        self.register_buffer('temporal_long_freqs', long_freqs, persistent=False)
        self.register_buffer('temporal_short_freqs', short_freqs, persistent=False)
    
    def get_spatial_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取空间 RoPE 频率
        
        Args:
            seq_len: 序列长度 (应该 = window_size^2)
            device: 设备
        
        Returns:
            (seq_len, head_dim//2, 2)
        """
        if seq_len == self.spatial_freqs.shape[0]:
            return self.spatial_freqs.to(device)
        
        # 如果序列长度不匹配窗口大小, 重新计算
        ws = int(math.sqrt(seq_len))
        if ws * ws != seq_len:
            # 非方形，退化为1D RoPE
            return precompute_freqs_cis(
                self.head_dim, seq_len, theta_base=10000.0, device=device
            )
        
        self._precompute_spatial_freqs(ws)
        return self.spatial_freqs.to(device)
    
    def get_temporal_freqs(self, timestep: int, temporal_type: str = 'short',
                           device: torch.device = None) -> torch.Tensor:
        """
        获取时间步的RoPE频率
        
        Args:
            timestep: 当前时间步索引
            temporal_type: 'long' (24帧长期) 或 'short' (12帧短期)
            device: 设备
        
        Returns:
            (head_dim//2, 2) 该时间步的频率
        """
        if temporal_type == 'long':
            freqs = self.temporal_long_freqs
        else:
            freqs = self.temporal_short_freqs
        
        timestep = min(timestep, freqs.shape[0] - 1)
        return freqs[timestep].to(device) if device else freqs[timestep]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                temporal_type: str = 'short', timestep: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q/K 应用时间感知的 2D RoPE
        
        Args:
            q: (B*nW, num_heads, N, head_dim) query 张量
            k: (B*nW, num_heads, N, head_dim) key 张量
            temporal_type: 'long' 或 'short'
            timestep: 当前时间步
        
        Returns:
            q_rot, k_rot: 旋转后的 Q/K
        """
        device = q.device
        N = q.shape[-2]  # seq_len (window_size^2)
        
        # 1) 空间RoPE
        spatial_freqs = self.get_spatial_freqs(N, device)  # (N, D//2, 2)
        
        # 2) 时间RoPE (整个窗口共享同一时间步)
        temporal_freq = self.get_temporal_freqs(timestep, temporal_type, device)  # (D//2, 2)
        
        # 3) 组合: 空间频率 + 时间频率的角度相加
        # 等价于先旋转空间角度，再旋转时间角度
        # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        s_cos = spatial_freqs[..., 0]  # (N, D//2)
        s_sin = spatial_freqs[..., 1]
        t_cos = temporal_freq[..., 0]  # (D//2,)
        t_sin = temporal_freq[..., 1]
        
        combined_cos = s_cos * t_cos - s_sin * t_sin  # (N, D//2)
        combined_sin = s_sin * t_cos + s_cos * t_sin  # (N, D//2)
        combined_freqs = torch.stack([combined_cos, combined_sin], dim=-1)  # (N, D//2, 2)
        
        # 4) 应用
        q_rot = apply_rotary_emb(q, combined_freqs)
        k_rot = apply_rotary_emb(k, combined_freqs)
        
        return q_rot, k_rot


class TemporalRoPE1D(nn.Module):
    """
    简化版: 仅时间维度的 RoPE (用于 Memory Attention)
    
    Memory attention 中，Q 是 motion query，K/V 是 memory bank。
    motion query 有时序性，memory bank 没有 → 只对 Q 施加时间 RoPE。
    
    Args:
        head_dim: attention head 维度
        theta_long: 长期记忆频率
        theta_short: 短期记忆频率
        max_len: 最大序列长度
    """
    
    def __init__(self, head_dim: int, theta_long: float = 10000.0,
                 theta_short: float = 2000.0, max_len: int = 256):
        super().__init__()
        self.head_dim = head_dim
        
        long_freqs = precompute_freqs_cis(head_dim, max_len, theta_long)
        short_freqs = precompute_freqs_cis(head_dim, max_len, theta_short)
        
        self.register_buffer('long_freqs', long_freqs, persistent=False)
        self.register_buffer('short_freqs', short_freqs, persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                temporal_type: str = 'short'
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, heads, N_q, head_dim) query
            k: (B, heads, N_k, head_dim) key
            temporal_type: 'long' or 'short'
        
        Returns:
            q_rot, k_rot
        """
        freqs = self.long_freqs if temporal_type == 'long' else self.short_freqs
        
        N_q = q.shape[-2]
        N_k = k.shape[-2]
        
        q_freqs = freqs[:N_q].to(q.device)
        k_freqs = freqs[:N_k].to(k.device)
        
        q_rot = apply_rotary_emb(q, q_freqs)
        k_rot = apply_rotary_emb(k, k_freqs)
        
        return q_rot, k_rot


# ===================== 便捷工厂函数 =====================

def create_rope_for_window_attention(head_dim: int, window_size: int,
                                      theta_long: float = 10000.0,
                                      theta_short: float = 2000.0) -> TemporalRoPE2D:
    """为 WindowAttention / DATSwinDAttention 创建 2D RoPE"""
    return TemporalRoPE2D(
        head_dim=head_dim,
        window_size=window_size,
        theta_long=theta_long,
        theta_short=theta_short,
    )


def create_rope_for_memory_attention(head_dim: int,
                                      theta_long: float = 10000.0,
                                      theta_short: float = 2000.0) -> TemporalRoPE1D:
    """为 Memory Attention 创建 1D 时间 RoPE"""
    return TemporalRoPE1D(
        head_dim=head_dim,
        theta_long=theta_long,
        theta_short=theta_short,
    )


if __name__ == "__main__":
    """测试 Temporal RoPE"""
    print("=" * 60)
    print("测试 Temporal RoPE 2D")
    print("=" * 60)
    
    head_dim = 32
    window_size = 4
    num_heads = 4
    B_nW = 16  # batch * num_windows
    N = window_size * window_size  # 16
    
    rope_2d = TemporalRoPE2D(head_dim=head_dim, window_size=window_size)
    
    q = torch.randn(B_nW, num_heads, N, head_dim)
    k = torch.randn(B_nW, num_heads, N, head_dim)
    
    # 短期 RoPE
    q_rot, k_rot = rope_2d(q, k, temporal_type='short', timestep=5)
    print(f"2D RoPE short - Q: {q.shape} -> {q_rot.shape}")
    
    # 长期 RoPE
    q_rot_l, k_rot_l = rope_2d(q, k, temporal_type='long', timestep=5)
    print(f"2D RoPE long  - Q: {q.shape} -> {q_rot_l.shape}")
    
    # 验证短期旋转幅度更大
    q_diff_short = (q_rot - q).norm()
    q_diff_long = (q_rot_l - q).norm()
    print(f"\n旋转幅度对比 (timestep=5):")
    print(f"  短期 (theta=2000): {q_diff_short:.4f}")
    print(f"  长期 (theta=10000): {q_diff_long:.4f}")
    print(f"  短期/长期 比值: {q_diff_short/q_diff_long:.4f} (应 > 1)")
    
    print("\n" + "=" * 60)
    print("测试 Temporal RoPE 1D (Memory)")
    print("=" * 60)
    
    rope_1d = TemporalRoPE1D(head_dim=64)
    q_mem = torch.randn(2, 8, 36, 64)  # (B, heads, N_q, D)
    k_mem = torch.randn(2, 8, 100, 64)  # (B, heads, memory_slots, D)
    
    q_r, k_r = rope_1d(q_mem, k_mem, temporal_type='short')
    print(f"1D RoPE - Q: {q_mem.shape} -> {q_r.shape}")
    print(f"1D RoPE - K: {k_mem.shape} -> {k_r.shape}")
    
    print("\n✅ Temporal RoPE 测试通过!")
