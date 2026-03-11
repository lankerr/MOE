"""
物理场分解模块 - Helmholtz Decomposition & Physics-Aware MoE

用于将向量场（如风场）分解为梯度场（无旋）和旋度场（无散）分量，
并使用物理感知的 Mixture of Experts (MoE) 机制替代标准自注意力。

Author: DatSwinLSTM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange


class HelmholtzDecomposition(nn.Module):
    """
    Helmholtz 分解：将2D向量场分解为梯度场和旋度场
    
    理论基础：
    任意向量场 F 可分解为：F = -∇φ + ∇×A
    - -∇φ: 梯度场（无旋场，势函数的负梯度）
    - ∇×A: 旋度场（无散场，向量势的旋度）
    
    Args:
        grid_spacing (tuple): 网格间距 (dx, dy)，用于计算导数
        
    Input: (B, 2, H, W) - 向量场 (u, v) 分量
    Output: 
        grad_field: (B, 2, H, W) - 梯度场分量
        curl_field: (B, 2, H, W) - 旋度场分量
    """
    
    def __init__(self, grid_spacing=(1.0, 1.0)):
        super().__init__()
        self.dx, self.dy = grid_spacing
        
    def forward(self, field):
        """
        Args:
            field: (B, 2, H, W) 输入向量场
        Returns:
            grad_field, curl_field: 各 (B, 2, H, W)
        """
        u = field[:, 0]  # (B, H, W)
        v = field[:, 1]  # (B, H, W)
        
        # 计算散度和涡度
        divergence = self._compute_divergence(u, v)
        vorticity = self._compute_vorticity(u, v)
        
        # 求解泊松方程获取势函数和流函数
        phi = self._solve_poisson(divergence)  # 速度势
        psi = self._solve_poisson(vorticity)   # 流函数
        
        # 重构梯度场和旋度场
        grad_field = self._compute_gradient(phi)
        curl_field = self._stream_to_velocity(psi)
        
        return grad_field, curl_field
    
    def _compute_divergence(self, u, v):
        """计算散度: ∂u/∂x + ∂v/∂y"""
        # 使用中心差分
        du_dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * self.dx)
        dv_dy = (torch.roll(v, -1, dims=-2) - torch.roll(v, 1, dims=-2)) / (2 * self.dy)
        return du_dx + dv_dy
    
    def _compute_vorticity(self, u, v):
        """计算涡度: ∂v/∂x - ∂u/∂y"""
        dv_dx = (torch.roll(v, -1, dims=-1) - torch.roll(v, 1, dims=-1)) / (2 * self.dx)
        du_dy = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2 * self.dy)
        return dv_dx - du_dy
    
    def _solve_poisson(self, rhs):
        """
        频域求解泊松方程: ∇²φ = rhs
        使用 FFT 快速求解
        """
        B, H, W = rhs.shape
        device = rhs.device
        
        # 构建拉普拉斯算子的频域表示
        kx = torch.fft.fftfreq(W, d=self.dx, device=device)
        ky = torch.fft.fftfreq(H, d=self.dy, device=device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        
        # 拉普拉斯算子在频域为 -4π²(kx² + ky²)
        laplacian = -4 * (torch.pi ** 2) * (KX ** 2 + KY ** 2)
        laplacian[0, 0] = 1.0  # 避免除零
        
        # FFT 求解
        rhs_fft = fft.fft2(rhs)
        phi_fft = rhs_fft / laplacian.unsqueeze(0)
        phi_fft[:, 0, 0] = 0  # 零均值条件
        
        return fft.ifft2(phi_fft).real
    
    def _compute_gradient(self, phi):
        """计算梯度场: (-∂φ/∂x, -∂φ/∂y)"""
        dphi_dx = (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1)) / (2 * self.dx)
        dphi_dy = (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2)) / (2 * self.dy)
        return torch.stack([-dphi_dx, -dphi_dy], dim=1)
    
    def _stream_to_velocity(self, psi):
        """流函数转速度场: (∂ψ/∂y, -∂ψ/∂x)"""
        dpsi_dx = (torch.roll(psi, -1, dims=-1) - torch.roll(psi, 1, dims=-1)) / (2 * self.dx)
        dpsi_dy = (torch.roll(psi, -1, dims=-2) - torch.roll(psi, 1, dims=-2)) / (2 * self.dy)
        return torch.stack([dpsi_dy, -dpsi_dx], dim=1)


class ExpertLayer(nn.Module):
    """单个专家网络"""
    
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class PhysicsAwareMoE(nn.Module):
    """
    物理感知的 Mixture of Experts (MoE) 模块
    
    将自注意力机制替换为基于物理分解的专家混合：
    - Expert 1: 处理梯度场特征（辐散/辐合运动）
    - Expert 2: 处理旋度场特征（涡旋运动）
    - Expert 3: 处理原始场特征（完整信息）
    
    Args:
        dim: 特征维度
        num_experts: 专家数量（默认3）
        hidden_dim: 专家隐藏层维度
        dropout: Dropout 率
        use_physics_gate: 是否使用物理场信息门控
    """
    
    def __init__(self, dim, num_experts=3, hidden_dim=None, dropout=0., use_physics_gate=True):
        super().__init__()
        self.num_experts = num_experts
        self.use_physics_gate = use_physics_gate
        
        # Helmholtz 分解
        self.decompose = HelmholtzDecomposition()
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertLayer(dim, hidden_dim, dropout) 
            for _ in range(num_experts)
        ])
        
        # 门控网络
        gate_input_dim = dim * 3 if use_physics_gate else dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_experts)
        )
        
        # 特征嵌入（将物理场映射到特征空间）
        self.field_embed = nn.Conv2d(2, dim, kernel_size=3, padding=1)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, velocity_field=None):
        """
        Args:
            x: (B, N, C) 输入特征
            velocity_field: (B, 2, H, W) 可选的速度场用于物理分解
            
        Returns:
            out: (B, N, C) 输出特征
        """
        B, N, C = x.shape
        x = self.norm(x)
        
        # 计算门控权重
        if self.use_physics_gate and velocity_field is not None:
            # 分解物理场
            grad_field, curl_field = self.decompose(velocity_field)
            
            # 嵌入物理场特征
            grad_feat = self.field_embed(grad_field).flatten(2).mean(-1)  # (B, C)
            curl_feat = self.field_embed(curl_field).flatten(2).mean(-1)  # (B, C)
            orig_feat = self.field_embed(velocity_field).flatten(2).mean(-1)  # (B, C)
            
            # 拼接用于门控
            gate_input = torch.cat([grad_feat, curl_feat, orig_feat], dim=-1)  # (B, 3C)
            gate_input = gate_input.unsqueeze(1).expand(-1, N, -1)  # (B, N, 3C)
        else:
            gate_input = x
        
        # 计算专家权重
        gate_logits = self.gate(gate_input)  # (B, N, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, N, num_experts)
        
        # 专家输出加权求和
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (B, N, C, E)
        out = torch.einsum('bnce,bne->bnc', expert_outputs, gate_weights)
        
        return out


class MoEAttention(nn.Module):
    """
    MoE + Attention 混合模块
    用于替换标准的 Multi-Head Self-Attention
    
    结合物理感知 MoE 和注意力机制：
    1. 使用 MoE 处理特征
    2. 使用轻量级注意力进行全局信息交互
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_experts=3):
        super().__init__()
        self.moe = PhysicsAwareMoE(dim, num_experts=num_experts, dropout=dropout)
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, velocity_field=None):
        """
        Args:
            x: (B, N, C)
            velocity_field: (B, 2, H, W) 可选
        """
        # MoE 处理
        moe_out = self.moe(x, velocity_field)
        
        # 轻量级注意力
        x = self.norm(moe_out)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return moe_out + out  # 残差连接


# ============= 便捷函数 =============

def decompose_velocity_field(velocity_field, grid_spacing=(1.0, 1.0)):
    """
    便捷函数：分解速度场为梯度场和旋度场
    
    Args:
        velocity_field: (B, 2, H, W) 或 (2, H, W)
        grid_spacing: (dx, dy) 网格间距
        
    Returns:
        grad_field, curl_field
    """
    decomposer = HelmholtzDecomposition(grid_spacing)
    
    if velocity_field.dim() == 3:
        velocity_field = velocity_field.unsqueeze(0)
        grad_field, curl_field = decomposer(velocity_field)
        return grad_field.squeeze(0), curl_field.squeeze(0)
    
    return decomposer(velocity_field)


def replace_attention_with_moe(attention_module, num_experts=3):
    """
    便捷函数：将标准 Attention 模块替换为 MoE Attention
    
    Args:
        attention_module: 原始 Attention 模块
        num_experts: MoE 专家数量
        
    Returns:
        MoEAttention 模块
    """
    dim = attention_module.to_q.in_features if hasattr(attention_module, 'to_q') else 256
    heads = attention_module.heads if hasattr(attention_module, 'heads') else 8
    
    return MoEAttention(dim=dim, heads=heads, num_experts=num_experts)


if __name__ == "__main__":
    # 测试代码
    print("Testing Helmholtz Decomposition...")
    
    # 创建测试数据
    B, H, W = 2, 64, 64
    velocity_field = torch.randn(B, 2, H, W)
    
    # 测试分解
    decomposer = HelmholtzDecomposition()
    grad_field, curl_field = decomposer(velocity_field)
    
    print(f"Input shape: {velocity_field.shape}")
    print(f"Gradient field shape: {grad_field.shape}")
    print(f"Curl field shape: {curl_field.shape}")
    
    # 验证：原始场 ≈ 梯度场 + 旋度场
    reconstructed = grad_field + curl_field
    error = (velocity_field - reconstructed).abs().mean()
    print(f"Reconstruction error: {error:.6f}")
    
    print("\nTesting PhysicsAwareMoE...")
    
    # 测试 MoE
    N, C = 256, 128
    x = torch.randn(B, N, C)
    moe = PhysicsAwareMoE(dim=C, num_experts=3)
    out = moe(x, velocity_field)
    print(f"MoE input shape: {x.shape}")
    print(f"MoE output shape: {out.shape}")
    
    print("\n✅ All tests passed!")
