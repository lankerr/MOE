"""
SwiGLU 门控线性单元 (Gated Linear Unit)
========================================
Qwen / LLaMA / PaLM 风格的门控激活函数。

SwiGLU(x, W, V, W₂) = W₂ · (SiLU(xW) ⊙ xV)

优势:
- 相比 GELU, SwiGLU 训练更稳定, 收敛更快
- 门控机制提供自适应特征选择
- 在大量实验中表现优于 ReLU/GELU/GeGLU

参数量修正:
- 标准FFN: 2 × (dim × hidden_dim) = 2dh 参数
- SwiGLU:   3 × (dim × hidden_dim') = 3dh' 参数
- 为保持参数量相当: h' = 2h/3

Reference:
- Shazeer (2020): GLU Variants Improve Transformer
- Touvron et al. (2023): LLaMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数模块
    
    SwiGLU(x) = SiLU(x_gate) ⊙ x_up
    
    用于在 FFN 內部替代 GELU。
    注意: 这个模块假设输入已经被分成两半 (gate + up)。
    """
    
    def forward(self, x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
        return F.silu(x_gate) * x_up


class SwiGLUFFN(nn.Module):
    """
    完整的 SwiGLU 前馈网络
    
    可直接替换标准 Mlp(in_features, hidden_features, out_features, act_layer, drop)
    
    对比标准 Mlp:
        Mlp:     x → Linear(d→h) → GELU → Dropout → Linear(h→d) → Dropout
        SwiGLU:  x → [Linear_gate(d→h'), Linear_up(d→h')] → SiLU(gate)⊙up → Dropout → Linear(h'→d) → Dropout
    
    Args:
        in_features: 输入维度
        hidden_features: 隐藏维度 (会自动调整为 2/3 以保持参数量)
        out_features: 输出维度 (默认=in_features)
        drop: Dropout rate
        bias: 是否使用 bias
    """
    
    def __init__(self, in_features: int, hidden_features: int = None, 
                 out_features: int = None, drop: float = 0.0, bias: bool = False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        # 参数量修正: 3个矩阵 vs 2个, 所以 hidden 缩小到 2/3
        adjusted_hidden = int(hidden_features * 2 / 3)
        # GPU 对齐: 确保能被 8 整除
        adjusted_hidden = max(8, (adjusted_hidden + 7) // 8 * 8)
        
        self.w_gate = nn.Linear(in_features, adjusted_hidden, bias=bias)
        self.w_up = nn.Linear(in_features, adjusted_hidden, bias=bias)
        self.w_down = nn.Linear(adjusted_hidden, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.act = SwiGLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        x = self.act(gate, up)
        x = self.drop(x)
        x = self.w_down(x)
        x = self.drop(x)
        return x


class GeGLUFFN(nn.Module):
    """
    GeGLU 变体: 使用 GELU 代替 SiLU 作为门控激活
    
    GeGLU(x) = GELU(x_gate) ⊙ x_up
    
    某些场景下 GeGLU 比 SwiGLU 更稳定。
    """
    
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0, bias: bool = False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        adjusted_hidden = int(hidden_features * 2 / 3)
        adjusted_hidden = max(8, (adjusted_hidden + 7) // 8 * 8)
        
        self.w_gate = nn.Linear(in_features, adjusted_hidden, bias=bias)
        self.w_up = nn.Linear(in_features, adjusted_hidden, bias=bias)
        self.w_down = nn.Linear(adjusted_hidden, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = self.drop(x)
        x = self.w_down(x)
        x = self.drop(x)
        return x


if __name__ == "__main__":
    """测试 SwiGLU"""
    print("=" * 60)
    print("测试 SwiGLU FFN")
    print("=" * 60)
    
    dim = 128
    hidden = 256
    B, N = 2, 64
    x = torch.randn(B, N, dim)
    
    # SwiGLU FFN
    swiglu = SwiGLUFFN(in_features=dim, hidden_features=hidden)
    out = swiglu(x)
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    
    # 标准 Mlp (对比)
    class StdMlp(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.fc1 = nn.Linear(d, h)
            self.fc2 = nn.Linear(h, d)
        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))
    
    std_mlp = StdMlp(dim, hidden)
    std_params = sum(p.numel() for p in std_mlp.parameters())
    
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"SwiGLU 参数量: {swiglu_params:,}")
    print(f"Std Mlp 参数量: {std_params:,}")
    print(f"参数比: {swiglu_params/std_params:.2f}x")
    
    # GeGLU
    geglu = GeGLUFFN(in_features=dim, hidden_features=hidden)
    geglu_out = geglu(x)
    geglu_params = sum(p.numel() for p in geglu.parameters())
    print(f"\nGeGLU 参数量: {geglu_params:,}")
    print(f"GeGLU 输出: {geglu_out.shape}")
    
    print("\n✅ SwiGLU 测试通过!")
