# DATSwinLSTM-Memory 实验 1-12 详细改动说明

本文档详细说明 12 个实验变体的具体代码改动和技术细节。

---

## 目录

1. [实验概览](#实验概览)
2. [基础架构说明](#基础架构说明)
3. [实验 1-6: 基础变体](#实验-1-6-基础变体)
4. [实验 7-12: Flash Attention 变体](#实验-7-12-flash-attention-变体)
5. [代码改动位置](#代码改动位置)
6. [关键模块详解](#关键模块详解)

---

## 实验概览

| 实验 | 名称 | MoE | SwiGLU | Load Balance | Orthogonal | RoPE | Flash |
|------|------|:---:|:------:|:------------:|:----------:|:----:|:-----:|
| Exp1 | MoE_GELU | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Exp2 | SwiGLU_MoE | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Exp3 | Balanced_MoE | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Exp4 | MoE_GELU_RoPE | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Exp5 | SwiGLU_MoE_RoPE | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| Exp6 | Balanced_MoE_RoPE | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Exp7 | MoE_GELU_Flash | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Exp8 | SwiGLU_MoE_Flash | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Exp9 | Balanced_MoE_Flash | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Exp10 | MoE_GELU_RoPE_Flash | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Exp11 | SwiGLU_MoE_RoPE_Flash | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| Exp12 | Balanced_MoE_RoPE_Flash | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 基础架构说明

### 原始模型: DATSwinLSTM-Memory

```
输入 [B, 8, 1, 384, 384]
    ↓
┌─────────────────────────────────────┐
│  PatchEmbed: Conv2d(1→64, k=4, s=4) │  → [B, 96, 96, 64]
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DownSample Stage 1                 │
│  ├─ DATSwinTransformerBlock ×2      │  ← 可变形注意力 + MLP
│  └─ SwinTransformerBlock ×2         │  ← 窗口注意力 + MLP
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DownSample Stage 2                 │
│  ├─ DATSwinTransformerBlock ×2      │
│  └─ SwinTransformerBlock ×2         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Memory Module                      │
│  ├─ Cross-Attention (Memory Query)  │
│  └─ Temporal Aggregation            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  UpSample Stages (对称结构)          │
└─────────────────────────────────────┘
    ↓
输出 [B, 12, 1, 384, 384]
```

### 关键组件位置

| 组件 | 文件 | 行号 | 功能 |
|------|------|------|------|
| `Mlp` | `dat_blocks.py` | 23-38 | 原始 MLP (GELU 激活) |
| `DATSwinTransformerBlock` | `dat_blocks.py` | 168-220 | 可变形注意力块 |
| `SwinTransformerBlock` | `DATSwinLSTM_D_Memory.py` | 588-662 | 窗口注意力块 |
| `WindowAttention` | `DATSwinLSTM_D_Memory.py` | 493-585 | 窗口注意力实现 |
| `Attention` (Memory) | `DATSwinLSTM_D_Memory.py` | 263-361 | 记忆交叉注意力 |
| `Memory` | `DATSwinLSTM_D_Memory.py` | 210-450 | 主模型类 |

---

## 实验 1-6: 基础变体

### Exp1: MoE (GELU, Top-2, 4专家)

**改动**: 将所有 `Mlp` 替换为 `MoELayer`

**配置**:
```python
ExperimentConfig(
    name="Exp1_MoE_GELU",
    use_moe=True,
    num_experts=4,
    top_k=2,
    use_swiglu=False,      # 使用 GELU 激活
    balance_loss_weight=0.0,
    ortho_loss_weight=0.0,
)
```

**代码路径** (`experiment_factory.py:236-285`):
```python
def _replace_mlp_with_moe(module: nn.Module, config: ExperimentConfig):
    """递归查找所有 Mlp 实例，替换为 MoE 层"""

    moe_config = MoEConfig(
        num_experts=config.num_experts,  # 4
        top_k=config.top_k,              # 2
        mlp_ratio=config.mlp_ratio,      # 2.0
        use_swiglu=config.use_swiglu,    # False
        ...
    )

    for name, child in module.named_children():
        # 检测 Mlp: 有 fc1, fc2, act 属性
        is_mlp = (hasattr(child, 'fc1') and hasattr(child, 'fc2') and
                  hasattr(child, 'act'))

        if is_mlp and name == 'mlp':
            dim = child.fc1.in_features
            moe_layer = MoELayer(dim=dim, config=moe_config)
            setattr(module, name, moe_layer)  # 替换!
```

**MoELayer 结构** (`modules/moe_layer.py`):
```
输入 x [B, N, D]
    ↓
┌──────────────────────┐
│  Router (Linear)     │  → gate_scores [B, N, num_experts]
│  Softmax             │
└──────────────────────┘
    ↓
┌──────────────────────┐
│  Top-K Selection     │  → 选 top-2 专家
│  expert_weights      │
└──────────────────────┘
    ↓
┌──────────────────────┐          ┌──────────────────────┐
│  Expert 0: MLP(GELU) │    ...   │  Expert 3: MLP(GELU) │
└──────────────────────┘          └──────────────────────┘
    ↓
┌──────────────────────┐
│  Weighted Sum        │  → output [B, N, D]
└──────────────────────┘
```

---

### Exp2: SwiGLU-MoE

**改动**: MoE 专家使用 SwiGLU 激活函数

**配置**:
```python
ExperimentConfig(
    name="Exp2_SwiGLU_MoE",
    use_moe=True,
    num_experts=4,
    top_k=2,
    use_swiglu=True,       # ← 使用 SwiGLU
    balance_loss_weight=0.0,
    ortho_loss_weight=0.0,
)
```

**SwiGLU vs GELU** (`modules/swiglu.py`):

```python
# 原始 GELU MLP
class Mlp(nn.Module):
    def forward(self, x):
        x = self.fc1(x)      # [B,N,D] → [B,N,4D]
        x = self.act(x)      # GELU(x)
        x = self.fc2(x)      # [B,N,4D] → [B,N,D]
        return x

# SwiGLU (Qwen 风格)
class SwiGLUFFN(nn.Module):
    def forward(self, x):
        x_gate = self.gate_proj(x)   # [B,N,D] → [B,N,inter_dim]
        x_up = self.up_proj(x)       # [B,N,D] → [B,N,inter_dim]
        x = F.silu(x_gate) * x_up    # 门控 + Swish 激活
        x = self.down_proj(x)        # [B,N,inter_dim] → [B,N,D]
        return x
```

**SwiGLU 优势**:
- 门控机制增强表达能力
- Swish (SiLU) 比 GELU 更平滑
- LLaMA/Qwen 等大模型采用

---

### Exp3: Balanced MoE (Load Balance + Orthogonalization)

**改动**: 添加辅助损失促进专家均衡使用

**配置**:
```python
ExperimentConfig(
    name="Exp3_Balanced_MoE",
    use_moe=True,
    num_experts=4,
    top_k=2,
    use_swiglu=True,
    balance_loss_weight=0.01,   # ← 负载均衡损失权重
    ortho_loss_weight=0.001,    # ← 正交化损失权重
)
```

**辅助损失计算** (`modules/moe_layer.py:150-180`):

```python
def collect_moe_aux_losses(model):
    """收集所有 MoE 层的辅助损失"""
    total_aux_loss = 0.0

    for module in model.modules():
        if isinstance(module, MoELayer):
            # 1) Load Balance Loss
            # 鼓励所有专家被均匀使用
            # loss = num_experts * sum(f_i * P_i)
            # f_i = 实际路由到专家 i 的比例
            # P_i = 专家 i 的平均路由概率
            if module.balance_loss_weight > 0:
                gates = module.last_gates  # [B, N, num_experts]
                expert_freq = gates.mean(dim=[0,1])  # 实际频率
                expert_prob = gates.softmax(-1).mean(dim=[0,1])  # 路由概率
                balance_loss = num_experts * (expert_freq * expert_prob).sum()
                total_aux_loss += module.balance_loss_weight * balance_loss

            # 2) Orthogonalization Loss
            # 鼓励专家学习不同的特征
            # loss = ||W_i · W_j^T|| for i != j
            if module.ortho_loss_weight > 0:
                experts = module.experts  # list of MLP modules
                ortho_loss = 0
                for i in range(len(experts)):
                    for j in range(i+1, len(experts)):
                        W_i = experts[i].fc1.weight
                        W_j = experts[j].fc1.weight
                        ortho_loss += (W_i @ W_j.T).norm()
                total_aux_loss += module.ortho_loss_weight * ortho_loss

    return total_aux_loss
```

**总损失**:
```python
total_loss = pred_loss + balance_loss_weight * balance_loss + ortho_loss_weight * ortho_loss
```

---

### Exp4-6: MoE + Temporal RoPE

**改动**: 为注意力机制添加时间感知的旋转位置编码

**配置** (以 Exp6 为例):
```python
ExperimentConfig(
    name="Exp6_Balanced_MoE_RoPE",
    use_moe=True,
    use_swiglu=True,
    balance_loss_weight=0.01,
    ortho_loss_weight=0.001,
    use_rope=True,            # ← 启用 RoPE
    theta_long=10000.0,       # 长期记忆 θ (低频)
    theta_short=2000.0,       # 短期记忆 θ (高频)
)
```

**RoPE 注入位置** (`experiment_factory.py:327-373`):

```python
def _inject_rope_to_window_attention(model, config):
    """为 WindowAttention 注入 RoPE"""

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'WindowAttention':
            # 创建 RoPE 模块
            rope = create_rope_for_window_attention(
                head_dim=head_dim,
                window_size=window_size,
                theta_long=config.theta_long,    # 10000
                theta_short=config.theta_short   # 2000
            )

            # 注册为子模块
            module.add_module('rope', rope)

            # Monkey-patch forward 方法
            module.forward = functools.partial(_window_attention_with_rope, module)
```

**RoPE 增强的 Attention** (`experiment_factory.py:376-436`):

```python
def _window_attention_with_rope(self, x, mask=None, temporal_type='short', timestep=0):
    """带 RoPE 的 WindowAttention"""

    # 1) 计算 QKV
    qkv = self.qkv(x).reshape(B_, N, 3, num_heads, head_dim).permute(2,0,3,1,4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # 2) ★ RoPE 注入点 ★
    if hasattr(self, 'rope'):
        q, k = self.rope(q, k, temporal_type=temporal_type, timestep=timestep)
        # temporal_type: 'short' (短期) 或 'long' (长期记忆)
        # timestep: 当前时间步 (0, 1, 2, ...)

    # 3) 继续正常的 attention 计算
    attn = (q @ k.transpose(-2, -1)) + relative_position_bias
    attn = softmax(attn)
    out = attn @ v

    return out
```

**Temporal RoPE 原理** (`modules/temporal_rope.py`):

```python
class TemporalRoPE2D:
    """
    时间感知的 2D 旋转位置编码

    关键思想:
    - 空间维度: 标准 2D RoPE (x, y 坐标)
    - 时间维度: 额外的旋转角度，随时间步变化

    θ_long = 10000  → 长期记忆用低频 (平滑变化)
    θ_short = 2000  → 短期记忆用高频 (快速变化)
    """

    def get_temporal_freqs(self, timestep, temporal_type, device):
        if temporal_type == 'short':
            theta = self.theta_short  # 2000 - 高频
        else:
            theta = self.theta_long   # 10000 - 低频

        # 时间步的频率编码
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
        angles = timestep * freqs
        return torch.stack([cos(angles), sin(angles)], dim=-1)
```

---

## 实验 7-12: Flash Attention 变体

### Flash Attention 改动

**配置** (以 Exp12 为例):
```python
ExperimentConfig(
    name="Exp12_Balanced_MoE_RoPE_Flash",
    use_moe=True,
    use_swiglu=True,
    balance_loss_weight=0.01,
    ortho_loss_weight=0.001,
    use_rope=True,
    use_flash=True,           # ← 启用 Flash Attention
)
```

**Flash Attention 注入** (`experiment_factory.py:288-322`):

```python
def _inject_flash_attention(model: nn.Module, enable: bool = True):
    """设置所有 Attention 模块的 use_flash 标志"""

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__

        # Memory 中的 cross-attention
        if cls_name == 'Attention' and hasattr(module, 'use_flash'):
            module.use_flash = enable and has_sdpa

        # Swin 的 window self-attention
        elif cls_name == 'WindowAttention' and hasattr(module, 'use_flash'):
            module.use_flash = enable and has_sdpa
```

**SDPA 路径** (在 `_window_attention_with_rope` 中):

```python
# === Flash Attention (SDPA) 路径 ===
if getattr(self, 'use_flash', False):
    # 构建 attention bias (relative position + mask)
    attn_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)

    if mask is not None:
        attn_bias = attn_bias + mask.unsqueeze(1).unsqueeze(0)

    # ★ 使用 PyTorch SDPA ★
    x = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        dropout_p=drop_p,
        is_causal=False
    )
    x = x.transpose(1, 2).reshape(B_, N, C)
else:
    # === 原始手动 Attention 路径 ===
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    attn = attn + relative_position_bias
    attn = softmax(attn)
    x = (attn @ v)
```

**Flash Attention 优势**:
1. **显存优化**: O(N) 而非 O(N²) 的显存
2. **计算加速**: IO 感知的内存访问模式
3. **数值等价**: 与手动实现结果一致

---

## 代码改动位置汇总

| 文件 | 改动 | 影响实验 |
|------|------|----------|
| `modules/moe_layer.py` | MoELayer 实现 | Exp1-12 |
| `modules/swiglu.py` | SwiGLU FFN | Exp2,5,8,11,12 |
| `modules/temporal_rope.py` | Temporal RoPE | Exp4-6,10-12 |
| `experiment_factory.py:236-285` | `_replace_mlp_with_moe` | Exp1-12 |
| `experiment_factory.py:288-322` | `_inject_flash_attention` | Exp7-12 |
| `experiment_factory.py:327-436` | `_inject_rope` + `*_with_rope` | Exp4-6,10-12 |
| `train_experiment_fast.py:229,247` | L1+MSE Loss | 所有实验 |

---

## 关键模块详解

### MoE Layer 参数量对比

```
原始 MLP:
  fc1: D × 4D = 4D²
  fc2: 4D × D = 4D²
  总计: 8D²

MoE (4专家, Top-2):
  Router: D × 4 = 4D
  Expert 0-3: 4 × 8D² = 32D²
  总计: 32D² + 4D ≈ 4× 原始

但由于 Top-2 稀疏激活，实际计算量仅增加约 2×
```

### Loss 函数差异

| 脚本 | Loss | 公式 |
|------|------|------|
| `train_384.py` | MSE | `(pred - target)²` |
| `train_experiment_fast.py` | L1+MSE | `\|pred - target\| + (pred - target)²` |

**L1+MSE 优势**:
- L1 对异常值更鲁棒
- MSE 提供平滑梯度
- 组合使用效果更好

---

## 运行指令

```powershell
cd c:\Users\97290\Desktop\MOE\datswinlstm_memory

# 运行评估 (CSI/HSS/POD/FAR)
python evaluate_metrics.py

# 快速测试 (50 batches)
python evaluate_metrics.py --max_batches 50
```

---

## 参考文献

1. **MoE**: Outrageously Large Neural Networks (Shazeer et al., 2017)
2. **SwiGLU**: GLU Variants Improve Transformer (Shazeer, 2020)
3. **RoPE**: RoFormer (Su et al., 2021)
4. **Flash Attention**: FlashAttention (Dao et al., 2022)
5. **DATSwinLSTM**: Original paper architecture
