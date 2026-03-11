# DATSwinLSTM-Memory 实验配置详解 (Exp1-12)

## 基线模型

**DATSwinLSTM-Memory** (384×384) — 来自论文 *Motion-Guided Global-Local Aggregation Transformer Network*

| 参数 | 值 |
|------|-----|
| 输入分辨率 | 384×384 |
| patch_size | 4 → 96×96 tokens |
| embed_dim | 64 |
| depths | [2, 2] (下采样/上采样各2层) |
| heads | [4, 4] |
| window_size | 4 |
| Memory Bank | 100 slots × 256 channels |
| 训练方式 | Phase1(update) + Phase2(fixed) 交替 |
| Loss | L1 + MSE (论文公式12) |

核心组件:
- **SwinLSTMCell**: 用 DATSwin (Deformable Attention Transformer + Swin) 替代原始 SwinTransformer
- **MotionEncoder2D**: 2D+3D卷积 + MotionSqueeze 提取运动特征
- **Memory Bank**: 可学习参数矩阵, 存储长期运动模式
- **Transformer Decoder**: Self-Attention + Cross-Attention 融合长短期特征

---

## 实验矩阵

所有实验在**基线模型之上**进行"后处理注入" — 先创建完全相同的 Memory 模型, 再替换/增强指定模块:

```
基线 Memory 模型 → apply_experiment() → 注入 MoE/SwiGLU/RoPE/Flash
```

### 3×2×2 = 12 个实验组合

| | **无 Flash** | **Flash Attention** |
|---|---|---|
| **MoE (GELU)** | Exp1 | Exp7 |
| **MoE + SwiGLU** | Exp2 | Exp8 |
| **MoE + SwiGLU + Balance** | Exp3 | Exp9 |
| **MoE (GELU) + RoPE** | Exp4 | Exp10 |
| **MoE + SwiGLU + RoPE** | Exp5 | Exp11 |
| **MoE + SwiGLU + Balance + RoPE** | Exp6 | Exp12 |

---

## 各改进技术详解

### 1. MoE (Mixture of Experts) — 所有实验都启用

**代码位置**: `modules/moe_layer.py` → `MoELayer`

**改动**: 将模型中所有 `Mlp` (FFN) 替换为 `MoELayer`

```python
# 原始: 1个FFN
class Mlp:
    fc1: Linear(dim → dim*mlp_ratio)  # 64→128
    act: GELU()
    fc2: Linear(dim*mlp_ratio → dim)  # 128→64

# MoE替换: 4个Expert, Top-2路由
class MoELayer:
    router: Linear(dim → num_experts)  # 64→4, 选top-2
    experts: [Expert_0, Expert_1, Expert_2, Expert_3]
    # 每个Expert结构 = 原Mlp
```

**参数**: `num_experts=4, top_k=2, mlp_ratio=2.0`

**效果**: 模型容量×4 (4个专家), 但每次只激活2个 → 计算成本仅×2

---

### 2. SwiGLU 激活函数 — Exp2,3,5,6,8,9,11,12

**代码位置**: `modules/swiglu.py` → `SwiGLUFFN`

**改动**: 将 MoE 中每个 Expert 的激活函数从 GELU 替换为 SwiGLU

```python
# GELU Expert (Exp1,4,7,10):
x → Linear → GELU() → Linear → out

# SwiGLU Expert (Exp2,3,5,6,8,9,11,12):
x → Linear_gate → SiLU() ⊙ Linear_up → Linear_down → out
#    (门控)         (sigmoid linear) (逐元素相乘)
```

**效果**: SwiGLU 来自 Google PaLM/LLaMA, 在 Transformer 中比 GELU 表现更好, 特别是在稀疏激活(MoE)场景

---

### 3. Balance + Orthogonal Loss — Exp3,6,9,12

**代码位置**: `modules/moe_layer.py` 中的 `aux_loss` 计算

**改动**: 在训练损失中添加两个正则项:

```python
total_loss = pred_loss + balance_loss_weight * L_balance + ortho_loss_weight * L_ortho
```

**Balance Loss** (`weight=0.01`):
- 惩罚 router 把所有 token 都发给同一个 Expert
- 鼓励负载均衡, 让每个 Expert 处理大致相同数量的 token
- `L_balance = num_experts * mean(fraction_i * probability_i)`

**Orthogonal Loss** (`weight=0.001`):
- 鼓励不同 Expert 学习不同的特征
- 计算 Expert 权重矩阵之间的余弦相似度, 惩罚过于相似的 Expert
- 避免 Expert 退化为同质化

---

### 4. Temporal RoPE — Exp4,5,6,10,11,12

**代码位置**: `modules/temporal_rope.py`

**改动**: 在 WindowAttention 和 DATSwinDAttention 的 Q/K 矩阵上施加旋转位置编码

```python
# 原始 Attention:
attn = Q @ K^T

# 加 RoPE 后:
Q_rot = apply_rotary_emb(Q, freqs)  # 对Q的每个head旋转
K_rot = apply_rotary_emb(K, freqs)  # 对K的每个head旋转
attn = Q_rot @ K_rot^T              # 旋转后做attention
```

**参数**: `theta_long=10000.0` (低频, 捕获长期模式), `theta_short=2000.0` (高频, 捕获短期变化)

**效果**: 让 Attention 感知 token 的时序位置, 区分"第1帧的token"和"第8帧的token", 增强时序建模能力

---

### 5. Flash Attention (SDPA) — Exp7-12

**代码位置**: `experiment_factory.py` → `_inject_flash_attention()`

**改动**: 设置所有 `Attention` 和 `WindowAttention` 模块的 `use_flash=True`

```python
# 原始手动 Attention:
attn = softmax(Q @ K^T / sqrt(d)) @ V   # O(N²) 显存

# Flash Attention (SDPA):
out = F.scaled_dot_product_attention(Q, K, V)  # O(N) 显存, 自动选最优kernel
```

**效果**:
- **数学等价** — 输出完全相同, 不影响精度
- **显存 O(N²) → O(N)** — 可处理更长序列
- **速度提升 20-40%** — 利用 GPU 内存层级优化
- RTX 5070 支持 FlashAttention-2 内核

---

## 实验配置一览

| 实验 | MoE | SwiGLU | Balance | RoPE | Flash | 改进维度 |
|------|:---:|:------:|:-------:|:----:|:-----:|---------|
| **Exp1** | ✅ | ❌ | ❌ | ❌ | ❌ | 稀疏激活 |
| **Exp2** | ✅ | ✅ | ❌ | ❌ | ❌ | +更好激活函数 |
| **Exp3** | ✅ | ✅ | ✅ | ❌ | ❌ | +负载均衡 |
| **Exp4** | ✅ | ❌ | ❌ | ✅ | ❌ | +时序位置编码 |
| **Exp5** | ✅ | ✅ | ❌ | ✅ | ❌ | +SwiGLU+RoPE |
| **Exp6** | ✅ | ✅ | ✅ | ✅ | ❌ | 全部改进 |
| **Exp7** | ✅ | ❌ | ❌ | ❌ | ✅ | MoE+Flash |
| **Exp8** | ✅ | ✅ | ❌ | ❌ | ✅ | +SwiGLU |
| **Exp9** | ✅ | ✅ | ✅ | ❌ | ✅ | +Balance |
| **Exp10** | ✅ | ❌ | ❌ | ✅ | ✅ | +RoPE |
| **Exp11** | ✅ | ✅ | ❌ | ✅ | ✅ | +SwiGLU+RoPE |
| **Exp12** | ✅ | ✅ | ✅ | ✅ | ✅ | **全部改进+Flash** |

---

## 训练结果 (10 epochs, L1+MSE Loss)

| 实验 | Best Val Loss | 说明 |
|------|:------------:|------|
| **Exp11** (SwiGLU+RoPE+Flash) | **0.0428** | 🥇 最佳 |
| **Exp12** (全部+Flash) | **0.0432** | 🥈 |
| **Exp8** (SwiGLU+Flash) | **0.0437** | 🥉 |
| **Exp9** (Balanced+Flash) | 0.0438 | |
| **Exp7** (MoE+Flash) | 0.0439 | |
| **Exp10** (GELU+RoPE+Flash) | 0.0470 | GELU 不如 SwiGLU |

**初步结论**:
1. **SwiGLU 比 GELU 效果更好** (Exp11 vs Exp10: 0.0428 vs 0.0470)
2. **RoPE 有帮助** (Exp11 vs Exp8: 0.0428 vs 0.0437)
3. **Balance Loss 帮助不大** (Exp12 vs Exp11: 0.0432 vs 0.0428, 反而略差)
4. 需要 CSI/HSS 指标验证以上结论

---

## 评估指标

论文使用的 SEVIR VIL 阈值: **0.14, 0.70, 3.50, 6.90 kg/m²**

| 指标 | 含义 | 方向 |
|------|------|------|
| CSI | 综合命中率 | ↑ |
| HSS | 技巧评分 | ↑ |
| POD | 检测概率 | ↑ |
| FAR | 虚警率 | ↓ |
| MSE | 均方误差 | ↓ |
| MAE | 平均绝对误差 | ↓ |

运行评估:
```bash
python -u evaluate.py --all
```
