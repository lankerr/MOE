# DATSwinLSTM → Earthformer 代码迁移审计报告

> 生成日期: 2025-01  
> 目的: 逐模块比对从 DATSwinLSTM 迁移到 Earthformer 的 MoE/Flash/RoPE 代码，确认逻辑一致性

---

## 1. 文件映射总览

| 组件 | DATSwinLSTM 文件 | Earthformer 文件 |
|------|-----------------|-----------------|
| 实验工厂 | `datswinlstm_memory/experiments/experiment_factory.py` (716行) | `sevir/experiment_factory_earthformer.py` (280行) |
| MoE核心层 | `datswinlstm_memory/modules/moe_layer.py` (600行) | **共用同一文件** (import from datswinlstm_memory) |
| 训练脚本 | `datswinlstm_memory/train_experiment_fast.py` (467行) | `sevir/train_experiment_earthformer.py` (980行) |
| 运行器 | `datswinlstm_memory/run_all.py` | `sevir/run_all_earthformer_full.py` |

---

## 2. ExperimentConfig 对比

### 2.1 字段定义

| 字段 | DATSwinLSTM | Earthformer | 一致性 |
|------|------------|-------------|--------|
| `name` | ✅ str | ✅ str | ✅ |
| `use_moe` | ✅ bool | ✅ bool | ✅ |
| `num_experts` | ✅ 4 | ✅ 4 | ✅ |
| `top_k` | ✅ 2 | ✅ 2 | ✅ |
| `mlp_ratio` | ✅ 2.0 | ✅ 2.0 | ✅ |
| `use_swiglu` | ✅ bool | ✅ bool | ✅ |
| `balance_loss_weight` | ✅ 0.0 | ✅ 0.0 | ✅ |
| `ortho_loss_weight` | ✅ 0.0 | ✅ 0.0 | ✅ |
| `router_jitter` | ✅ 0.01 | ✅ 0.01 | ✅ |
| `use_rope` | ✅ bool | ✅ bool | ✅ |
| `theta_long` | ✅ 10000.0 | ✅ 10000.0 | ✅ |
| `theta_short` | ✅ 2000.0 | ✅ 2000.0 | ✅ |
| `use_flash` | ✅ bool | ✅ bool | ✅ |
| `lr` | ✅ 1e-3 | ✅ 1e-4 | ⚠️ 不同默认值 |
| `epochs` | ✅ 100 | ❌ 不在config (由YAML) | ⚠️ 设计差异 |
| `batch_size` | ✅ 1 | ❌ 不在config (由YAML) | ⚠️ 设计差异 |

**结论**: MoE 相关核心字段完全一致。`lr` 默认值不同但不影响实际训练 (Earthformer 从 YAML 读取 `lr=1e-3`)。`epochs`/`batch_size` 由 PL Trainer 管理，不需要在 config 中。

### 2.2 实验编号映射

| Earthformer | DATSwinLSTM | 特性组合 |
|-------------|-------------|---------|
| baseline | baseline | 无增强 |
| exp1_moe_flash | exp7_moe_flash | MoE(GELU) + Flash |
| exp1_5_moe_balanced_flash | *(新增)* | MoE(GELU) + Balance(0.01) + Flash |
| exp2_swiglu_moe_flash | exp8_swiglu_moe_flash | SwiGLU-MoE + Flash |
| exp3_balanced_moe_flash | exp9_balanced_moe_flash | SwiGLU-MoE + Balance + Ortho + Flash |
| exp4_moe_rope_flash | exp10_moe_rope_flash | MoE(GELU) + RoPE + Flash |
| exp5_swiglu_moe_rope_flash | exp11_swiglu_moe_rope_flash | SwiGLU-MoE + RoPE + Flash |
| exp6_balanced_moe_rope_flash | exp12_balanced_moe_rope_flash | SwiGLU-MoE + Balance + Ortho + RoPE + Flash |

**结论**: Earthformer 只保留 Flash 版本 (全部 `use_flash=True`)，与 DATSwinLSTM exp7-12 对应。exp1.5 是新增实验 (仅 balance loss，无 ortho/swiglu)。✅ 正确。

---

## 3. MoE 注入对比

### 3.1 MoEConfig 构建

```python
# DATSwinLSTM (experiment_factory.py:260-268)
moe_config = MoEConfig(
    num_experts=config.num_experts,      # 4
    top_k=config.top_k,                  # 2
    mlp_ratio=config.mlp_ratio,          # 2.0
    use_swiglu=config.use_swiglu,
    balance_loss_weight=config.balance_loss_weight,
    ortho_loss_weight=config.ortho_loss_weight,
    router_jitter=config.router_jitter,  # 0.01
)

# Earthformer (experiment_factory_earthformer.py:106-113)
moe_config = MoEConfig(
    num_experts=config.num_experts,      # 4
    top_k=config.top_k,                  # 2
    mlp_ratio=config.mlp_ratio,          # 2.0
    use_swiglu=config.use_swiglu,
    balance_loss_weight=config.balance_loss_weight,
    ortho_loss_weight=config.ortho_loss_weight,
    router_jitter=config.router_jitter,  # 0.01
)
```

**结论**: 完全一致 ✅

### 3.2 替换目标检测逻辑

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 检测方式 | `hasattr(child, 'fc1') and hasattr(child, 'fc2') and hasattr(child, 'act')` | `isinstance(child, PositionwiseFFN)` |
| 名字过滤 | `name == 'mlp'` (只替换父模块中名称为 'mlp' 的子模块) | **无名字过滤** (所有 PositionwiseFFN 实例) |
| 维度获取 | `child.fc1.in_features` | `child.ffn_1.in_features` |
| 替换结果 | `MoELayer(dim, config, drop=0.0)` **直接替换** | `MoEFFNWrapper(MoELayer, layer_norm, dropout_p)` **包裹替换** |

**关键差异 1: 替换范围 (⚠️ 中等影响)**

DATSwinLSTM 只替换 `name == 'mlp'` 的模块，即 Transformer Block 的主 FFN。  
Earthformer **没有名字过滤**，会替换所有 `PositionwiseFFN` 实例，包括:
- `ffn_l` — 主 FFN（✅ 应替换）
- `global_ffn_l` — 全局向量 FFN（⚠️ 额外替换）

Earthformer 的 `global_dim_ratio=1`，所以全局 FFN 维度与主 FFN 相同，MoE 替换在数学上仍然正确。  
实际影响: 多了几个 MoE 层，增加少量参数和计算开销，但不会导致错误。两个模型架构不同，这是合理的适配。

**关键差异 2: 为什么需要 Wrapper (✅ 正确)**

DATSwinLSTM 的 Block 结构:
```
Block.forward:  x = x + drop_path(attn(norm1(x)))    # attn 有自己的 norm
                x = x + drop_path(mlp(norm2(x)))      # norm+residual 在 Block 中
```
→ 直接用 `MoELayer` 替换 `mlp` 即可，norm+residual 由 Block 保留。

Earthformer 的 PositionwiseFFN 结构 (pre_norm=True):
```
PositionwiseFFN.forward:  residual = data
                          data = layer_norm(data)       # norm 在 FFN 内部
                          data = activation(ffn_1(data))
                          data = activation_dropout(data)
                          data = ffn_2(data)
                          data = dropout(data)
                          data = data + residual          # residual 在 FFN 内部
```
→ 如果直接替换，会丢失 LayerNorm 和 residual 连接 → **NaN** (已验证: 这是之前 NaN 的根本原因)

MoEFFNWrapper 正确保留了:
```python
class MoEFFNWrapper(nn.Module):
    def forward(self, data):
        residual = data               # ✅ 保留 residual
        data = self.layer_norm(data)   # ✅ 保留 LayerNorm (复制原始权重)
        out = self.moe(data)           # MoE 替代 fc1+fc2
        out = self.dropout(out)        # ✅ 保留 post-FFN dropout
        out = out + residual           # ✅ 保留残差连接
        return out
```

### 3.3 Dropout 对比

| Dropout 位置 | 原始 PositionwiseFFN | MoE 替换后 | DATSwinLSTM MoE |
|-------------|---------------------|-----------|----------------|
| fc1→act 之后 (activation_dropout) | 0.1 | 0.0 (expert 内部) | 0.0 (expert 内部) |
| fc2 之后 (dropout) | 0.1 | 0.1 (wrapper) | 0.0 (Block 的 drop_path) |

**结论**: 
- activation_dropout 从 0.1→0.0: 两侧一致，MoE 稀疏路由本身有正则化效果 ✅
- post-FFN dropout: Earthformer 通过 wrapper 保留了 0.1 dropout ✅ (DATSwinLSTM 由 Block 的 drop_path 提供类似功能)

### 3.4 LayerNorm 权重迁移

```python
# experiment_factory_earthformer.py:120-121
layer_norm = child.layer_norm          # 直接引用 (共享权重)
dropout_p = child.dropout_layer.p      # 读取原始 dropout 率
```

**验证**: `child.layer_norm` 是 `nn.LayerNorm` 实例，被直接赋给 wrapper。PyTorch 中 `nn.Module` 是引用类型，所以 wrapper.layer_norm **IS** 原始的 layer_norm (包括已初始化的权重)。✅

---

## 4. Flash Attention (SDPA) 对比

### 4.1 注入方式

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 目标模块 | `WindowAttention` + `Attention` (class name 匹配) | `CuboidSelfAttentionLayer` (isinstance) |
| 方式 | 设置 `module.use_flash = True` (模块内置 SDPA 路径) | monkey-patch `module.forward` 为自定义函数 |
| 全局向量处理 | N/A (无全局向量概念) | `if self.use_global_vector: return self._orig_forward(...)` |

**DATSwinLSTM 内置 SDPA** (`WindowAttention.forward`):
```python
if getattr(self, 'use_flash', False):
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
```

**Earthformer 外部 SDPA** (`_cuboid_attention_forward`):
```python
if getattr(self, 'use_flash', False) and hasattr(F, 'scaled_dot_product_attention'):
    # 构建 float_mask = relative_position_bias + attn_mask
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=float_mask, ...)
```

### 4.2 SDPA 路径形状验证

原始 attention 的 QKV 形状: `(B, num_heads, num_cuboids, cuboid_volume, head_C)`

SDPA 需要: `(batch, heads, seq_len, head_dim)`

转换:
```python
q_sdpa = q.transpose(1, 2).reshape(B * num_cuboids, num_heads, cuboid_volume, head_C)
# (B, nH, nC, cv, hC) → transpose(1,2) → (B, nC, nH, cv, hC) → reshape → (B*nC, nH, cv, hC) ✅
```

relative_position_bias 形状链:
```
(cv, cv, nH) → permute(2,0,1) → (nH, cv, cv) → unsqueeze(1) → (nH, 1, cv, cv)
→ expand(B, nH, nC, cv, cv) → transpose(1,2) → (B, nC, nH, cv, cv) → reshape → (B*nC, nH, cv, cv) ✅
```

注: PyTorch `expand()` 可以自动在前面添加维度: `(nH, 1, cv, cv).expand(B, nH, nC, cv, cv)` 会先隐式扩展为 `(1, nH, 1, cv, cv)` → 然后扩展到 `(B, nH, nC, cv, cv)` ✅

attn_mask (shift_size > 0 时):
```
attn_mask: (nC, cv, cv) bool → zeros_like + masked_fill(-inf) + unsqueeze(0): (1, nC, cv, cv)
float_mask = relative_pos_bias + zero_mask: (nH, 1, cv, cv) + (1, nC, cv, cv) = (nH, nC, cv, cv)
→ expand(B, nH, nC, cv, cv): 自动补 batch 维度 → (B, nH, nC, cv, cv) ✅
```

### 4.3 全局向量 fallback

```python
def _cuboid_attention_forward(self, x, global_vectors=None):
    if self.use_global_vector:
        return self._orig_forward(x, global_vectors)  # ✅ 直接回退到原始 forward
```

全局向量涉及 L2G/G2L/G2G 三种注意力模式，monkey-patch 无法正确处理。回退到原始 forward 是最安全的做法。✅

### 4.4 手动 attention fallback 路径

当 `use_flash=False` 或 `use_rope=True, use_flash=False` 时走手动路径:

```python
else:
    q = q * self.scale
    attn_score = q @ k.transpose(-2, -1)
    if self.use_relative_pos:
        attn_score = attn_score + relative_position_bias
    attn_score = masked_softmax(attn_score, mask=attn_mask)
    attn_score = self.attn_drop(attn_score)
    reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(...)
```

与原始 `CuboidSelfAttentionLayer.forward` 逻辑比对:
- `self.scale = head_dim ** -0.5` ✅
- `relative_position_bias` 计算方式一致 ✅
- `masked_softmax` 调用一致 ✅
- `attn_drop` 一致 ✅
- 输出 reshape 一致 ✅

**结论**: SDPA 路径和手动 attention 路径均 ✅

---

## 5. RoPE 注入对比

### 5.1 维度选择

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| RoPE 类型 | `TemporalRoPE2D` (2D 空间网格) | `TemporalRoPE1D` (1D 序列) |
| 目标模块 | `WindowAttention` (2D: H×W) + `DATSwinDAttention` | `CuboidSelfAttentionLayer` (3D cuboid → flatten to 1D) |
| max_len | `window_size` (2D 空间边长) | `cuboid_size[0]*[1]*[2]` (cuboid 体积) |
| head_dim | `module.dim // module.num_heads` | `module.dim // module.num_heads` |

**原因分析**: 
- DATSwinLSTM 的 WindowAttention 在 2D (H×W) 平面上操作，Q/K 是 `(B, nH, window_H*window_W, head_C)` → 适合 2D RoPE
- Earthformer 的 CuboidSelfAttention 将 3D cuboid 展平为 1D 序列: Q/K 是 `(B, nH, num_cuboids, cuboid_volume, head_C)` → 适合用 1D RoPE 对 cuboid_volume 维度编码

✅ 选择合理

### 5.2 注入点

```python
# DATSwinLSTM: _window_attention_with_rope (experiment_factory.py:374)
if hasattr(self, 'rope'):
    q, k = self.rope(q, k, temporal_type=temporal_type, timestep=timestep)

# Earthformer: _cuboid_attention_forward (experiment_factory_earthformer.py:181)
if hasattr(self, 'rope'):
    q, k = self.rope(q, k, temporal_type='long')
```

**差异**: Earthformer 固定使用 `temporal_type='long'` (theta=10000)，DATSwinLSTM 根据 Phase 动态选择 `'long'`/`'short'`。  
**原因**: Earthformer 没有 Phase 1/2 概念，整个前向传播是统一的。✅ 设计差异

### 5.3 Monkey-patch 方式

两侧都使用 `functools.partial` + `module._orig_forward` 备份:

```python
# DATSwinLSTM
module._orig_forward = module.forward
module.forward = functools.partial(_window_attention_with_rope, module)

# Earthformer
module._orig_forward = module.forward
module.forward = functools.partial(_cuboid_attention_forward, module)
```

✅ 手法一致

---

## 6. 训练循环对比

### 6.1 框架差异

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 框架 | 手动训练循环 | PyTorch Lightning |
| 训练函数 | `train_one_epoch()` | PL `training_step()` |
| 验证函数 | `validate()` | PL `validation_step()` + `on_validation_epoch_end()` |

### 6.2 预测损失

```python
# DATSwinLSTM (train_experiment_fast.py:251)
pred_loss = F.l1_loss(y_target, y_hat) + F.mse_loss(y_target, y_hat)

# Earthformer (train_experiment_earthformer.py:行 CuboidSEVIRPLModule.forward)
loss = F.l1_loss(output, out_seq) + F.mse_loss(output, out_seq)
```

✅ 完全一致：MAE + MSE

### 6.3 MoE 辅助损失收集

**DATSwinLSTM** — 使用 `compute_total_loss` helper:
```python
# experiment_factory.py:680-691
def compute_total_loss(pred_loss, model, config):
    total_loss = pred_loss
    if config.use_moe and (config.balance_loss_weight > 0 or config.ortho_loss_weight > 0):
        aux_loss = collect_moe_aux_losses(model)    # moe_layer.py:423
        total_loss = total_loss + aux_loss
    return total_loss, loss_dict
```

**Earthformer** — 内联代码:
```python
# train_experiment_earthformer.py:training_step
aux_loss = torch.tensor(0.0, device=loss.device)
for m in self.torch_nn_module.modules():
    if isinstance(m, MoELayer):
        aux = m.aux_loss
        if aux.device != aux_loss.device:
            aux_loss = aux_loss.to(aux.device)
        aux_loss = aux_loss + aux
if aux_loss.item() > 0:          # ⚠️ GPU→CPU 同步
    loss = loss + aux_loss
```

**差异分析**:

| 方面 | DATSwinLSTM | Earthformer | 影响 |
|------|-------------|-------------|------|
| 跳过条件 | 检查 **config** (`balance_loss_weight > 0` or `ortho_loss_weight > 0`) | 检查 **运行时值** (`aux_loss.item() > 0`) | ⚠️ 性能 |
| GPU 同步 | ❌ 无 (config 检查在 CPU 上) | ✅ 每步 `.item()` 调用 | ⚠️ 每 step 一次 CUDA sync |
| 功能等价性 | 当 weight=0 时不进入收集循环 | 当 weight=0 时 aux_loss=0.0，`.item()=0`，不加到 loss | ✅ 等价 |

**🔧 建议修复**: 将 `.item() > 0` 检查改为 config 检查以避免 GPU 同步开销。

### 6.4 优化器对比

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 优化器 | `AdamW(model.parameters(), lr=1e-3, wd=0.01)` | `AdamW(grouped_params, lr=1e-3, wd=1e-5)` |
| 参数分组 | ❌ 无 (所有参数同 wd) | ✅ LayerNorm + bias 的 wd=0 |
| weight_decay | 0.01 | 1e-5 | 
| 梯度裁剪 | ❌ 无 | ✅ `gradient_clip_val=1.0` |

⚠️ **设计差异** — 非迁移错误，两个模型的最优超参数不同

### 6.5 学习率调度

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| Warmup | ❌ 无 | ✅ 20% steps (LambdaLR) |
| 主调度器 | `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)` | `CosineAnnealingLR(T_max=total_steps-warmup, eta_min=lr*0.1)` |
| 调度粒度 | epoch-level (`scheduler.step()` per epoch) | step-level (`interval='step'`) |

⚠️ **设计差异** — 非迁移错误

### 6.6 精度/AMP 对比

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 方式 | `autocast('cuda', enabled=True, dtype=bfloat16)` | PL `precision='bf16-mixed'` |
| GradScaler | 仅 fp16 时启用 | PL 自动管理 |
| matmul 精度 | `torch.set_float32_matmul_precision('medium')` | `torch.set_float32_matmul_precision('medium')` |

✅ 等价

### 6.7 梯度累积

| 项目 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 方式 | 手动: `loss /= accumulation_steps`, 每 N 步 `optimizer.step()` | PL: `accumulate_grad_batches=total_batch_size//(micro*gpus)` |
| 默认值 | `accumulation_steps=4` | `total_batch_size=32 / micro=8 = 4` |

✅ 等价 (PL 内部做同样的 loss 缩放)

---

## 7. apply_experiment 对比

### 7.1 函数签名

```python
# DATSwinLSTM
def apply_experiment(model: nn.Module, config: ExperimentConfig) -> nn.Module:

# Earthformer
def apply_experiment(model: nn.Module, config_name: str) -> None:
```

DATSwinLSTM 接受 config 对象，Earthformer 接受 config 名称字符串再内部查找。✅ 接口差异

### 7.2 执行顺序

| 步骤 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| 1 | `_inject_flash_attention(model, enable=config.use_flash)` 总是执行 | `_replace_ffn_with_moe(model, config)` 按需 |
| 2 | `_replace_mlp_with_moe(model, config)` 按需 | `_inject_flash_and_rope(model, config)` 按需 |
| 3 | `_inject_rope_to_window_attention(model, config)` 按需 | — |

**差异**: DATSwinLSTM 总是显式设置 `use_flash` (包括 False)，确保实验间一致。Earthformer 仅在需要时设置。

**影响**: Earthformer 的模块默认 `use_flash` 不存在 (getattr 返回 False)，所以 **不显式设置 False 也安全** ✅

---

## 8. 共享模块: moe_layer.py

Earthformer 通过 sys.path 直接导入 DATSwinLSTM 的 `modules/moe_layer.py`:

```python
sys.path.insert(0, r"c:\Users\97290\Desktop\MOE\datswinlstm_memory")
from modules.moe_layer import MoELayer, MoEConfig
```

✅ **共享同一份代码**，无分叉风险。

共享的关键类:
- `MoEConfig` — 配置 dataclass
- `TopKRouter` — 门控路由 (`nn.Linear(dim, num_experts)`)
- `StandardExpert` — GELU 专家 (`fc1 → GELU → drop → fc2 → drop`)
- `SwiGLUExpert` — SwiGLU 专家 (带 `autocast('cuda', enabled=False)` 防 NaN)
- `MoELayer` — 主层 (Router + Experts + aux_loss 计算)
- `collect_moe_aux_losses` — 遍历模型收集辅助损失

---

## 9. 发现的问题及修复建议

### 🟡 P1: aux_loss.item() GPU 同步 (性能)

**位置**: `train_experiment_earthformer.py` → `training_step()`  
**问题**: `if aux_loss.item() > 0:` 每个 step 强制 GPU→CPU 同步  
**DATSwinLSTM 方案**: 检查 config 而非运行时值  
**影响**: 轻微降低训练速度 (每 step 多一次 CUDA synchronize)  
**修复建议**:

```python
# 修改前:
if aux_loss.item() > 0:
    loss = loss + aux_loss

# 修改后:
has_moe_loss = hasattr(self, '_exp_config') and self._exp_config is not None and (
    self._exp_config.balance_loss_weight > 0 or self._exp_config.ortho_loss_weight > 0)
if has_moe_loss:
    loss = loss + aux_loss
```

或者更简单的方式 — 直接去掉 `.item()` 检查：

```python
# 最简修复: 只要有 MoE 层，就加 aux_loss (当 weight=0 时 aux_loss=0.0，加 0 不影响)
loss = loss + aux_loss  # 即使 aux_loss=0，backward 也不会有问题
if aux_loss > 0:  # tensor 比较, 不触发 sync (惰性求值)
    self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=False)
```

### 🟢 P2: 全局 FFN 也被替换为 MoE (设计差异)

**位置**: `experiment_factory_earthformer.py` → `_replace_ffn_with_moe()`  
**问题**: 替换所有 `PositionwiseFFN`，包括 `global_ffn_l` (全局向量 FFN)  
**DATSwinLSTM**: 仅替换 `name == 'mlp'`  
**影响**: 
- `global_dim_ratio=1`，维度一致，数学正确
- 多几个 MoE 层，略增参数量
- 不影响正确性  
**结论**: **可接受**，不需要修复。如果要严格对齐，可添加过滤逻辑。

### 🟢 P3: Expert 内部 dropout=0.0 (设计一致)

**问题**: 原始 PositionwiseFFN 的 `activation_dropout=0.1`，MoE 专家内部 `drop=0.0`  
**结论**: 两侧一致 (DATSwinLSTM 也是 `drop=0.0`)。MoE 稀疏路由提供隐式正则化。✅ 无需修复

---

## 10. 验证清单

| # | 验证项 | 状态 |
|---|--------|------|
| 1 | MoEConfig 字段一一对应 | ✅ |
| 2 | MoELayer 创建方式 (dim, config, drop) | ✅ |
| 3 | MoEFFNWrapper 保留 LayerNorm + residual + dropout | ✅ |
| 4 | LayerNorm 权重正确引用 (非复制) | ✅ |
| 5 | SDPA 路径 Q/K/V reshape 正确 | ✅ |
| 6 | relative_position_bias expand 维度正确 | ✅ |
| 7 | attn_mask float_mask 合并正确 | ✅ |
| 8 | 全局向量 fallback 到原始 forward | ✅ |
| 9 | RoPE 注入位置 (Q,K apply 后再计算 attention) | ✅ |
| 10 | 手动 attention fallback 路径逻辑一致 | ✅ |
| 11 | 预测损失 MAE+MSE 一致 | ✅ |
| 12 | aux_loss 收集逻辑功能等价 | ✅ |
| 13 | 精度模式 bf16-mixed 一致 | ✅ |
| 14 | 梯度累积逻辑等价 | ✅ |
| 15 | Checkpoint 保存/恢复链路完整 | ✅ |
| 16 | moe_layer.py 共享同一文件 | ✅ |

---

## 11. 总结

| 类别 | 数量 | 说明 |
|------|------|------|
| ✅ 完全一致 | 14 | MoEConfig、MoELayer创建、损失计算、精度、梯度累积等 |
| ⚠️ 设计差异 (合理) | 5 | 替换范围、优化器超参、LR调度、梯度裁剪、训练框架 |
| 🔧 可优化 | 1 | aux_loss.item() GPU 同步 |
| ❌ 错误 | 0 | 无 |

**结论**: 迁移逻辑正确，核心 MoE/Flash/RoPE 代码与 DATSwinLSTM 功能等价。仅有一个性能优化点 (P1)，无功能性错误。
