# MoE 替换 FFN 时的 LayerNorm + 残差连接问题

## 一句话总结

**NaN 的根因不是精度问题，而是 MoE 替换 Earthformer 的 `PositionwiseFFN` 时，把原本包含在 FFN 内部的 LayerNorm 和残差连接一起丢掉了。**

---

## 背景：两种模型的 FFN 架构差异

### DATSwinLSTM：LayerNorm + 残差在 Block 外面

DATSwinLSTM 的 Transformer Block 长这样（`dat_blocks.py`）：

```python
# DATSwinTransformerBlock.forward()
x = shortcut + self.drop_path(x)                    # attention 残差
x = x + self.drop_path(self.mlp(self.norm2(x)))      # FFN 残差
#   ↑ 残差               ↑ Mlp      ↑ LayerNorm
```

结构图：
```
┌─────────────────────────────────────────┐
│ DATSwinTransformerBlock                 │
│                                         │
│   x ──┬──── norm2 ──── Mlp ──── + ──── │
│       │                         ↑       │
│       └─────── (残差) ──────────┘       │
└─────────────────────────────────────────┘
```

Mlp 本身只是纯计算：
```python
# Mlp.forward()
x = self.fc1(x)    # Linear
x = self.act(x)    # GELU
x = self.fc2(x)    # Linear
return x            # 没有 norm，没有残差
```

**所以当我们把 `self.mlp` 替换成 `MoELayer` 时，残差和 LayerNorm 仍然在 Block 里，完全不受影响。** 这就是为什么 DATSwinLSTM 的 exp7-12（包括 SwiGLU 实验）能正常训练。

---

### Earthformer：LayerNorm + 残差在 FFN 内部

Earthformer 的 `PositionwiseFFN` 把 LayerNorm 和残差连接**打包在自己内部**（`cuboid_transformer.py`）：

```python
# PositionwiseFFN.forward()  (pre_norm=True)
residual = data                       # ← 保存残差
data = self.layer_norm(data)          # ← LayerNorm
out = self.activation(self.ffn_1(data))
out = self.ffn_2(out)
out = self.dropout_layer(out)
out = out + residual                  # ← 残差连接
return out
```

结构图：
```
┌─────────────────────────────────────────┐
│ PositionwiseFFN                         │
│                                         │
│   data ─┬── LayerNorm ── fc1 ── fc2 ─+ │
│         │                             ↑ │
│         └────────── (残差) ───────────┘ │
└─────────────────────────────────────────┘
```

**当我们用 `MoELayer` 直接替换整个 `PositionwiseFFN` 时，LayerNorm 和残差连接就一起丢失了。**

---

## 丢失后的后果

替换前（正常）：
```
input → LayerNorm → FFN → + input → output
```

替换后（有 bug 的旧版本）：
```
input → MoE(Router → Experts) → output   ← 没有 norm，没有残差！
```

### 为什么会产生 NaN？

1. **没有 LayerNorm**：MoE 的输入是未归一化的，随着训练推进，激活值会越来越大
2. **没有残差连接**：信息完全通过 MoE 前馈，不存在"直通通道"来稳定梯度传播
3. **SwiGLU 是放大器**：SwiGLU 做的是 `SiLU(W_gate · x) ⊙ (W₁ · x)`，这个 gate × up 的乘积会**指数级放大**已经偏大的值

### 实际观察到的 NaN 传播链

通过调试监控脚本（`debug/nan_debug_train.py`），我们捕获到了完整的崩溃过程：

```
Step 57 (fp32) / Step 101 (bf16)

第1步: decoder.cross_blocks.0.0.attn_l.0.norm 输出部分 NaN
       → 有限值范围正常 (abs_max ≈ 4.0)，但部分位置已经是 NaN

第2步: NaN 流入 MoE Router
       → router.gate 输入 abs_max 飙升到 7×10¹⁹（正常应该 < 10）

第3步: SwiGLU gate × up 乘积
       → experts.0.w_gate 输出 abs_max = 3×10¹⁹
       → experts.0.w1 输出 abs_max = 2.7×10¹⁹
       → gate * up = 3×10¹⁹ × 2.7×10¹⁹ ≈ 10³⁸ (逼近 fp32 上限 3.4×10³⁸)

第4步: drop 输出 abs_max = 3.3×10³⁸ → 部分溢出为 NaN

第5步: 所有权重被 NaN 梯度污染，模型永久崩溃
```

### 为什么 exp1（标准 MoE，无 SwiGLU）没崩？

exp1 使用 `StandardExpert`（GELU 激活），没有 gate × up 乘积。虽然也缺失了 LayerNorm + 残差：
- GELU 会自然压缩极端值（GELU(x) ≈ 0 当 x < -3）
- 没有乘法放大效应

所以 exp1 "碰巧"能跑完 10 个 epoch，但它的训练质量实际上也是受影响的（梯度流不够好，收敛比理想情况慢）。

---

## 修复方案

创建 `MoEFFNWrapper`，保留原 FFN 的 LayerNorm 和残差连接：

```python
class MoEFFNWrapper(nn.Module):
    """包装 MoELayer，保留 PositionwiseFFN 的 pre-LayerNorm + 残差"""
    
    def __init__(self, moe_layer, layer_norm, dropout_p=0.1):
        super().__init__()
        self.moe = moe_layer
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, data):
        residual = data
        data = self.layer_norm(data)     # ← 恢复 LayerNorm
        out = self.moe(data)
        out = self.dropout(out)
        out = out + residual             # ← 恢复残差连接
        return out
```

替换逻辑从原 FFN 复制 LayerNorm 权重和 dropout 率：
```python
if isinstance(child, PositionwiseFFN):
    dim = child.ffn_1.in_features
    moe_layer = MoELayer(dim=dim, config=moe_config, drop=0.0)
    layer_norm = child.layer_norm        # ← 复用已有的 LayerNorm（含预训练权重）
    dropout_p = child.dropout_layer.p    # ← 复用已有的 dropout rate
    wrapped = MoEFFNWrapper(moe_layer, layer_norm, dropout_p)
    setattr(module, name, wrapped)
```

修复后的结构：
```
input → [LayerNorm → MoE(Router → Experts) → Dropout] + input → output
         ↑ Wrapper 内部                                  ↑ 残差
```

### 修复效果

| 条件 | 修复前 | 修复后 |
|------|--------|--------|
| bf16-mixed, SwiGLU+MoE | Step 101 NaN | 300+ 步 0 NaN |
| fp32, SwiGLU+MoE | Step 57 NaN | 300+ 步 0 NaN |

---

## 为什么 DATSwinLSTM 不需要 Wrapper？

| | DATSwinLSTM | Earthformer |
|---|---|---|
| LayerNorm 位置 | Block 内，Mlp 外 | PositionwiseFFN **内** |
| 残差连接位置 | Block 内，Mlp 外 | PositionwiseFFN **内** |
| 替换 Mlp/FFN 时 | ✅ 不影响 norm 和残差 | ❌ 丢失 norm 和残差 |
| 需要 Wrapper | **不需要** | **需要** |

这是两种不同的 Transformer 设计风格：
- **DATSwinLSTM** 采用 Swin Transformer 风格：Block 管理 norm + residual，Mlp 只做纯 FFN 计算
- **Earthformer** 采用 BERT/GPT-2 风格：每个子层（Attention、FFN）自包含 norm + residual

两种都是标准做法，但替换 FFN 时需要注意设计差异。

---

## 相关文件

| 文件 | 修改内容 |
|------|----------|
| `experiment_factory_earthformer.py` | 新增 `MoEFFNWrapper`，修改 `_replace_ffn_with_moe` |
| `train_experiment_earthformer.py` | `training_step` 添加 MoE 辅助损失收集 |
| `run_all_earthformer_full.py` | 实验列表添加 exp1.5 |
| `debug/nan_debug_train.py` | NaN 调试监控脚本 |
| `debug/nan_report_*.json` | NaN 事件详细报告 |
| `datswinlstm_memory/experiments/experiment_factory.py` | DATSwinLSTM 的替换（无需修改） |

---

## 完整调试过程时间线

### Phase 1: 精度假说 (❌ 被推翻)

最初怀疑 NaN 是精度问题，因为 SwiGLU 使用 bf16-mixed 时出现 NaN：
- 尝试 `16-mixed` → NaN
- 尝试 `bf16-mixed` → NaN
- 在 `SwiGLUExpert.forward()` 中添加 `autocast(enabled=False)` + `x.float()` → 仍然 NaN

### Phase 2: NaN 监控脚本

创建 `debug/nan_debug_train.py`，对模型每个子模块安装 forward/backward hooks：
- 检测每个张量的 NaN/Inf
- 记录 abs_max、shape、dtype
- 首次检测到 NaN 时保存完整 JSON 报告

### Phase 3: 关键实验

| 测试 | 精度 | 步数上限 | 首次 NaN | 模块 |
|------|------|----------|----------|------|
| #1 | fp32 | 30 | 无 NaN | - |
| #2 | bf16-mixed | 3000 | Step 101 | `decoder.cross_blocks.0.0.attn_l.0.norm` |
| #3 | fp32 | 300 | **Step 57** | `decoder.cross_blocks.0.0.attn_l.0.norm` |

**关键发现**：fp32 比 bf16 更早出现 NaN！**所以 NaN 不是精度问题。**

### Phase 4: 根因定位

分析 JSON 报告中的 abs_max 传播链：
1. 阅读 `PositionwiseFFN` 源码 → 发现 LayerNorm + 残差在 FFN 内部 (pre_norm=True)
2. 阅读 `dat_blocks.py` → 确认 DATSwinLSTM 的 Mlp 不包含 norm/残差
3. 对比后发现：MoE 直接替换 PositionwiseFFN 删除了 LayerNorm + 残差连接

### Phase 5: 修复验证

创建 `MoEFFNWrapper` 后重新测试：
- bf16-mixed 300 步：**0 NaN**（修复前 Step 101 NaN）
- VRAM：0.86 GB（反而更低了）

---

## 第二个 Bug：MoE 辅助损失未收集

### 问题

`train_experiment_earthformer.py` 的 `training_step` 只计算了预测损失：
```python
loss = F.l1_loss(output, out_seq) + F.mse_loss(output, out_seq)
```

但 `MoELayer` 在 forward 中计算的 `balance_loss` 和 `ortho_loss` 被丢弃了。
这意味着 exp3 和 exp6 的 `balance_loss_weight=0.01` 形同虚设。

### 修复

在 `training_step` 中添加 MoE 辅助损失收集：
```python
# 收集 MoE 辅助损失 (balance_loss + ortho_loss)
from modules.moe_layer import MoELayer
aux_loss = torch.tensor(0.0, device=loss.device)
for m in self.torch_nn_module.modules():
    if isinstance(m, MoELayer):
        aux = m.aux_loss
        if aux.device != aux_loss.device:
            aux_loss = aux_loss.to(aux.device)
        aux_loss = aux_loss + aux
if aux_loss.item() > 0:
    loss = loss + aux_loss
    self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=False)
```

对于 baseline 和不使用 balance/ortho loss 的实验，aux_loss = 0，不影响。

---

## 实验列表 (修复后)

| 实验 | MoE | SwiGLU | Balance | RoPE | Flash | 说明 |
|------|-----|--------|---------|------|-------|------|
| baseline | ❌ | ❌ | ❌ | ❌ | ❌ | 原始 Earthformer |
| exp1_moe_flash | ✅ | ❌ | ❌ | ❌ | ✅ | 基础 MoE (StandardExpert) |
| **exp1_5_moe_balanced_flash** | ✅ | ❌ | ✅ 0.01 | ❌ | ✅ | MoE + 负载均衡 (无 SwiGLU) |
| exp2_swiglu_moe_flash | ✅ | ✅ | ❌ | ❌ | ✅ | SwiGLU MoE |
| exp3_balanced_moe_flash | ✅ | ✅ | ✅ 0.01 | ❌ | ✅ | SwiGLU + 负载均衡 |
| exp4_moe_rope_flash | ✅ | ❌ | ❌ | ✅ | ✅ | MoE + RoPE |
| exp5_swiglu_moe_rope_flash | ✅ | ✅ | ❌ | ✅ | ✅ | SwiGLU + RoPE |
| exp6_balanced_moe_rope_flash | ✅ | ✅ | ✅ 0.01 | ✅ | ✅ | 全配置 |

**exp1.5 的设计意图**：在 exp1（无平衡）和 exp3（SwiGLU+平衡）之间插入一个数据点，测试负载均衡损失对标准 MoE（无 SwiGLU）的影响。

所有 MoE 实验现在都使用 `MoEFFNWrapper`（LayerNorm + 残差连接），且辅助损失正确反传。
