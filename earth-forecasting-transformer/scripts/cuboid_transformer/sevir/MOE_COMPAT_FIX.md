# Earthformer × MoE 跨架构兼容性修复记录

> 时间: 2026年3月11日  
> 涉及文件: `datswinlstm_memory/modules/moe_layer.py`, `experiment_factory_earthformer.py`

---

## 一句话总结

**MoE 层是为 DATSwinLSTM 的 3D 输入写的，直接移植到 Earthformer 时遇上了 5D 张量，"水土不服" 导致崩溃。**

---

## 故事的起因

我们的实验架构是这样的：

```
DATSwinLSTM (exp7-12)   ←── 共享 ──→   modules/moe_layer.py
                                              ↑
Earthformer (exp1-6)    ←── 也调用 ──→   同一个 MoE 层
```

两个模型都通过 `experiment_factory` 把 FFN 替换成 MoE 专家层。代码上看起来很优雅，一套 MoE 代码服务两个模型。但问题藏在一个微妙的维度差异里……

---

## Bug 1: "维度世界观" 的冲突

### DATSwinLSTM 的世界

DATSwinLSTM 内部把时空数据拍平后送入 FFN：

```
FFN 输入: (B, N, C)     ← 3D 张量，N 是拍平后的 token 数
```

MoE 层的代码就是照着这个写的：

```python
if x.dim() == 3:
    B, N, C = x.shape
    x_flat = x.reshape(-1, C)   # ✅ 正常工作
else:
    x_flat = x                   # ← 这里假设已经是 2D，直接用
```

### Earthformer 的世界

Earthformer 的 FFN 收到的是原汁原味的 5D 时空张量：

```
FFN 输入: (B, T, H, W, C)   ← 5D 张量！时间×高×宽×通道
```

**当 5D 张量走到 `else` 分支，`x_flat` 还是 5D**。后面的 `index_select(0, idx_chunk)` 期望操作一个 2D `(N, C)` 矩阵，却拿到了一个 5D 怪物——直接报错：

```
RuntimeError: Index is supposed to be an empty tensor or a vector
```

### 修复

```python
# 修复前: 只处理 3D 和 2D
if x.dim() == 3:
    ...
else:
    x_flat = x  # 5D 时炸了

# 修复后: 任何 ≥3D 的输入都统一 flatten
C = x.shape[-1]
if x.dim() >= 3:
    x_flat = x.reshape(-1, C)  # (B*T*H*W, C) ← 万物皆可拍平
else:
    x_flat = x
```

最后的 `output.reshape(orig_shape)` 本来就会恢复原始形状，所以只需要修这一处。

---

## Bug 2: AMP 下的 "精度分裂"

Bug 1 修完后，还有一个隐藏 boss：

```
RuntimeError: index_add_(): self (Half) and source (Float) must have the same scalar type
```

**原因**: PyTorch AMP (`precision="16-mixed"`) 下的一场"类型内战"：

| 谁 | 精度 | 为什么 |
|---|---|---|
| `output = torch.zeros_like(x_flat)` | **fp16** | 因为 `x_flat` 是 fp16 (AMP autocast) |
| `expert(expert_input)` | **fp32** | 专家的 Linear 层在 AMP 下可能输出 fp32 |
| `index_add_()` | 💥 | 要求两边类型一致 |

这个 bug 在 DATSwinLSTM 里没暴露，是因为 DATSwinLSTM 用的是 `bf16` 而非 `16-mixed`，autocast 行为略有不同。

### 修复

```python
# 修复前
output.index_add_(0, idx_chunk, expert_output * selected_weights)

# 修复后: 强制统一精度
output.index_add_(0, idx_chunk, (expert_output * selected_weights).to(output.dtype))
```

---

## 一个差点被冤枉的无辜者: Flash Attention

在排查过程中，我们一度怀疑 Flash Attention 的 monkey-patch 在处理 `global_vectors` 时有 bug（因为原始 `forward` 在 `use_global_vector=True` 时返回元组 `(x, new_global_vector)`，而 monkey-patch 只返回单个 `x`）。

但仔细阅读代码后发现，**这个问题早就被正确处理了**：

```python
def _cuboid_attention_forward(self, x, global_vectors=None):
    # 遇到 global vectors → 直接回退到原始 forward，不碰不改
    if self.use_global_vector:
        return self._orig_forward(x, global_vectors)  # ← 安全回退
    
    # 只有 use_global_vector=False 的层才走 Flash 加速路径
    ...
```

Earthformer 配置中 `num_global_vectors=8`，编码器的 6 个注意力层全部是 `use_global_vector=True`（会回退），只有解码器的 3 个层是 `False`（走 Flash）。设计是正确的，Flash 没有罪。

---

## 验证结果

修复后对全部 6 个实验做前向传播测试：

```
Testing exp1_moe_flash...          OK → torch.Size([1, 12, 384, 384, 1])
Testing exp2_swiglu_moe_flash...   OK → torch.Size([1, 12, 384, 384, 1])
Testing exp3_balanced_moe_flash... OK → torch.Size([1, 12, 384, 384, 1])
Testing exp4_moe_rope_flash...     OK → torch.Size([1, 12, 384, 384, 1])
Testing exp5_swiglu_moe_rope_flash... OK → torch.Size([1, 12, 384, 384, 1])
Testing exp6_balanced_moe_rope_flash... OK → torch.Size([1, 12, 384, 384, 1])

All 6 experiments passed! ✓
```

---

## 深入解析：为什么一个是 3D，另一个是 5D？

这是整个 bug 的核心问题。两个模型处理同样的 SEVIR 气象雷达视频帧，但"世界观"完全不同。

### 原始输入

两个模型的原始输入是一样的——一段气象雷达视频序列：

```
输入: (B, T, C, H, W)
      B=batch, T=帧数, C=1(灰度), H=384, W=384
```

差异从"怎么看待这些帧"开始分叉。

---

### DATSwinLSTM 的处理方式：逐帧独立 → 拍扁空间

> 论文: *Motion-Guided Global-Local Aggregation Transformer Network for Precipitation Nowcasting*

DATSwinLSTM 的核心是 **Swin Transformer + LSTM**。它把时间维度交给 LSTM 处理，每个时间步内部用 Swin Transformer 处理**单帧**的空间信息。

```
单帧输入: (B, C=1, H=384, W=384)
```

**Step 1: Patch Embedding — 把图像切成小块**

```python
# dat_blocks.py → PatchEmbed
x = self.proj(x)                    # Conv2d: (B, 1, 384, 384) → (B, 128, 96, 96)
x = x.flatten(2).transpose(1, 2)    # → (B, 9216, 128) = (B, N, C)
```

一张 384×384 的图被切成 4×4 的 patch，每个 patch 变成一个 token：

$$N = \frac{384}{4} \times \frac{384}{4} = 96 \times 96 = 9{,}216 \text{ 个 token}$$

**这就是 N 的含义——不是"把一张图直接 flatten"，而是把空间切成 patch 后得到的 token 序列长度。**

**Step 2: Swin Transformer Block — 窗口内注意力**

```python
# dat_blocks.py → DATSwinTransformerBlock.forward()
B, L, C = x.shape                   # (B, 9216, 128) ← 3D token 序列

# 恢复空间结构用于分窗
x = x.view(B, H, W, C)              # (B, 96, 96, 128)

# 切窗口: 每个窗口 4×4=16 个 token
x_windows = window_partition(x, window_size=4)
# → (nW*B, 4, 4, 128) → reshape → (nW*B, 16, 128)
#   nW = 24×24 = 576 个窗口

# 窗口内做注意力
attn_output = self.attn(x_windows)   # (nW*B, 16, 128)

# 合并窗口，恢复空间
x = window_reverse(attn_output, ...)  # → (B, 96, 96, 128)

# ★ 关键: 拍扁回 3D 再送入 FFN
x = x.view(B, H * W, C)              # → (B, 9216, 128) ← FFN 看到的是 3D
x = x + self.mlp(self.norm2(x))      # FFN 输入: (B, 9216, 128)
```

**全程数据流:**

```
(B,1,384,384) → PatchEmbed → (B, 9216, 128)
                                  ↓
             ┌────────────────────────────────────┐
             │  view → (B, 96, 96, 128)           │
             │  window_partition → (nW*B, 16, 128) │
             │  Attention                          │
             │  window_reverse → (B, 96, 96, 128)  │
             │  view → (B, 9216, 128) ← 拍扁！     │ 
             │  FFN((B, 9216, 128)) ← 3D           │
             └────────────────────────────────────┘
                                  ↓
              LSTM 在时间步之间传递隐藏状态
```

---

### Earthformer 的处理方式：时空一体 → 保持 5D

> 论文: *Earthformer: Exploring Space-Time Transformers for Earth System Forecasting*

Earthformer 的核心是 **Cuboid Attention**——它把**时间和空间放在一起**处理，而不是像 DATSwinLSTM 那样把时间交给 LSTM。

```
完整序列输入: (B, T=8, H=384, W=384, C=1)
```

**Step 1: Initial Embedding**

```
(B, T, H, W, C_in) → PatchMerging3D → (B, T, H, W, C=128)
```

注意：Earthformer 做完 embedding 后，**T、H、W 三个空间维度全部保留**。

**Step 2: Cuboid Attention — 时空立方体分块**

Cuboid Attention 不像 Swin 那样只切 2D 窗口，它切的是**三维立方体**（cuboid），同时覆盖时间和空间：

```python
# cuboid_transformer.py → CuboidSelfAttentionLayer.forward()
# 输入: x = (B, T, H, W, C)

# 把 5D 时空体切成若干 cuboid
reordered_x = cuboid_reorder(x, cuboid_size=(2, 7, 7))
# → (B, num_cuboids, cuboid_volume, C) ← 4D，仅在注意力内部

# 计算 cuboid 内部的注意力
attn_output = multi_head_attention(reordered_x)

# ★ 关键: 恢复 5D 形状
x = cuboid_reorder_reverse(attn_output, ...)
# → (B, T, H, W, C) ← 恢复成 5D！
```

**Step 3: FFN — 直接收 5D**

```python
# cuboid_transformer.py → StackCuboidSelfAttentionBlock.forward()
for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
    x = x + attn(x)       # 注意力输出: (B, T, H, W, C)
    x = ffn(x)             # ★ FFN 直接收 5D 张量！
```

Earthformer 的 `PositionwiseFFN` 内部用的是 `nn.Linear` 和 `nn.LayerNorm`——它们只操作**最后一个维度**，自动广播到任意前缀维度：

```python
# PositionwiseFFN.forward()
def forward(self, data):        # data: (B, T, H, W, C)
    out = self.ffn_1(data)      # Linear(C→hidden): 自动广播
    out = self.activation(out)  # → (B, T, H, W, hidden)
    out = self.ffn_2(out)       # Linear(hidden→C): → (B, T, H, W, C)
    return out + residual
```

**`nn.Linear` 的广播特性：只要最后一维匹配 `in_features`，前面有多少维都行。** 这就是为什么原版 Earthformer 不需要 flatten——它的 FFN 天然支持 5D。

**全程数据流:**

```
(B, T, H, W, 1) → Embed → (B, T, H, W, 128)
                               ↓
          ┌──────────────────────────────────────┐
          │  cuboid_reorder → (B, nCub, Vol, C)   │  仅在注意力内部
          │  Cuboid Attention                     │  变成 4D
          │  cuboid_reorder_reverse               │
          │  → (B, T, H, W, C) ← 恢复 5D！       │
          │                                       │
          │  FFN((B, T, H, W, C)) ← 直接 5D！    │
          └──────────────────────────────────────┘
                               ↓
                    时间和空间始终"在一起"
```

---

### 对比总结

| 特征 | DATSwinLSTM | Earthformer |
|------|-------------|-------------|
| **时间处理** | LSTM 逐帧递推 | Cuboid Attention 时空一体 |
| **单步输入** | 单帧 (B, C, H, W) | 全序列 (B, T, H, W, C) |
| **Patch 化** | Conv2d → flatten → (B, N, C) | PatchMerging3D → 保持 5D |
| **注意力分块** | 2D 窗口 (window_size²) | 3D 立方体 (T×H×W cuboid) |
| **注意力后** | `view(B, H*W, C)` 拍扁 | `cuboid_reorder_reverse` 恢复 5D |
| **FFN 输入** | **(B, N, C)** — 3D | **(B, T, H, W, C)** — 5D |

### 为什么 flatten 是正确的？

我们的 MoE 修复中把 5D `(B, T, H, W, C)` flatten 成 2D `(B*T*H*W, C)` 喂给专家，**这在数学上完全等价于原版 FFN 的行为**：

- 原版 FFN 的 `nn.Linear` 对每个空间位置独立做线性变换：`y[b,t,h,w,:] = W @ x[b,t,h,w,:] + bias`
- MoE 的 flatten 把所有空间位置排成一列，对每个 token 独立路由到不同专家
- 两者都是 **position-wise**（逐位置独立）操作，token 之间互不影响
- flatten 后 `output.reshape(orig_shape)` 完美恢复原始 5D 结构

**区别仅在于**：原版 FFN 所有位置走同一个 Linear，MoE 让路由器决定每个位置走哪个专家。位置的独立性保证了 flatten 不会引入任何信息泄漏或错乱。

---

## 教训

> 当你拿着一个为 3D (B, N, C) 世界写的模块，丢进一个 5D (B, T, H, W, C) 的世界，  
> 记得先问自己：**这个模块见过这么多维度吗？**

跨模型复用模块时，**输入形状假设**是最容易被忽视、也最容易出问题的地方。

两篇论文原文参考：
- DATSwinLSTM: `datswinlstm_memory/Motion-Guided_GlobalLocal_Aggregation_Transformer_Network_for_Precipitation_Nowcasting.pdf`
- Earthformer: `earth-forecasting-transformer/earthformer_paper.pdf`

---
---

# Bug 3: SwiGLU fp16 溢出导致训练 NaN

> 时间: 2026年3月11日  
> 涉及文件: `datswinlstm_memory/modules/moe_layer.py` → `SwiGLUExpert.forward()`

---

## 现象

训练进行到若干 epoch 后，loss 突然变成 NaN，模型输出全部是 NaN，之后永不恢复。

| 实验 | 首次 NaN 位置 | 训练日志中 NaN 行数 |
|------|-------------|-------------------|
| exp2_swiglu_moe_flash (20f) | epoch 1, step 391 | 1790 |
| exp3_balanced_moe_flash (20f) | epoch 2, step 529 | 609 |
| 49f_exp2_swiglu_moe_flash | epoch 0 | 11 (刚开始就出) |

**共同点**: 全部使用 `SwiGLU` 激活的实验。不使用 SwiGLU 的 baseline 和 exp1 训练完全正常。

---

## 根因: fp16 的 65504 天花板

SwiGLU 的计算公式：

$$\text{SwiGLU}(x) = W_2 \cdot \big(\text{SiLU}(W_{gate} \cdot x) \odot W_1 \cdot x\big)$$

关键在中间的**逐元素乘法** $\text{SiLU}(W_{gate} \cdot x) \odot W_1 \cdot x$：

- `gate = SiLU(W_gate @ x)` → 值范围大致 [-0.3, +∞)
- `up = W1 @ x` → 值范围随训练变化
- `gate * up` → **两个向量逐元素相乘**

fp16 的最大可表示值是 **65504**。当模型训练了几个 epoch 后，权重增长导致 `gate` 和 `up` 的值变大，它们的乘积超过 65504 → **溢出为 Inf** → 传播变成 NaN → 整个模型永久性污染。

手动复现：

```python
# 模拟训练若干 epoch 后权重增长
expert = SwiGLUExpert(dim=128, hidden_dim=256).cuda()
with torch.no_grad():
    expert.w_gate.weight.mul_(15)  # 模拟训练后权重增大
    expert.w1.weight.mul_(15)

x = (torch.randn(1000, 128, device='cuda') * 10).half()

# fp16 方式: gate*up 溢出!
with torch.amp.autocast('cuda', dtype=torch.float16):
    gate = F.silu(x @ expert.w_gate.weight.half().T)
    up = x @ expert.w1.weight.half().T
    prod = gate * up  # ← 4 个 Inf, 512 个传播到输出
```

---

## 为什么 Standard FFN 不会溢出？

标准 FFN: `y = W2(GELU(W1(x)))`

- 只有一次矩阵乘法后接激活，没有两个中间结果相乘
- GELU 的输出范围有界，不会像 SiLU 那样放大后再乘另一个大值

SwiGLU 的 `gate * up` 是**两个无界值的乘积**，这是 fp16 溢出的根本原因。

---

## 修复: 关键陷阱 `autocast` 劫持

### ❌ 第一次尝试 (失败)

```python
def forward(self, x):
    x = x.float()  # 转 fp32
    gate = F.silu(self.w_gate(x))
    up = self.w1(x)
    x = gate * up   # 看起来应该是 fp32 了？
    ...
```

**但这完全没用！** 验证：

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    x_f32 = x_fp16.float()
    out = expert.w_gate(x_f32)
    print(out.dtype)  # → torch.float16 !!!
```

**`torch.amp.autocast` 会劫持 `nn.Linear` 的输入**，强制把 fp32 重新 cast 回 fp16。你以为是 fp32 在算，实际上 `w_gate` 和 `w1` 内部还是 fp16 矩阵乘法。这意味着 `gate` 和 `up` 都是 fp16，乘积照样溢出。

### ✅ 正确修复

```python
def forward(self, x):
    input_dtype = x.dtype
    with torch.amp.autocast('cuda', enabled=False):  # ← 彻底关闭 autocast
        x = x.float()
        gate = F.silu(self.w_gate(x))   # 真正的 fp32 matmul
        up = self.w1(x)                  # 真正的 fp32 matmul
        x = gate * up                    # fp32 乘法，不会溢出
        x = self.drop(x)
        x = self.w2(x)
        x = self.drop(x)
    return x.to(input_dtype)  # 转回原精度返回
```

`autocast(enabled=False)` 是一个**上下文管理器**，在其作用域内完全禁用自动混合精度，`nn.Linear` 不再被劫持，`x.float()` 才真正生效。

### 验证修复效果

```
# 极端条件: 权重*15, 输入*10
NEW (autocast disabled): NaN=0  Inf=0   max_abs=17072
OLD (fp16 under autocast): NaN=0  Inf=512  prod_inf=4
```

---

## 精度选择分析 (RTX 5070 Laptop GPU, 8GB VRAM)

在决定是否从 `16-mixed` 改为 `32` (fp32) 时做了 VRAM 测试：

### VRAM 使用量 (MoE exp2, 含优化器 Adam)

| 精度 | batch=1 | batch=2 | batch=4 | batch=8 |
|------|---------|---------|---------|---------|
| **fp32** | 1.11 GB | 2.19 GB | 4.35 GB | **8.66 GB (109%)** ⚠️ |
| **fp16-mixed** | 0.82 GB | 1.58 GB | 3.09 GB | **6.12 GB (77%)** ✅ |
| **bf16-mixed** | 0.81 GB | 1.57 GB | 3.08 GB | **6.09 GB (77%)** ✅ |

> GPU 报告可用 VRAM: **7.96 GB**

### 结论

- **fp32 batch=8**: 超出 VRAM 8.8%，依赖 Windows WDDM 共享内存 (借系统 RAM) 勉强运行，速度慢且不稳定
- **fp16-mixed + SwiGLU 修复**: 安全，已验证无 NaN，VRAM 充裕
- **bf16-mixed**: 理论上最佳 (bf16 动态范围 = fp32，SwiGLU 天然不溢出)，但与已完成实验的精度不一致

**最终决定**: 保持 `precision: "16-mixed"`，SwiGLU 的 `autocast(enabled=False)` 修复已经彻底解决了 fp16 溢出问题。

---

## 受影响实验清单

### 已完成 (保持不动)

| 实验 | 最终 epoch | 状态 |
|------|----------|------|
| 20f baseline | epoch 38 | ✅ 有 earthformer_sevir.pt |
| 20f exp1_moe_flash | epoch 8 | ✅ 有 earthformer_sevir.pt |
| 49f baseline | epoch 9 | ✅ 有 earthformer_sevir.pt |
| 49f exp1_moe_flash | epoch 9 | ✅ 有 earthformer_sevir.pt |

### NaN 污染 (需删除 checkpoint 重训)

| 实验 | 状态 | 处理 |
|------|------|------|
| 20f exp2_swiglu_moe_flash | NaN from epoch 1 | 删除重训 |
| 20f exp3_balanced_moe_flash | NaN from epoch 2 | 删除重训 |
| 20f exp6_balanced_moe_rope_flash | epoch 0, SwiGLU | 删除重训 |
| 49f exp2_swiglu_moe_flash | NaN from epoch 0 | 删除重训 |

### 未开始 (需新训)

| 实验 | 说明 |
|------|------|
| 20f exp4_moe_rope_flash | 无 SwiGLU, 应无问题 |
| 20f exp5_swiglu_moe_rope_flash | 有 SwiGLU, 已修复 |
| 49f exp3~6 | 全部待训 |
