# 混合精度训练与 NaN 问题完全指南

> 时间: 2026年3月11日  
> 硬件: NVIDIA RTX 5070 Laptop GPU (8GB VRAM)  
> 框架: PyTorch Lightning + AMP  
> 最终方案: **`precision: "16-mixed"` + SwiGLU autocast 修复**

---

## 目录

1. [三种精度模式对比](#1-三种精度模式对比)
2. [NaN 的根因分析](#2-nan-的根因分析)
3. [修复方案与验证](#3-修复方案与验证)
4. [VRAM 实测数据](#4-vram-实测数据)
5. [为什么不用 fp32 / bf16-mixed](#5-为什么不用-fp32--bf16-mixed)

---

## 1. 三种精度模式对比

### 1.1 什么是 "精度"

神经网络中每个数字（权重、激活值、梯度）都用浮点数存储。不同浮点格式的区别：

| 格式 | 总位数 | 指数位 | 尾数位 | 最大值 | 最小正值 | 精度(有效十进制位) |
|------|--------|--------|--------|--------|---------|------------------|
| **fp32** | 32 | 8 | 23 | 3.4×10³⁸ | 1.2×10⁻³⁸ | ~7位 |
| **fp16** | 16 | 5 | 10 | **65504** | 6.1×10⁻⁵ | ~3.3位 |
| **bf16** | 16 | 8 | 7 | 3.4×10³⁸ | 1.2×10⁻³⁸ | ~2.4位 |

关键区别图示：

```
fp32:  [1 符号][8 指数][23 尾数] → 范围大，精度高，占4字节
fp16:  [1 符号][5 指数][10 尾数] → 范围小(max=65504)，精度中，占2字节
bf16:  [1 符号][8 指数][ 7 尾数] → 范围大(=fp32)，精度低，占2字节
```

### 1.2 PyTorch Lightning 中的三种模式

#### `precision: 32` — 纯 fp32

```
所有计算都用 fp32:
  权重: fp32 (4字节/参数)
  前向: fp32
  反向: fp32
  优化器状态: fp32

特点: 最安全，绝不会溢出，但是——
  - VRAM 占用最大 (权重+梯度+优化器 全是 fp32)
  - 速度最慢 (GPU Tensor Core 无法加速 fp32 matmul)
```

#### `precision: "16-mixed"` — fp16 混合精度 ⭐ 当前使用

```
混合使用 fp32 和 fp16:
  权重: fp32 (主副本) + fp16 (前向计算时自动转)
  前向: fp16 (大部分运算) + fp32 (部分敏感运算如 LayerNorm, Softmax)
  反向: fp16 (梯度) + GradScaler 防止梯度下溢
  优化器状态: fp32

PyTorch 的 autocast 自动决定每个运算用什么精度:
  - matmul, conv → fp16 (Tensor Core 加速)
  - layernorm, softmax, loss → fp32 (需要精度)
```

工作原理：

```python
# PyTorch Lightning 在幕后做的事:
scaler = torch.amp.GradScaler()

with torch.amp.autocast('cuda', dtype=torch.float16):
    # autocast 区域内:
    #   nn.Linear → 输入自动 cast 到 fp16, 用 Tensor Core 算 fp16 matmul
    #   nn.LayerNorm → 保持 fp32
    #   torch.softmax → 保持 fp32
    output = model(input)
    loss = criterion(output, target)

# GradScaler: 把 loss 乘以一个大数(比如 1024)再 backward
# 防止 fp16 梯度太小变成 0 (下溢)
scaler.scale(loss).backward()

# 检查梯度是否有 Inf/NaN:
#   如果有 → 跳过这个 step (不更新权重)
#   如果没有 → 正常 unscale + optimizer.step()
scaler.step(optimizer)
scaler.update()  # 动态调整 scale factor
```

#### `precision: "bf16-mixed"` — bf16 混合精度

```
和 16-mixed 结构相同，但用 bf16 替代 fp16:
  权重: fp32 (主副本) + bf16 (前向)
  前向: bf16 (大部分) + fp32 (敏感运算)
  反向: bf16
  优化器状态: fp32

关键差异: bf16 动态范围 = fp32 (最大值 3.4×10³⁸)
  → 不存在 fp16 的 65504 溢出问题
  → 不需要 GradScaler
  代价: 尾数只有 7 位 (fp16 有 10 位)，精度更低
```

### 1.3 "16-mixed" vs "16" vs "bf16-mixed" vs "bf16"

| 模式 | 含义 | VRAM | 速度 |
|-----|------|------|------|
| `32` | 纯 fp32 | 最大 | 最慢 |
| `"16-mixed"` | fp32 权重 + fp16 计算 + GradScaler | 中 | 快 |
| `16` | 纯 fp16 (权重也是 fp16) | 最小 | 快但**极易 NaN** |
| `"bf16-mixed"` | fp32 权重 + bf16 计算 | 中 | 快 |
| `"bf16"` | 纯 bf16 | 小 | 快但精度差 |

> ⚠️ `16` (纯 fp16) 和 `"16-mixed"` **完全不同**！  
> 纯 fp16 连权重主副本都是 fp16，优化器状态也是 fp16，训练极不稳定。  
> `"16-mixed"` 权重主副本和优化器状态保持 fp32，只有前向/反向用 fp16。

### 1.4 autocast 到底做了什么

`torch.amp.autocast` 会拦截每个运算符，根据预设规则决定精度：

```
自动用 fp16 (加速):          保持 fp32 (需要精度):
├── nn.Linear (matmul)       ├── nn.LayerNorm
├── nn.Conv2d                ├── nn.BatchNorm
├── torch.matmul             ├── nn.Softmax
├── torch.bmm                ├── nn.CrossEntropyLoss
└── torch.addmm             ├── torch.sum (大规模累加)
                             └── torch.log, torch.exp
```

**关键机制**: autocast 会**劫持 nn.Linear 的输入类型**。即使你在 autocast 区域内手动写 `x = x.float()`，传给 `nn.Linear` 时 autocast 还是会偷偷把它转回 fp16：

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    x = some_input.float()      # x 是 fp32
    y = nn.Linear(128, 256)(x)  # autocast 偷偷把 x cast 回 fp16！
    print(y.dtype)               # → torch.float16，不是 fp32！
```

这是我们 SwiGLU 修复中遇到的最大陷阱（后文详述）。

---

## 2. NaN 的根因分析

### 2.1 现象

使用 `precision: "16-mixed"` 训练含 SwiGLU 激活的 MoE 实验时：

| 实验 | 首次 NaN | 日志中 NaN 行数 |
|------|---------|---------------|
| exp2_swiglu_moe_flash (20f) | epoch 1, step 391 | 1790 |
| exp3_balanced_moe_flash (20f) | epoch 2, step 529 | 609 |
| 49f_exp2_swiglu_moe_flash | epoch 0 | 11 |

**不使用 SwiGLU 的实验 (baseline, exp1) 训练完全正常，0 个 NaN。**

### 2.2 SwiGLU 为什么特殊

标准 FFN（不会溢出）:

```
y = W₂(GELU(W₁(x)))
```

每一步的值域都有天然约束：
- `W₁(x)` → 矩阵乘法，值取决于权重和输入
- `GELU(·)` → 激活函数，输出值域 ≈ (-0.17, +∞)，但实际上权重初始化时输出不会很大
- `W₂(·)` → 再一次矩阵乘法

**没有任何一步是"两个可能很大的中间结果直接相乘"。**

SwiGLU（**有溢出风险**）:

```
y = W₂(SiLU(W_gate(x)) ⊙ W₁(x))
                         ↑
                    这里是逐元素乘法！
```

- `gate = SiLU(W_gate(x))` → 值域 (-0.28, +∞)
- `up = W₁(x)` → 值域 (-∞, +∞)
- `gate * up` → **两个无界值逐元素相乘** ← 灾难发生处

### 2.3 fp16 溢出的数学

fp16 最大可表示值: **65504**

假设训练了几个 epoch 后，某个 token 的：
- gate 值 = 300（一个不算很大的值）
- up 值 = 250

$$300 \times 250 = 75000 > 65504 \implies \text{Inf (溢出)}$$

在 fp32 中，$75000$ 完全在范围内（fp32 max = $3.4 \times 10^{38}$）。

**训练越久，权重越大，溢出概率越高**——这就是为什么 NaN 不是一开始就出现，而是训了几个 epoch 后才爆发。

### 2.4 溢出 → NaN 的传播链

```
gate * up = 300 * 250 = 75000 > 65504
  → fp16 表示为 Inf
    → W₂(Inf) = Inf 或 NaN
      → loss = NaN
        → 梯度 = NaN
          → 权重更新 = NaN
            → 所有后续输出 = NaN (永久性污染)
```

一旦权重被 NaN 污染，模型就永远恢复不了——即使后面的输入是正常的，NaN 权重 × 任何值 = NaN。

### 2.5 为什么 GradScaler 救不了

`GradScaler` 的机制是：检测到梯度中有 Inf/NaN → 跳过这个 step。

但问题是：
1. 溢出发生在**前向传播**（`gate * up`），不是梯度中
2. `GradScaler` 只在 `scaler.step()` 时检查梯度
3. 前向传播产生的 Inf 已经传播成 NaN loss
4. 即使跳过这个 step，**当前 batch 的前向传播已经产生了 NaN 输出**
5. 如果某些运算（如 EMA、running stats）用了这些 NaN 值，污染就扩散了

更关键的是：**如果每个 step 都溢出**（权重已经长到了总是产生 Inf 的程度），GradScaler 会持续跳过 step，等于训练停滞。

---

## 3. 修复方案与验证

### 3.1 修复原理

核心思路：SwiGLU 的 `gate * up` 必须在 fp32 下执行，因为 fp32 max = 3.4×10³⁸ 远大于可能的乘积值。

### 3.2 错误的修复（为什么 `x.float()` 没用）

```python
def forward(self, x):
    x = x.float()  # ← 看起来转了 fp32
    gate = F.silu(self.w_gate(x))  # ← 但 autocast 劫持了！
    up = self.w1(x)                # ← 实际上还是 fp16 matmul
    x = gate * up                  # ← 所以这里还是 fp16 乘法
    ...
```

验证：
```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    x_f32 = x_fp16.float()              # 确实是 fp32
    out = linear_layer(x_f32)            # autocast 拦截！
    print(out.dtype)                     # → torch.float16 😱
```

**autocast 的 nn.Linear 分发规则**: 无论输入是什么 dtype，都 cast 到 autocast 指定的 dtype (fp16) 来做矩阵乘法。`x.float()` 是自欺欺人。

### 3.3 正确的修复

```python
def forward(self, x):
    input_dtype = x.dtype
    with torch.amp.autocast('cuda', enabled=False):  # 彻底关闭 autocast
        x = x.float()                                # 真正的 fp32
        gate = F.silu(self.w_gate(x))                 # fp32 matmul ✓
        up = self.w1(x)                               # fp32 matmul ✓
        x = gate * up                                 # fp32 × fp32 = fp32 ✓
        x = self.drop(x)
        x = self.w2(x)                               # fp32 matmul ✓
        x = self.drop(x)
    return x.to(input_dtype)  # 转回原精度，融入后续 fp16 计算流
```

`torch.amp.autocast('cuda', enabled=False)` 是一个上下文管理器，在其作用域内：
- autocast 不再拦截任何运算符
- `nn.Linear` 按照实际输入 dtype 执行 → `float()` 输入 = fp32 matmul
- 退出这个 with 块后，外层的 autocast 恢复生效

### 3.4 对性能的影响

SwiGLU 内部 3 个矩阵乘法从 fp16 变成 fp32：

| 指标 | fp16 SwiGLU | fp32 SwiGLU | 影响 |
|-----|-------------|-------------|------|
| 单次 matmul 速度 | 1× | ~0.5× | 慢一倍左右 |
| SwiGLU 在整体前向中占比 | ~15-20% | ~15-20% | 仅影响这部分 |
| **整体训练速度影响** | — | **~5-10% 变慢** | 可以接受 |
| NaN 风险 | ❌ 高 | ✅ 零 | 决定性优势 |

**一个永远出 NaN 的快速训练 vs 稳健完成的稍慢训练，选择很明白。**

### 3.5 验证结果

极端条件测试（权重放大 15 倍，输入放大 10 倍）：

```
修复后 (autocast disabled): NaN=0  Inf=0   max_abs=17072
未修复 (fp16 autocast):     NaN=0  Inf=512  prod_inf=4 → 传播到输出
```

实际模型前向传播测试：

```
exp2_swiglu_moe_flash:   output NaN=False Inf=False ✓
exp3_balanced_moe_flash: output NaN=False Inf=False ✓
exp5_swiglu_moe_rope:    output NaN=False Inf=False ✓
exp6_balanced_moe_rope:  output NaN=False Inf=False ✓
```

### 3.6 修复代码位置

```
文件: datswinlstm_memory/modules/moe_layer.py
类:   SwiGLUExpert
方法: forward()
行:   ~235-245
```

---

## 4. VRAM 实测数据

测试模型: Earthformer + MoE (exp2_swiglu_moe_flash)，含 Adam 优化器 step。

### 不同 batch size

| 精度 | batch=1 | batch=2 | batch=4 | batch=8 |
|------|---------|---------|---------|---------|
| fp32 | 1.11 GB | 2.19 GB | 4.35 GB | **8.66 GB** |
| 16-mixed | 0.82 GB | 1.58 GB | 3.09 GB | **6.12 GB** |
| bf16-mixed | 0.81 GB | 1.57 GB | 3.08 GB | **6.09 GB** |

### batch=8 (训练实际使用) 分析

| 精度 | 峰值 VRAM | 占 7.96GB 的比例 | 是否安全 |
|------|----------|----------------|---------|
| fp32 | 8.66 GB | **109%** ⚠️ | 超出显存，靠系统 RAM 借用，速度显著下降 |
| **16-mixed** | **6.12 GB** | **77%** ✅ | 安全，留有 1.84GB 余量 |
| bf16-mixed | 6.09 GB | 77% ✅ | 安全，但与已完成实验精度不一致 |

### VRAM 怎么算出来的

以 16-mixed batch=8 为例：

```
权重 (fp32 主副本):        ~30 MB    ← 模型参数量较小
权重 (fp16 前向副本):      ~15 MB
激活值 (前向保存用于反向):  ~4.5 GB   ← 这是大头！8×20帧×384²×128维
梯度 (fp16):              ~15 MB
优化器 (Adam, 2×fp32):    ~60 MB
输入+输出张量:            ~1.4 GB
PyTorch 管理开销:         ~0.1 GB
─────────────────────────────────
总计:                     ~6.1 GB
```

fp32 翻倍的主要原因: **激活值从 fp16 变成 fp32**，占用直接翻倍 (4.5→9 GB)。

---

## 5. 为什么不用 fp32 / bf16-mixed

### 5.1 fp32 不可行

| 问题 | 详情 |
|------|------|
| VRAM 超标 | batch=8 需要 8.66 GB，你只有 7.96 GB |
| 借用系统 RAM | Windows WDDM 允许 GPU 借用系统 RAM，但速度慢 10-100× |
| 训练不稳定 | 每个 step 可能因 GPU↔CPU 数据搬运出现不可预测的延迟 |
| 速度 | 无 Tensor Core 加速，比 16-mixed 慢 ~2× |

如果要用 fp32，必须把 batch size 降到 4（4.35 GB），但这会改变训练动态（effective batch size 不同，学习率需要调整），与已完成实验不对等比较。

### 5.2 bf16-mixed 可行但不推荐

| 优点 | 缺点 |
|------|------|
| 动态范围 = fp32，SwiGLU 天然不溢出 | 与已完成的 baseline/exp1 (16-mixed) **精度类型不一致** |
| 不需要 GradScaler | 尾数只有 7 位 (fp16 = 10 位)，微小数值差异更大 |
| 和 16-mixed 一样快 | 需要改 YAML + 重新验证所有实验 |

对比赛/论文而言，**所有实验应该在相同精度设定下比较**。baseline 和 exp1 已经用 16-mixed 训完了，后续实验改成 bf16-mixed 会引入不可控变量。

### 5.3 16-mixed + SwiGLU 修复 = 最佳选择

```
✅ 与已完成实验精度一致 (16-mixed)
✅ SwiGLU 内部 fp32 计算，消除溢出风险
✅ VRAM 占用 77%，安全余量充足
✅ Tensor Core 加速大部分运算
✅ GradScaler 防止梯度下溢
✅ 已通过极端条件测试验证
```

---

## 附录: 常见混合精度 NaN 排查清单

如果将来遇到新的 NaN 问题，按以下顺序排查：

1. **定位 NaN 首次出现位置**
   ```python
   torch.autograd.set_detect_anomaly(True)  # 自动检测 NaN 产生处
   ```

2. **检查是否是 fp16 溢出**
   - 65504 是 fp16 上限
   - 任何 `a * b > 65504` 的元素级乘法都有风险
   - 常见嫌疑人: SwiGLU, Gating mechanisms, large residual connections

3. **检查是否是梯度下溢**
   - GradScaler 的 scale factor 持续下降 → 梯度太大
   - GradScaler 持续跳过 step → 梯度或 loss 中有 Inf

4. **修复模板**
   ```python
   # 对任何需要 fp32 精度的代码块:
   with torch.amp.autocast('cuda', enabled=False):
       x = x.float()
       # ... fp32 计算 ...
   return result.to(input_dtype)
   ```

5. **千万不要**
   - ❌ 只写 `x = x.float()` 不关 autocast (会被劫持)
   - ❌ 把整个模型改成 fp32 (VRAM 不够)
   - ❌ 关掉混合精度训练 (速度太慢)
   - ❌ 忽略 NaN 继续训练 (权重已永久污染)
