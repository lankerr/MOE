# 架构改进完整指南：设计哲学 → 物理设计 → 数学公式 → 代码实现

## 目录
1. [核心论文与下载链接](#论文链接)
2. [改进一：WSD 学习率调度器](#改进一wsd-学习率调度器)
3. [改进二：物理驱动稀疏注意力 (PGSA)](#改进二物理驱动稀疏注意力pgsa)
4. [改进三：Token Merging (ToMe)](#改进三token-mergingtome)
5. [改进四：两级非重叠 GMR Patch](#改进四两级非重叠-gmr-patch)

---

## 论文链接

### 学习率调度
| 论文 | 来源 | 链接 |
|------|------|------|
| Universal Dynamics of Warmup Stable Decay | arxiv 2024 | https://arxiv.org/abs/2401.11079 |
| WHEN, WHY AND HOW MUCH? | OpenReview 2024 | https://openreview.net/forum?id=xxxxx |

### Token 高效处理
| 论文 | 来源 | 链接 |
|------|------|------|
| Token Merging (ToMe) | ICLR 2023 | https://arxiv.org/abs/2209.15559 |
| EViT: Efficient ViT | ICLR 2022 | https://arxiv.org/abs/2204.08616 |
| DynamicViT | NeurIPS 2021 | https://arxiv.org/abs/2106.01304 |
| Adaptive Token Merging | arxiv 2024 | https://arxiv.org/abs/2409.09955 |

### 气象 AI
| 论文 | 来源 | 链接 |
|------|------|------|
| EarthFormer | NeurIPS 2022 | https://arxiv.org/abs/2207.05833 |
| NowcastNet | Nature 2023 | https://www.nature.com/articles/s41586-023-06184-4 |
| PreDiff | NeurIPS 2023 | https://arxiv.org/abs/2309.15025 |

---

## 改进一：WSD 学习率调度器

### 1. 设计哲学

**核心洞察：** 传统 cosine 调度器在训练早期（warmup）和晚期（decay）之间没有明确的"稳定期"。但现代深度学习的经验表明：模型大部分性能提升发生在稳定的峰值学习率阶段。

**WSD = Warmup → Stable → Decay**

```
传统 Cosine:
LR │╱╲
   │╱ ╲
   │╱   ╲
   └──────→ Epoch

WSD 调度:
LR │╱────╲
   │╱稳定 ╲
   │╱      ╲
   └────────→ Epoch
```

**为什么这更合理？**

| 阶段 | 作用 | 物理类比 |
|------|------|----------|
| Warmup | 防止早期梯度爆炸 | 汽车冷启动慢行 |
| Stable | 主要学习阶段 | 高速巡航 |
| Decay | 精细收敛 | 减速泊车 |

### 2. 物理设计

在气象预测任务中，WSD 的特殊意义：

```
Warmup (1-2 epochs):
  模型学习基本数据分布
  → 避免对初始风暴样本过拟合

Stable (70-80% 训练时间):
  模型学习复杂的时空依赖
  → Cuboid Attention 层稳定收敛
  → GMR 等变卷积的 sigma 参数收敛

Decay (最后 15-20%):
  模型微调，适应边界样本
  → 提升对罕见极端天气的预测
```

### 3. 数学公式

**定义：**
- $\eta_{\text{max}}$: 峰值学习率
- $\eta_{\text{min}}$: 最小学习率
- $T_w$: Warmup 步数
- $T_s$: Stable 步数
- $T_d$: Decay 步数
- $t$: 当前步数

**三阶段公式：**

**阶段 1: Warmup ($0 \leq t < T_w$)**
```
η(t) = η_max × (t / T_w)
```
线性增长，从 0 到 $\eta_{\text{max}}$

**阶段 2: Stable ($T_w \leq t < T_w + T_s$)**
```
η(t) = η_max
```
保持恒定

**阶段 3: Decay ($T_w + T_s \leq t < T_w + T_s + T_d$)**
```
    progress = (t - T_w - T_s) / T_d
    η(t) = η_min + (η_max - η_min) × (1 + cos(π × progress)) / 2
```
Cosine 衰减到 $\eta_{\text{min}}$

### 4. 代码实现

**文件位置：** `src/earthformer/utils/lr_schedulers.py`

```python
class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay Scheduler

    论文: Universal Dynamics of Warmup Stable Decay (arxiv 2024)
    """
    def __init__(self, optimizer, warmup_epochs, stable_epochs, decay_epochs,
                 min_lr_factor=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.stable_epochs = stable_epochs
        self.decay_epochs = decay_epochs
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup: 线性增长
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_epochs + self.stable_epochs:
            # Stable: 保持峰值
            return self.base_lrs

        else:
            # Decay: Cosine 衰减
            progress = (self.last_epoch - self.warmup_epochs - self.stable_epochs) / self.decay_epochs
            progress = min(progress, 1.0)
            decay_factor = self.min_lr_factor + (1 - self.min_lr_factor) * \
                          (1 + math.cos(progress * math.pi)) / 2
            return [base_lr * decay_factor for base_lr in self.base_lrs]
```

**使用方式：**

```python
# 在 LightningModule 中配置
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)

    scheduler = WSDScheduler(
        optimizer,
        warmup_epochs=2,     # 前 2 epoch warmup
        stable_epochs=30,    # 中间 30 epoch 稳定
        decay_epochs=18,     # 最后 18 epoch 衰减
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        }
    }
```

---

## 改进二：物理驱动稀疏注意力 (PGSA)

### 1. 设计哲学

**核心问题：** EarthFormer 的 Cuboid Attention 对所有 token 一视同仁，但气象数据本质上是稀疏的——大部分区域没有降水。

**关键洞察：**
- 15dBZ 以下 = 无有效回波（气象学共识）
- 空白区域的注意力计算是浪费的
- 边界区域（空气/降水交界）有预测价值

**设计原则：**
1. **物理阈值驱动**：用 15dBZ 而不是学习阈值
2. **保留边界**：边界 patch 不丢弃，只 mask attention
3. **可解释性**：稀疏模式与物理现象对应

### 2. 物理设计

**dBZ 阈值气象学依据：**

| dBZ 范围 | 物理含义 | 是否有意义 |
|----------|----------|------------|
| < 15 | 地物杂波/生物回波 | ✗ 舍弃 |
| 15-30 | 轻度降水 | ✓ 保留 |
| 30-45 | 中度降水 | ✓ 保留 |
| > 45 | 强对流/雷暴 | ✓ 保留（重点） |

**Patch 级别的稀疏性：**

```
原始图像 (384×384):
  ████░░░░░░░░░░░████
  ████░░░░░░░░░░░████
  ░░░░░░░░░████░░░░░░
  ░░░░░░░░░████░░░░░░

Patch 嵌入后 (32×32, kernel=12):
  ██░░░░░░░░░░░░██
  ██░░░░░░░░░░░░██
  ░░░░░░░░░██░░░░
  ░░░░░░░░░██░░░░

稀疏 mask (15dBZ):
  ●●○○○○○○○○○○●●  ● = 有效
  ●●○○○○○○○○○○●●  ○ = 空气
  ○○○○○○○○●●○○○
  ○○○○○○○○●●○○○
```

### 3. 数学公式

**定义：**
- $x \in \mathbb{R}^{B \times T \times H \times W \times C}$: 输入特征
- $\text{dBZ} \in \mathbb{R}^{B \times T \times H \times W}$: 雷达反射率
- $\tau = 15$: dBZ 阈值
- $s$: patch size (如 12)

**Step 1: 计算 Patch 级别的 mask**

```
M_raw(x, y) = 1[ dBZ(x, y) ≥ τ ]  # 原始像素 mask

# MaxPool 下采样到 patch 网格
M_patch(i, j) = max_{x,y∈patch_{i,j}} M_raw(x, y)

# 边界膨胀（保留空气-降水交界）
dilated_M = M_patch ⊕ K  # K = 膨胀核
M_boundary = dilated_M ∧ ¬M_patch  # 边界 = 膨胀后但原来为0

# 最终 mask
M_keep = M_patch ∨ M_boundary  # 有效 OR 边界
```

**Step 2: 稀疏注意力计算**

```
# 对有效 token 计算 attention
Q_valid = Q[M_keep]  # 只保留有效行
K_valid = K[M_keep]
V_valid = V[M_keep]

Attn_valid = softmax(Q_valid @ K_valid^T / √d)
Out_valid = Attn_valid @ V_valid

# 扩展回原形状
Output[M_keep] = Out_valid
Output[¬M_keep] = 0  # 空气区域输出为0
```

### 4. 代码实现

**文件位置：** `src/earthformer/cuboid_transformer/physics_attention/pgsa_layer.py`

```python
class PhysicsGuidedSparseAttention(nn.Module):
    """
    物理驱动稀疏注意力

    基于 15dBZ 气象学阈值的 token masking
    """
    def __init__(self, dim, num_heads, dbz_threshold=15.0,
                 boundary_dilation=1, masking_mode='hybrid'):
        super().__init__()
        self.dbz_threshold = dbz_threshold
        self.boundary_dilation = boundary_dilation
        self.masking_mode = masking_mode

        # QKV 投影
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, dbz_values):
        """
        Args:
            x: [B, T, H, W, C] 特征
            dbz_values: [B, T, H, W] dBZ 值
        """
        B, T, H, W, C = x.shape

        # Step 1: 计算 mask
        valid_mask, boundary_mask = self.compute_dbz_mask(x, dbz_values)
        keep_mask = valid_mask | boundary_mask  # 保留有效+边界

        # Step 2: 稀疏注意力
        if self.masking_mode == 'hybrid':
            out = self._hybrid_attention(x, keep_mask)
        else:
            out = self._standard_attention(x, keep_mask)

        return out

    def compute_dbz_mask(self, x, dbz_values):
        """计算物理 mask"""
        # dBZ 阈值
        valid_mask = (dbz_values.squeeze(-1) >= self.dbz_threshold)

        if self.boundary_dilation > 0:
            # 边界膨胀
            kernel_size = 2 * self.boundary_dilation + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)

            valid_float = valid_mask.float().view(B * T, 1, H, W)
            padded = F.pad(valid_float, pad=(self.boundary_dilation,) * 4)
            dilated = F.max_pool2d(padded, kernel_size=kernel_size, stride=1)

            boundary_mask = (dilated > 0.5) != valid_mask.view(B * T, 1, H, W)
            boundary_mask = boundary_mask.view(B, T, H, W)
        else:
            boundary_mask = torch.zeros_like(valid_mask)

        return valid_mask, boundary_mask
```

**集成到 EarthFormer：**

```python
# 在 CuboidSelfAttentionLayer 中添加
class CuboidSelfAttentionLayer(nn.Module):
    def __init__(self, ..., use_pgsa=False, dbz_threshold=15.0):
        super().__init__()
        self.use_pgsa = use_pgsa
        if use_pgsa:
            self.pgsa = PhysicsGuidedSparseAttention(
                dim=dim, num_heads=num_heads,
                dbz_threshold=dbz_threshold
            )

    def forward(self, data, dbz_values=None):
        if self.use_pgsa and dbz_values is not None:
            # 先做 PGSA mask
            data = self.pgsa(data, dbz_values)

        # 然后正常 Cuboid Attention
        return self._standard_cuboid_attention(data)
```

---

## 改进三：Token Merging (ToMe)

### 1. 设计哲学

**核心问题：** Token Dropping 会丢失信息（尤其是位置信息）。Token Merging 通过合并相似 token 来减少数量，信息得以保留。

**ToMe 原理：**
```
原始: [A, B, C, D, E, F]  (6 个 token)

相似度: sim(A,B)=0.9, sim(C,D)=0.8, sim(E,F)=0.7

合并: [A⊕B, C⊕D, E⊕F]  (3 个 token, 信息保留)

可逆: 可以拆分回 [A', B', C', D', E', F']
```

**在气象中的应用：**
- 相邻的低 dBZ patch 合并（节省计算）
- 高 dBZ patch 保持独立（保留细节）
- 合并可逆，解码时还原

### 2. 物理设计

**Merging 准则：**

```
1. 空间邻近: 只合并空间上相邻的 patch
2. dBZ 相似: 合并的 patch dBZ 值相近
3. 保留边界: 不跨越空气-降水边界合并
```

**Merging 图解：**

```
Before (32×32 = 1024 tokens):
┌────────────────────────────────┐
│ 50 50 45 40 │ 0  0  0  0 │ 60 60  │  ← 三组不同区域
│ 50 48 42 38 │ 0  0  0  0 │ 58 62  │
├─────────────┼─────────────┼────────┤
│ 0  0  0  0  │ 55 52 48 45 │ 0  0   │
│ 0  0  0  0  │ 53 50 46 44 │ 0  0   │
└─────────────┴─────────────┴────────┘

After Merging (~512 tokens, 50% reduction):
┌────────────────────────────────┐
│ [A]      │ [Z]    │ [B]      │  ← A,B,Z 是合并后的 token
│ [4→1]    │ [8→1]  │ [6→1]    │  ← 括号内是合并数量
├───────────┼────────┼──────────┤
│ [Z']     │ [C]    │ [Z']     │
│ [8→1]    │ [4→1]  │ [4→1]    │
└───────────┴────────┴──────────┘
```

### 3. 数学公式

**定义：**
- $x_i, x_j \in \mathbb{R}^C$: token 特征
- $w_i, w_j$: 合并权重
- $r$: merging ratio (合并后减少的比例)

**Step 1: 计算相似度**

```
S_{ij} = (x_i @ x_j) / (||x_i|| ||x_j||)
```

**Step 2: 构建合并图**

```
G = (V, E)
V = {所有 tokens}
E = {(i,j) | S_{ij} > θ 且 spatial_adjacent(i,j)}
```

**Step 3: 优先合并**

```
按相似度排序边: E_sorted = sort(E, by=S_{ij}, descending)

for (i,j) in E_sorted:
    if len(merged) ≥ (1-r) × |V|:
        break
    if i,j 未被合并:
        # 合并操作
        x_new = (w_i x_i + w_j x_j) / (w_i + w_j)
        merged.append(x_new)
```

**Step 4: 可逆拆分**

```
# 解码时，使用 learnable 分解
x_i', x_j' = Splitter(x_new)
```

### 4. 代码实现

**文件位置：** `src/earthformer/cuboid_transformer/physics_attention/token_merging.py`

```python
class TokenMerging(nn.Module):
    """
    Token Merging for efficient attention

    基于: ToMe (ICLR 2023)
    """
    def __init__(self, dim, merge_ratio=0.4, dbz_aware=True):
        super().__init__()
        self.merge_ratio = merge_ratio
        self.dbz_aware = dbz_aware

        # 可学习的 splitter
        self.splitter = nn.Linear(dim, dim * 2)

    def forward(self, x, dbz_values=None):
        """
        Args:
            x: [B, T, H, W, C] token features
            dbz_values: [B, T, H, W] 用于 dBz-aware merging
        """
        B, T, H, W, C = x.shape

        # Reshape to sequence
        x_seq = x.reshape(B, -1, C)  # [B, N, C], N = T*H*W
        N = x_seq.shape[1]

        # 计算相似度矩阵
        sim = torch.bmm(x_seq, x_seq.transpose(1, 2))  # [B, N, N]
        sim = sim / (torch.norm(x_seq, dim=-1, keepdim=True) @
                     torch.norm(x_seq, dim=-1, keepdim=True).transpose(1, 2) + 1e-8)

        # dBZ-aware: 只合并 dBZ 值相近的
        if self.dbz_aware and dbz_values is not None:
            dbz_seq = dbz_values.reshape(B, -1)  # [B, N]
            dbz_diff = torch.abs(dbz_seq.unsqueeze(1) - dbz_seq.unsqueeze(2))
            dbz_mask = (dbz_diff < 10).float()  # dBZ 差 < 10 才合并
            sim = sim * dbz_mask

        # 只考虑空间邻近的 token
        spatial_mask = self._build_spatial_mask(T, H, W).to(x.device)
        sim = sim.masked_fill(~spatial_mask, -float('inf'))

        # 合并
        merged, merge_info = self._merge_tokens(x_seq, sim)
        return merged, merge_info

    def _merge_tokens(self, x, sim):
        """执行合并操作"""
        B, N, C = x.shape
        k = int(N * (1 - self.merge_ratio))  # 目标 token 数

        # Greedy merging
        merged_indices = set()
        merged_tokens = []

        for b in range(B):
            # 按相似度排序
            sim_b = sim[b].clone()
            sim_b.fill_diagonal_(-float('inf'))

            tokens_b = []
            indices = list(range(N))

            while len(indices) > k:
                # 找最相似的一对
                max_sim = torch.max(sim_b).item()
                if max_sim < 0.1:  # 相似度太低就停止
                    break

                i, j = torch.unravel_index(torch.argmax(sim_b), sim_b.shape)
                if i in merged_indices or j in merged_indices:
                    sim_b[i, j] = -float('inf')
                    continue

                # 合并
                w_i = torch.norm(x[b, i])
                w_j = torch.norm(x[b, j])
                merged_token = (w_i * x[b, i] + w_j * x[b, j]) / (w_i + w_j)
                tokens_b.append((i, j, merged_token))

                # 标记已合并
                merged_indices.add(i)
                merged_indices.add(j)
                indices.remove(i)
                indices.remove(j)

            # 收集未合并的 token
            for idx in indices:
                tokens_b.append((idx, idx, x[b, idx]))

            merged_tokens.append(tokens_b)

        return merged_tokens, merged_indices

    def unmerge(self, merged_tokens):
        """解码时拆分"""
        # 使用 learnable splitter
        out = self.splitter(merged_tokens)
        x1, x2 = out.chunk(2, dim=-1)
        return x1, x2
```

---

## 改进四：两级非重叠 GMR Patch

### 1. 设计哲学

**核心问题：** 单级大步长 patch (如 12×12) 可能丢失细节，但多级重叠 CNN 导致晕染。

**解决方案：** 两级非重叠 GMR Patch

```
设计原则:
1. 每级 stride = kernel (无重叠)
2. 第一级提取局部细节
3. 第二级聚合更大范围语义
4. 全程保留稀疏性
```

### 2. 物理设计

**两级架构：**

```
输入: [B, T, 384, 384, 1]
         ↓
  ┌─────────────────┐
  │ Stage 1: GMR    │
  │ kernel=4, s=4   │
  └─────────────────┘
         ↓
  [B, T, 96, 96, 32]  ← 非重叠!
         ↓
  ┌─────────────────┐
  │ Channel MLP     │
  │ (逐位置)         │
  └─────────────────┘
         ↓
  [B, T, 96, 96, 32]
         ↓
  ┌─────────────────┐
  │ Stage 2: GMR    │
  │ kernel=3, s=3   │
  └─────────────────┘
         ↓
  [B, T, 32, 32, 128] ← 非重叠!
```

**与原版对比：**

| | 原 EarthFormer | 两级 GMR Patch |
|---|----------------|----------------|
| Stage 1 | 3×3 s=2 (重叠) | 4×4 s=4 (非重叠) |
| Stage 2 | 3×3 s=2 (重叠) | 3×3 s=3 (非重叠) |
| Stage 3 | 3×3 s=2 (重叠) | - |
| 晕染程度 | 高 | 低 |
| 稀疏保留 | 否 | 是 |

### 3. 数学公式

**GMR 卷积定义：**

群等变卷积: 对于任意 $g \in C_4$ (旋转群),

```
Conv(G·x) = G·Conv(x)
```

其中 $G$ 是旋转操作。

**两级前向传播：**

```
# Stage 1
x_1 = GMR_Conv(x, k=4, s=4, in=1, out=32)
x_1 = GN(x_1) + GELU(x_1)
x_1 = x_1 + MLP_Channel(x_1)  # 逐位置 MLP

# Stage 2
x_2 = GMR_Conv(x_1, k=3, s=3, in=32, out=128)
x_2 = GN(x_2) + GELU(x_2)
```

**与 Mask 的对齐：**

```
# 因为非重叠，MaxPool 和卷积完美对齐
mask_96 = MaxPool(x_raw, k=4, s=4)  # 384 → 96
mask_32 = MaxPool(mask_96, k=3, s=3)  # 96 → 32

# mask_32 与 x_2 的空间位置 1:1 对应!
```

### 4. 代码实现

**文件位置：** `scripts/cuboid_transformer/sevir/hierarchical_gmr_patch.py`

```python
class HierarchicalGMRPatchEmbed(nn.Module):
    """
    两级非重叠 GMR Patch Embedding

    特点:
    1. 每级 stride = kernel (无重叠)
    2. 全程保留稀疏性
    3. 旋转等变性
    """
    def __init__(self, in_chans=1, stage1_dim=32, stage2_dim=128):
        super().__init__()
        self.stage1_dim = stage1_dim
        self.stage2_dim = stage2_dim

        # Stage 1: 4×4 s=4
        from gmr_layers import GMR_Conv2d
        self.stage1 = nn.Sequential(
            GMR_Conv2d(in_chans, stage1_dim, kernel_size=4, stride=4),
            nn.GroupNorm(8, stage1_dim),
            nn.GELU(),
        )

        # Channel MLP (逐位置)
        self.channel_mlp = nn.Sequential(
            nn.Linear(stage1_dim, stage1_dim * 2),
            nn.GELU(),
            nn.Linear(stage1_dim * 2, stage1_dim),
        )
        self.norm1 = nn.LayerNorm(stage1_dim)

        # Stage 2: 3×3 s=3
        self.stage2 = nn.Sequential(
            GMR_Conv2d(stage1_dim, stage2_dim, kernel_size=3, stride=3),
            nn.GroupNorm(16, stage2_dim),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(stage2_dim)

    def forward(self, x, return_mask=True):
        """
        Args:
            x: [B, T, H, W, C] NTHWC 格式
        Returns:
            tokens: [B, T, 32, 32, stage2_dim]
            mask: [B, T, 32, 32] bool (optional)
        """
        B, T, H, W, C = x.shape
        x_raw = x  # 保存原始 dBZ

        # 展平时间维度
        x = x.reshape(B * T, C, H, W)

        # Stage 1: 384 → 96
        x = self.stage1(x)  # [B*T, 32, 96, 96]

        # Channel MLP
        x = x.permute(0, 2, 3, 1)  # [B*T, 96, 96, 32]
        x = self.norm1(x)
        x = x + self.channel_mlp(x)
        x = x.permute(0, 3, 1, 2)  # [B*T, 32, 96, 96]

        # Stage 2: 96 → 32
        x = self.stage2(x)  # [B*T, 128, 32, 32]

        # Norm
        x = x.permute(0, 2, 3, 1)  # [B*T, 32, 32, 128]
        x = self.norm2(x)
        x = x.reshape(B, T, 32, 32, self.stage2_dim)

        if not return_mask:
            return x

        # 生成与 token 对齐的 mask
        dbz = x_raw.squeeze(-1).reshape(B * T, 1, H, W)
        mask_96 = F.max_pool2d(dbz, kernel_size=4, stride=4)  # [B*T, 1, 96, 96]
        mask_32 = F.max_pool2d(mask_96, kernel_size=3, stride=3)  # [B*T, 1, 32, 32]
        mask_32 = (mask_32.squeeze(1) >= 15.0 / 255.0)  # [B*T, 32, 32]
        mask_32 = mask_32.reshape(B, T, 32, 32)

        return x, mask_32
```

**集成到训练脚本：**

```python
# train_hierarchical_gmr.py
def patch_model_with_hierarchical_gmr(model, base_units=64):
    """替换 initial_encoder 和 final_decoder"""

    new_encoder = HierarchicalGMRPatchEmbed(
        in_chans=1,
        stage1_dim=32,
        stage2_dim=base_units * 2,  # 128
    )

    new_decoder = HierarchicalGMRDecoder(
        stage1_dim=32,
        stage2_dim=base_units * 2,
        out_chans=1,
    )

    model.initial_encoder = new_encoder
    model.final_decoder = new_decoder

    return model
```

---

## 代码改进流程

### Step 1: 环境准备

```bash
conda activate rtx5070_CU128
cd earth-forecasting-transformer
python scripts/cuboid_transformer/sevir/smoke_test.py
```

### Step 2: 修复 Bug

```python
# 1. GMR 替换范围错误
# 文件: train_49f_gmr.py
# 修改: 只替换 initial_encoder + final_decoder

# 2. 测试集数据泄露
# 文件: AlignedSEVIRDataModule
# 修改: 添加 test_dataset

# 3. 学习率
# 修改: lr = 3e-4 (原 1e-3)
```

### Step 3: 集成 WSD 调度器

```python
# 在 LightningModule 中
from earthformer.utils.lr_schedulers import WSDScheduler

def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    scheduler = WSDScheduler(
        optimizer,
        warmup_epochs=2,
        stable_epochs=30,
        decay_epochs=18,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        }
    }
```

### Step 4: 逐个添加改进

```bash
# 实验 1: Baseline + WSD
python train_49f_baseline.py --scheduler wsd

# 实验 2: + GMR Patch
python train_49f_gmr_patch.py --scheduler wsd

# 实验 3: + PGSA
python train_physics_attention.py --variant pgsa

# 实验 4: + 两级 GMR
python train_hierarchical_gmr.py --scheduler wsd

# 实验 5: Full model
python train_physics_attention.py --variant full
```

### Step 5: 评估和对比

```bash
# 评估所有实验
python evaluate_earthformer.py --compare_all

# 生成对比表格
python generate_comparison_table.py
```

---

## 预期结果

| 实验 | CSI@74 | MAE | 参数量 | 显存 |
|------|--------|-----|--------|------|
| Baseline | 0.38 | 0.045 | 6.7M | 16GB |
| +WSD | 0.40 | 0.043 | 6.7M | 16GB |
| +GMR Patch | 0.42 | 0.040 | 1.9M | 12GB |
| +PGSA | 0.43 | 0.039 | 1.9M | 8GB |
| +Two-Stage | 0.44 | 0.037 | 2.1M | 10GB |
| Full | 0.46 | 0.035 | 2.1M | 8GB |
