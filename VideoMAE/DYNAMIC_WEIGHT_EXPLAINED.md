# MAE 动态 Loss 加权实验 — 原理详解

## 一、为什么需要加权？问题在哪里？

### 1.1 背景：多 Loss 训练

在之前的高频 Loss 消融实验中，我们发现：

- **纯 MSE** 重建模糊（高频恢复仅 33.3%）
- **Sobel Edge Loss** 高频恢复最好（40.8%）
- **FFT Loss** 也有帮助（39.7%）

所以现在的做法是**同时用 3 个 Loss**：

```
L_total = w₁ × L_MSE + w₂ × L_FFT + w₃ × L_Sobel
```

其中：
| Loss | 计算方式 | 作用 |
|------|---------|------|
| **MSE** | 像素级均方误差 | 保证整体亮度/结构正确 |
| **FFT** | 频域幅度谱差异 | 保持频率分布一致 |
| **Sobel** | 梯度图差异 | 保持边缘/纹理锐利 |

### 1.2 核心问题：权重怎么设？

三个 Loss 的**量纲不同、数值范围不同**：

- MSE ≈ 0.03（像素空间，0~1 之间）
- FFT ≈ 0.13（频域幅度，数值可以很大）
- Sobel ≈ 0.56（梯度空间，也可以较大）

如果权重都设 1.0，那 Sobel 的梯度会主导优化方向，MSE 被忽略。固定权重 `[1.0, 0.5, 1.0]` 是人工拍脑袋的，**不一定最优**。

### 1.3 加权在哪里发生？

加权发生在 `mae_dynamic_weight.py` 训练循环的**每个 batch** 中：

```python
# 每个 batch 的前向过程:
mse_loss, pred_patches, mask = model(imgs)    # ① MAE 前向，得到 MSE loss
pred_img = model.unpatchify(full_pred)         # ② 重建完整图像
fft_loss = FFTLoss(pred_img, imgs)             # ③ 计算频域 loss
sobel_loss = SobelEdgeLoss(pred_img, imgs)     # ④ 计算边缘 loss

individual_losses = [mse_loss, fft_loss, sobel_loss]

# ⑤ 动态加权器决定 w₁, w₂, w₃
weights = dynamic_weighter.get_weights(individual_losses)

# ⑥ 加权求和得到总 loss
L_total = w₁ × mse_loss + w₂ × fft_loss + w₃ × sobel_loss

# ⑦ 反向传播
L_total.backward()
```

**关键点**：权重 `w₁, w₂, w₃` 不是固定的，而是根据不同策略**每个 batch 动态调整**。

---

## 二、五个实验的原理

### 实验 1：Fixed（固定权重基线）

**来源**：最朴素的做法，人工设定。

**原理**：
```
L_total = 1.0 × L_MSE + 0.5 × L_FFT + 1.0 × L_Sobel
```
权重在整个训练过程中**完全不变**。FFT 权重设 0.5 是因为其数值较大，避免主导梯度。

**作用**：作为**对照组**，其他 4 种动态方法都跟它比较。

**预期效果**：能用、不差，但不是最优——因为训练的不同阶段，三个 Loss 的最优比例其实不一样（早期该重 MSE 学大结构，后期该重 Sobel 补细节）。

---

### 实验 2：Uncertainty Weighting（不确定性加权）

**来源**：Kendall et al., **"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"**, CVPR 2018, arXiv:1705.07115

**核心思想**：每个 Loss 对应一个"不确定性" σ²，**不确定性高的 Loss → 权重低**。σ² 是可学习参数，跟着模型一起训练。

**数学公式**：

$$L_{total} = \sum_{i=1}^{3} \frac{1}{2\sigma_i^2} L_i + \log \sigma_i$$

- $\frac{1}{2\sigma_i^2}$：精度（precision），σ² 越小 → 权重越大
- $\log \sigma_i$：**正则项**，防止网络把所有 σ² 都调成无穷大（那样所有 Loss 权重都是 0，Loss 也是 0，但什么都没学到）

**实现**：
```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_losses):
        # s_i = log(σ²_i) 作为可学习参数，初始化为 0（即 σ²=1）
        self.log_vars = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_losses)])

    def forward(self, losses):
        for i, loss in enumerate(losses):
            precision = exp(-self.log_vars[i])  # 1/σ²
            total += precision * loss + self.log_vars[i]  # 1/σ² × L + log(σ²)/2
```

`log_vars` 参数加入优化器，跟模型参数一起通过梯度下降更新。

**直觉理解**：
- 训练初期，MSE 下降快（容易优化），σ²_MSE 会变小 → 权重变大
- Sobel 下降慢（hard to learn），σ²_Sobel 会变大 → 权重暂时变小
- 随着训练进行，MSE 接近收敛后下降变慢，σ²_MSE 变大 → 权重减小
- 这时 Sobel 开始有更大权重，优化方向自然转向边缘细节

**预期效果**：应该能自动找到接近最优的权重比例。缺点是增加了可学习参数（3 个标量），需要适当调 lr。

---

### 实验 3：EMA Loss Ratio Balancing（指数移动平均均衡）

**来源**：工程实践中常见的 heuristic，无单一论文出处。核心思路类似 GradNorm (Chen et al., ICML 2018) 的目标——让各个 Loss 的贡献保持均衡。

**核心思想**：**Loss 值大的 → 权重小，Loss 值小的 → 权重大**。使用指数移动平均（EMA）平滑各 Loss 的历史值，避免单步噪声干扰。

**数学公式**：

$$\text{EMA}_i^{(t)} = \alpha \cdot \text{EMA}_i^{(t-1)} + (1-\alpha) \cdot L_i^{(t)} \quad (\alpha = 0.99)$$

$$\tilde{w}_i = \frac{1}{\text{EMA}_i + \epsilon}$$

$$w_i = N \cdot \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}$$

其中 N=3（Loss 个数），保证权重之和 = 3。

**实现**：
```python
class EMALossBalancer:
    def get_weights(self, losses_values):
        for i, lv in enumerate(losses_values):
            self.ema[i] = 0.99 * self.ema[i] + 0.01 * lv  # 平滑更新
            weights.append(1.0 / self.ema[i])              # 大 loss → 小权重
        # 归一化使权重之和 = N
```

**直觉理解**：
- Sobel loss ≈ 0.56，EMA_Sobel 也约 0.56 → w_Sobel ≈ 1/0.56 ≈ 1.8
- MSE loss ≈ 0.03，EMA_MSE 也约 0.03 → w_MSE ≈ 1/0.03 ≈ 33.3
- 归一化后，MSE 权重远大于 Sobel → **自动补偿量纲差异**

**预期效果**：各 Loss 对梯度的贡献基本均衡，不会出现某个 Loss 主导的情况。简单有效，无需额外参数。缺点是纯 heuristic，没有理论最优保证。

---

### 实验 4：DWA — Dynamic Weight Averaging（动态权重平均）

**来源**：Liu et al., **"End-to-End Multi-Task Learning with Attention"**, CVPR 2019, arXiv:1803.10704

**核心思想**：**下降慢的 Loss → 权重大**。测量每个 Loss 的下降速率，下降慢说明这个任务更难/更需要关注，应该给更多权重。

**数学公式**：

下降速率: $r_i^{(t)} = \frac{L_i^{(t)}}{L_i^{(t-1)}}$

- $r_i < 1$：Loss 在下降（进展好）
- $r_i > 1$：Loss 在上升（进展差）
- $r_i \approx 1$：Loss 停滞

然后用带温度的 softmax 转换为权重：

$$w_i = N \cdot \frac{\exp(r_i / T)}{\sum_j \exp(r_j / T)} \quad (T = 2)$$

温度 T 控制权重的"集中度"：
- T → 0：winner-take-all（全部权重给下降最慢的）
- T → ∞：均匀分布（等权重）
- T = 2：折中

**实现**：
```python
class DWAWeighting:
    def get_weights(self, current_losses):
        ratios = [current_losses[i] / prev_losses[i] for i in range(N)]
        exp_r = [exp(r / T) for r in ratios]  # softmax
        weights = [N * e / sum(exp_r) for e in exp_r]
```

**直觉理解**：
- 假设 ep50→ep51：MSE 从 0.030→0.029（r=0.97，下降好）
- Sobel 从 0.56→0.57（r=1.02，没下降甚至上升）
- 那 DWA 会给 Sobel 更大权重，下一步优化更关注边缘质量

**预期效果**：自适应地把优化资源分配给"进展最差"的方面。但依赖相邻 epoch 的 Loss 变化，可能有噪声。

---

### 实验 5：RLW — Random Loss Weighting（随机 Loss 加权）

**来源**：Lin et al., **"Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning"**, TMLR 2022, arXiv:2111.10603

**核心思想**：**完全不用"智能"策略，每步随机采样权重**。作者的惊人发现是——随机权重居然跟很多精心设计的方法效果差不多！

**数学公式**：

$$\mathbf{w} \sim \text{Dirichlet}(\mathbf{1}_N) \quad \text{然后} \quad w_i \leftarrow N \cdot w_i$$

Dirichlet(1,1,1) 就是在"权重之和=1"的单纯形上均匀采样。乘以 N 使期望权重 = 1。

**逐步理解 Dirichlet 分布**：
- 想象一个三角形（3 个顶点对应 3 个 Loss）
- 每步在三角形内部**随机扔一个点**
- 点到各顶点的"距离"决定各 Loss 的权重
- 有时 MSE 权重大，有时 Sobel 大——完全随机

**实现**：
```python
class RLWWeighting:
    def __init__(self, n_losses):
        self.dist = torch.distributions.Dirichlet(torch.ones(n_losses))  # 均匀 Dirichlet

    def get_weights(self):
        w = self.dist.sample()  # 采样一组权重，和=1
        return (w * N).tolist()  # 缩放使期望=1
```

**为什么随机有用？** 论文的解释：
1. 随机权重提供了一种**隐式的正则化**——模型需要对各种权重组合都 robust
2. 避免了其他方法可能出现的**过拟合到特定权重模式**
3. 在梯度层面，随机权重相当于在多 Loss 的 Pareto 前沿上做随机探索

**预期效果**：意外地有效，可能跟精心设计的方法打平甚至更好。缺点是方差较大（每步随机），收敛曲线可能不够光滑。

---

## 三、加权在训练流程中的位置

```
每个 batch 的流程：

[输入图像 28×28]
       ↓
[MAE Encoder + Decoder]  →  重建图像 + MSE Loss
       ↓
[重建图像 vs 原图]
   ├── FFTLoss(重建, 原图)     → L_FFT
   ├── SobelEdgeLoss(重建, 原图) → L_Sobel
   └── MSE(重建, 原图)           → L_MSE
       ↓
╔══════════════════════════════════╗
║  🎯 动态加权器（5 种策略之一）  ║
║                                  ║
║  输入: [L_MSE, L_FFT, L_Sobel]  ║
║  输出: [w₁, w₂, w₃]            ║
║                                  ║
║  L_total = Σ wᵢ × Lᵢ           ║
╚══════════════════════════════════╝
       ↓
[L_total.backward()]  →  梯度
       ↓
[AdamW optimizer.step()]  →  更新模型参数
```

**注意**：验证时（val_loss）始终用**纯 MSE**，保证公平比较。

---

## 四、五个实验对比总结

| # | 方法 | 论文 | 权重更新频率 | 额外参数 | 核心策略 |
|---|------|------|-------------|---------|---------|
| 1 | **Fixed** | — | 永不更新 | 无 | 人工设定 `[1.0, 0.5, 1.0]` |
| 2 | **Uncertainty** | Kendall CVPR'18 | 每步梯度更新 | 3 个 log(σ²) | 不确定性大 → 权重小 |
| 3 | **EMA Ratio** | 工程实践 | 每步 EMA 更新 | 无 | Loss 大 → 权重小（均衡化） |
| 4 | **DWA** | Liu CVPR'19 | 每 epoch 更新 | 无 | 下降慢 → 权重大 |
| 5 | **RLW** | Lin TMLR'22 | 每步随机采样 | 无 | 完全随机（隐式正则化） |

### 预期排名猜测

1. **Uncertainty / DWA** — 最有可能胜出，因为它们有理论指导
2. **EMA** — 工程上稳健，量纲自动对齐
3. **RLW** — 可能意外地好（论文结论），也可能方差太大
4. **Fixed** — 合理的 baseline，不会太差

### 真正有趣的点

如果 RLW（纯随机）跟 Uncertainty（精心设计、有额外参数）效果差不多，那说明：
- 权重选择的**精确值可能没那么重要**
- 重要的是**多 Loss 的方向多样性**本身
- 这对我们后续迁移到 EarthFormer 有指导意义：不用太纠结权重调参

---

## 五、当前实验配置

```
模型: MAE-Small (dim=192, depth=8, heads=6, 4.04M params)
数据: Fashion-MNIST 28×28, patch_size=4, mask_ratio=75%
Loss: MSE + FFT + Sobel (3 个)
优化器: AdamW (lr=1.5e-4, wd=0.05)
LR Schedule: Cosine (周期200), warmup 10 epochs
Early Stopping: patience=15 (连续 15 次验证无改善则停止)
Max Epochs: 1000 (由 early stopping 自然决定停止时间)
VRAM: ~0.54 GB (RTX 3050 Ti)
每 epoch: ~40s (num_workers=0)
```

### 参考文献

1. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. *CVPR 2018*. arXiv:1705.07115
2. Chen, Z., Badrinarayanan, V., Lee, C.Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing. *ICML 2018*. arXiv:1711.02257
3. Liu, S., Johns, E., & Davison, A.J. (2019). End-to-End Multi-Task Learning with Attention. *CVPR 2019*. arXiv:1803.10704
4. Lin, B., Ye, F., Zhang, Y., & Tsang, I.W. (2022). Reasonable Effectiveness of Random Weighting. *TMLR 2022*. arXiv:2111.10603
5. Xu, Z., et al. (2021). Focal Frequency Loss for Image Reconstruction and Synthesis. *ICCV 2021*. arXiv:2012.12821
