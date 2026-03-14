# 🌩️ Radar-JEPA: 基于掩码联合嵌入的4D雷达世界模型

> **目标**: 用 V-JEPA/MAE 自监督预训练方法，构建重庆雷达回波的 4D 时空特征提取器，
> 在隐空间中用 Transformer 预测未来，再反演回真实 3D 物理场。
>
> **硬件现实**:
> | 机器 | GPU | 显存 | 定位 |
> |------|-----|------|------|
> | 笔记本 (本机) | RTX 3050 Ti | **4 GB** | 调试/小规模推理 |
> | 好电脑 | RTX 5070 | **8.55 GB** | 主力训练机 |
> | 实验室服务器 | 4× TITAN RTX | 24 GB × 4 | 大规模预训练 (如有权限) |
>
> **数据源**: 重庆 TDMOSAIC 雷达 — 21 高度层, ~240帧/天, 384×384
>
> ⚠️ **算力约束决定一切架构选择** — 以 RTX 5070 (8.55GB) 为主力设计

---

## 一、核心思路（三阶段流水线）

```
阶段1: 预训练 3D/4D 特征提取器 (Encoder + Decoder)
  └─ 海量历史雷达 → 切4D时空Patch → 掩码80% → 重建/对齐 → 学会物理特征

阶段2: 冻结 Encoder/Decoder → 将全部数据编码为紧凑特征序列
  └─ (240, 21, 384, 384)  ──Encoder──>  (240, latent_dim)

阶段3: 训练轻量 Transformer → 特征预测特征
  └─ 前 210 帧特征 ──Transformer──>  后 30 帧特征
  └─ 后 30 帧特征 ──冻结Decoder──>  真实 3D 雷达重建
```

### 为什么这么做？
| 问题 | 解决方案 |
|------|----------|
| 21层 × 384×384 原始数据太大，不可能端到端训练 | Encoder 极限压缩到 (256, 24, 24) 或更小 |
| 高空(>10km)稀疏度达99%，大量算力浪费 | MAE 80%掩码 + 自适应特征提取，自动忽略空区 |
| 端到端预测远未来图像模糊 (blurry effect) | 空间重建与时序预测彻底解耦 |
| 算力有限 (RTX 5070 仅8.55GB) | 掩码后只处理10-20% token + fp16 + 梯度累积 |

---

## 1.5、硬件真实约束分析

### RTX 3050 Ti (4GB) — 只能做什么？
```
❌ 不可能: 训练任何 ViT (哪怕 ViT-Tiny 也需要 ~3GB 模型+激活，留不出 batch)
✅ 可以做:
   - 调试数据 pipeline (CPU模式, batch=1, 不反向传播)
   - 推理/评估 已训练好的小模型 (torch.no_grad + fp16)
   - SVD/小波等纯数学分解 (不吃显存)
   - 跑 Earthformer/DATSwinLSTM 评估 (已验证可行)
```

### RTX 5070 (8.55GB) — 主力训练配置

**核心策略: ViT-Tiny + 极致压缩 + fp16 + 梯度累积**

```
模型: ViT-Tiny (embed_dim=192, depth=12, heads=3, ~5M参数)
      ← 权重+优化器: ~0.15 GB (fp16+AdamW)

输入空间: 128×128 (不是384! 先下采样再训练)
      ← 128/16 = 8×8 = 64 空间tokens

时间帧:  8帧 (48分钟, 不是240帧)
Tubelet: 2帧合一 → 4 个时间tokens
      ← 总tokens = 64 × 4 = 256

掩码率:  90% → 实际只处理 256 × 10% = 26 tokens!
      ← 26 tokens 的 Attention 矩阵只有 26×26 = 676 个元素

显存预估:
  模型权重 (fp16):     ~0.15 GB
  激活值 (26 tokens):  ~1.5 GB
  梯度+优化器:         ~1.0 GB
  数据 (batch=2):      ~0.3 GB
  PyTorch 开销:        ~1.5 GB
  ─────────────────────────────
  总计:               ~4.5 GB / 8.55 GB  ✅ 可行!

梯度累积: accum_steps=8 → 等效 batch=16
```

### 如果有服务器 TITAN RTX (24GB) 的权限
```
可以升级为:
  模型: ViT-Small (384, 12层, 22M参数)
  输入: 192×192, 16帧
  掩码: 85%
  batch: 4-8
  ← 这是 Gemini 对话中讨论的理想配置
```

### 各阶段硬件分配策略
```
Phase 0 数据准备:     任意机器 (CPU为主)
Phase 1 VideoMAE验证: RTX 5070 (ViT-Tiny, 128×128, 8帧)
Phase 2 V-JEPA预训练: RTX 5070 (ViT-Tiny) 或 服务器 (ViT-Small)
Phase 3 Autoencoder:  RTX 5070 (轻量CNN, 逐帧训练, ~2GB)
Phase 4 Transformer:  RTX 5070 (隐空间极小, ~1GB) 或 RTX 3050Ti
Phase 5 评估:         RTX 3050Ti 也够 (torch.no_grad)
```

---

## 二、候选方案对比与推荐

### 方案 A: VideoMAE (首选快速验证)

| 项目 | 内容 |
|------|------|
| **GitHub** | https://github.com/MCG-NJU/VideoMAE |
| **论文** | NeurIPS 2022 Spotlight |
| **架构** | 3D ViT + Masked Autoencoder (像素重建) |
| **优势** | 原生 3D Conv3d token 化；90% 极高掩码率；代码清晰，数据加载易改；显存极省 |
| **劣势** | MAE 做像素重建，可能学到低级纹理而非高级动力学特征 |
| **适配难度** | ★★☆ (改 `in_chans`, `num_frames`, dataset 即可) |
| **RTX 5070配置** | `ViT-Tiny` (5M参数), 128×128, 8帧, 90%mask, batch=2, fp16 → **~4.5GB** |
| **服务器配置** | `ViT-Small` (22M参数), 192×192, 16帧, 90%mask, batch=4 → **~6GB** |
| **许可证** | CC-BY-NC 4.0 (非商业/学术OK) |

### 方案 B: V-JEPA (终极方案)

| 项目 | 内容 |
|------|------|
| **GitHub** | https://github.com/facebookresearch/jepa |
| **论文** | Bardes et al., 2024 — Yann LeCun 力推的世界模型 |
| **架构** | 3D ViT + Joint-Embedding (特征空间对齐，不重建像素) |
| **优势** | 提取高级语义/动力学特征(非纹理)；不需要 Decoder 做像素重建；理论上表征质量更高 |
| **劣势** | 需要 EMA Target Encoder (×2权重)+ Predictor，代码复杂度高；没有小模型预训练权重 |
| **RTX 5070配置** | `ViT-Tiny` (5M×2=10M含EMA), 128×128, 8帧, 85%mask → **~5.5GB** ⚠️偏紧 |
| **服务器配置** | `ViT-Small` (22M), 192×192, 30帧, 80%mask → **~8-10GB** |
| **许可证** | CC-BY-NC 4.0 |
| **⚠️ 注意** | V-JEPA 需要同时存 Context Encoder + Target Encoder (EMA)，显存消耗约为 VideoMAE 的 1.3-1.5 倍 |

### 方案 C: I-JEPA (图像级baseline)

| 项目 | 内容 |
|------|------|
| **GitHub** | https://github.com/facebookresearch/ijepa |
| **论文** | CVPR 2023 — 2D图像版JEPA |
| **架构** | 2D ViT + Joint-Embedding |
| **优势** | 代码最简洁；2D逻辑清晰；可作为"不含时间维度"的baseline |
| **劣势** | 纯2D，无时间维度；需自行扩展到3D/4D |
| **适配难度** | ★★ (但扩展到4D要重写多处) |
| **推荐模型** | `ViT-Base` (86M) — 单卡24GB轻松 |

### 方案 D: ClimaX (气象专用参考)

| 项目 | 内容 |
|------|------|
| **GitHub** | https://github.com/microsoft/ClimaX |
| **论文** | ICML 2023 — 微软气候基础模型 |
| **架构** | ViT + 多变量多层级输入 (Flatten-to-Channel) |
| **优势** | 专为气候设计；MIT License；支持多物理量 |
| **劣势** | 处理全球低分辨率数据(25km)，非雷达级高分辨率(1km) |
| **参考价值** | 学习它如何把多层级物理量编码进 ViT 的通道设计 |

### ⭐ 推荐路线 (适配 RTX 5070 8.55GB)

```
第一步 (快速验证，1-2周):
  → VideoMAE ViT-Tiny + 重庆雷达单层VIL + 128×128 + 8帧
  → 在 RTX 5070 上验证掩码重建能力，跑通pipeline
  → 预计显存 ~4.5GB，留有余量

第二步 (核心实验，2-4周):
  → V-JEPA ViT-Tiny + 21层3D雷达 (高度做通道)
  → 在 RTX 5070 上训练4D时空特征提取器
  → 如果显存紧张 → 降到 96×96 或 升 mask 到 92%

第三步 (完整模型):
  → 冻结V-JEPA Encoder → 轻量Transformer时序预测 → Decoder反演
  → Phase 4 的 Transformer 极轻，RTX 3050Ti 都能跑

如果拿到服务器权限:
  → 所有阶段升级为 ViT-Small + 192×192 + 16帧，效果飞跃提升
```

---

## 三、详细执行计划

### Phase 0: 数据准备与稀疏度验证 (Day 1-2)

**目标**: 用数据证明高空稀疏性，为论文 Introduction 提供科学依据

- [ ] 运行稀疏度探针脚本 `check_3d_sparsity.py`
  - 统计21个高度层的非零像素比例
  - 预期：0.5km层 ~30% 非零，15km层 ~0.1% 非零
- [ ] 准备 3D 原始数据（不做VIL积分）
  - 修改 `chongqing_to_vil_gpu_daily.py`，保存 `(240, 21, 384, 384)` 而非 `(240, 1, 384, 384)`
  - 或保存为 `(240, 21, 384, 384)` 的 `.npy` 文件

### Phase 1: VideoMAE 快速验证 (Day 3-7)

**目标**: 用最简单的方案跑通整条pipeline

```bash
# 克隆仓库
git clone https://github.com/MCG-NJU/VideoMAE.git

# 需要修改的文件
# 1. datasets.py — 添加 RadarDataset 类
# 2. modeling_pretrain.py — 修改 in_chans=1 (VIL单通道) 或 in_chans=21 (多层)
# 3. run_mae_pretraining.py — 数据加载配置
```

**关键参数 (RTX 5070 8.55GB 配置)**:
```yaml
model: vit_tiny_patch16        # ViT-Tiny! 不是 Small
mask_ratio: 0.90               # 90%掩码，只保留10% token
num_frames: 8                  # 8帧（48分钟的雷达）
tubelet_size: 2                # 时间方向每2帧一个patch
input_size: 128                # 128×128，从384下采样
in_chans: 1                    # 先用VIL单通道验证
batch_size: 2                  # 单卡batch (梯度累积=8 → 等效batch=16)
epochs: 100
lr: 1.5e-4
precision: fp16                # 必须开！
gradient_accumulation: 8
gradient_checkpointing: true   # 用时间换显存
```

**显存估算 (ViT-Tiny, 90% masking, fp16, RTX 5070)**:
- 空间Token: 128/16=8, 8×8=64
- 时间Token: 8/2=4
- 总Token: 64×4 = 256
- 掩码后: 256×10% = **26 tokens** → 极轻
- 模型权重 (fp16): ~0.15 GB
- 激活+梯度: ~1.5 GB  
- 优化器: ~0.8 GB
- 数据+开销: ~1.5 GB
- 总计: **~4.0 GB / 8.55 GB** ← 安全 ✅

**如果有服务器 (24GB)**:
```yaml
model: vit_small_patch16       # 升级到 ViT-Small (22M)
input_size: 192                # 空间更大
num_frames: 16                 # 时间更长
batch_size: 4
# 总Token: 12×12×8 × 10% ≈ 115 → 依然轻松
```

**验收标准**: 
- Loss 稳定下降
- Decoder 重建的雷达图与原图视觉相似
- 显存占用 < 12 GB

### Phase 2: V-JEPA 核心预训练 (Day 7-14)

**目标**: 训练4D气象特征提取器

```bash
git clone https://github.com/facebookresearch/jepa.git
```

**需要修改的关键文件**:
```
src/models/utils/patch_embed.py  → in_chans=21 (21高度层作通道)
src/models/vision_transformer.py → crop_size, num_frames 调整
app/vjepa/transforms.py          → 去掉RGB augmentation
src/datasets/                    → 自写 RadarDataset
src/masks/multiblock3d.py        → 可能需调整mask比例
configs/pretrain/                 → 新建 radar_vits16.yaml
```

**关键配置 (RTX 5070 8.55GB 版)**:
```yaml
# configs/pretrain/radar_vitt16.yaml
model_name: vit_tiny           # ViT-Tiny (5M参数) ← 8.55GB的极限
patch_size: 16
tubelet_size: 2
in_chans: 1                    # 先用VIL单通道; 之后试 in_chans=21
crop_size: 128                 # 128×128 (从384下采样)
num_frames: 8                  # 8帧片段 (48分钟)
mask_ratio: 0.85               # 85%掩码 (V-JEPA有EMA，比VideoMAE多吃显存)
predictor_depth: 4             # Predictor层数 (必须浅！)
batch_size_per_gpu: 2
num_gpus: 1                    # 单卡
mixed_precision: fp16
ema_momentum: 0.996            # EMA衰减率 (重要！)
gradient_checkpointing: true   # 必须开，换时间省显存
epochs: 200
lr: 1e-4
```

**⚠️ V-JEPA 显存比 VideoMAE 多 ~30%** (因为要同时存 Context Encoder + Target Encoder):
- Context Encoder (ViT-Tiny): ~0.15 GB
- Target Encoder (EMA copy): ~0.15 GB  
- Predictor (4层): ~0.05 GB
- 激活+梯度: ~2.0 GB (gradient_checkpointing)
- 优化器: ~0.8 GB
- 数据+开销: ~1.5 GB
- 总计: **~4.7 GB / 8.55 GB** ← 偏紧但可行 ✅

**如果 in_chans=21 (21高度层)**:
- 只增加 PatchEmbed3D 第一层卷积的权重 (~0.03 GB)
- 数据量增大: batch 需降为 1，配合梯度累积=16
- 或 crop_size 降为 96×96 确保安全

**服务器配置 (24GB, 如果有权限)**:
```yaml
model_name: vit_small          # 22M参数
crop_size: 192
num_frames: 30                 # 3小时片段
mask_ratio: 0.80
batch_size_per_gpu: 4
num_gpus: 4                    # DDP 4卡
```

**EMA 训练核心逻辑**:
```python
# 伪代码 — V-JEPA 训练循环的灵魂
for x in dataloader:
    # 1. Target Encoder (EMA, 不参与梯度)
    with torch.no_grad():
        target_features = target_encoder(x)  # 全局特征
    
    # 2. Context Encoder (学生, 只看未被遮挡的部分)
    context_features = context_encoder(x_masked)
    
    # 3. Predictor (极浅网络, 在特征空间预测被遮挡的特征)
    predicted_features = predictor(context_features, mask_positions)
    
    # 4. Loss: 特征空间对齐 (不是像素重建!)
    loss = F.smooth_l1_loss(predicted_features, target_features[masked])
    
    # 5. 反向传播 (只更新 context_encoder + predictor)
    loss.backward()
    optimizer.step()
    
    # 6. EMA 更新 target_encoder
    for p_target, p_context in zip(target_encoder.params, context_encoder.params):
        p_target.data = 0.996 * p_target.data + 0.004 * p_context.data
```

### Phase 3: Autoencoder 空间压缩器 (Day 10-14, 可与Phase 2并行)

**目标**: 训练 3D CNN Encoder-Decoder，将雷达图压缩/还原

```
RTX 5070 配置 (128×128 输入):
输入: (1, 128, 128) → Encoder → (128, 8, 8) → Decoder → (1, 128, 128)
压缩比: 128×128 / 128×8×8 = 16,384 / 8,192 ≈ 2:1

或 21层输入:
输入: (21, 128, 128) → Encoder → (128, 8, 8) → Decoder → (21, 128, 128)
压缩比: 21×128×128 / 128×8×8 ≈ 42:1
```

**架构要点**:
- **Encoder**: 4层 Conv2d (stride=2) + ResBlock + GroupNorm — 轻量！
- **Decoder**: 4层 ConvTranspose2d (stride=2) + ResBlock — **不用全连接！**
- Loss: MSE + 感知Loss (可选) + 质量守恒正则 (加分项)
- 参数量: ~2M → **显存 ~1.5 GB** ← RTX 3050Ti 都能训练！

**训练**: 不需要时间序列，逐帧独立训练。
数据量极大(几百天×240帧)，收敛快。

### Phase 4: Transformer 时序预测 (Day 14-21)

**目标**: 在隐空间中预测未来

```
冻结Encoder → 编码所有帧 → (240, latent_dim) 时序特征
Transformer: (210, latent_dim) → (30, latent_dim) 
冻结Decoder → 解码30帧 → 真实3D雷达
```

**Transformer 配置 (RTX 5070 够用, RTX 3050Ti 也行)**:
```yaml
model: Transformer (4层, 4头, dim=128)  # 极轻量
input_seq: 210  # 前21小时
output_seq: 30  # 后3小时
latent_dim: 128 × 8 × 8 = 8192  # 或 flatten 后 PCA 到 256 维
# 显存: ~0.5-1.0 GB ← RTX 3050Ti 4GB 轻松跑
```

**显存**: 极轻（隐空间维度远小于原始图像），RTX 3050Ti 单卡即可。

### Phase 5: 评估与对比 (Day 21-28)

**Baseline 对比组**:

| Baseline | 类型 | 备注 |
|----------|------|------|
| ConvLSTM | 经典端到端 | 最基础baseline |
| SVD + LSTM | 线性降维 + 时序 | 传统降阶模型 |
| 小波 + Transformer | 小波降维 + 时序 | 信号处理方案 |
| VideoMAE (Phase 1) | MAE像素重建 | 同源对比 |
| **Radar-JEPA (ours)** | JEPA特征预训练 | 打他们全部 |

**评估指标**:
- MSE, MAE (像素级)
- CSI, HSS, POD, FAR (@ 多阈值，如 VIL=16, 74, 133, 160)
- SSIM (结构相似度)
- 视觉对比图 (最后一帧还清不清楚？)

---

## 四、论文故事线 (Story)

### Title (暂拟)
> **Radar-JEPA: Self-Supervised 4D Spatiotemporal Representation Learning 
> for High-Resolution Radar Echo Extrapolation**

### Story 骨架
1. **Introduction**: 雷达短临预报重要 → 端到端方法算力大/图像模糊 → 
   高空数据极度稀疏(附稀疏度统计图) → 我们提出隐空间世界模型
2. **Method**: Encoder预训练(JEPA) + 时序预测(Transformer) + 物理反演(Decoder)
3. **Experiments**: 
   - 与ConvLSTM/SVD/小波对比 → 我们全面碾压
   - 消融实验: 掩码率影响/ViT-S vs ViT-B/有无EMA
4. **Visualization**: 
   - 3D雷达重建效果图
   - t-SNE 看隐空间特征聚类 (暴雨vs晴天是否分开？)
   - 3小时长序列预测 vs baseline的清晰度对比

### 潜在发文水平
| 级别 | 需要什么 |
|------|----------|
| **顶会海报 (AAAI/KDD/IJCAI)** | 跑通 pipeline + 指标优于 baseline |
| **顶会 Oral (NeurIPS/ICLR)** | + 物理守恒约束 + 地形融合 + 深入消融实验 |
| **顶刊 (Nature Comms级)** | + 真实极端天气案例分析 + 气象专家共同验证 |

---

## 五、风险与应对

| 风险 | 应对 |
|------|------|
| 预训练不收敛 | 先用 VideoMAE(更简单) 验证数据流水线正确性 |
| **RTX 5070 显存不够** | **降 crop_size (128→96→64)，升 mask_ratio (90→95%)，batch=1+梯度累积** |
| **ViT-Tiny 太弱学不到好特征** | **增加 depth (12→16层) 代替增加 width，或改用更大 patch_size=32** |
| Decoder 重建模糊 | 引入对抗性Loss (GAN) 或 VQ-VAE 离散化 |
| 特征空间塌缩 (V-JEPA) | 检查 EMA momentum (0.996) + Predictor 不能太深 |
| 长时序预测衰减 | 自回归 + scheduled sampling 训练策略 |
| **128×128 分辨率丢失细节** | **Phase 5 时用 super-resolution 上采样回 384** |

---

## 六、权威参考文献

### JEPA / MAE 核心
1. **V-JEPA** — Bardes et al., 2024. "Revisiting Feature Prediction for Learning Visual Representations from Video"
2. **I-JEPA** — Assran et al., CVPR 2023. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
3. **VideoMAE** — Tong et al., NeurIPS 2022. "Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
4. **MAE** — He et al., CVPR 2022. "Masked Autoencoders Are Scalable Vision Learners"

### 气象AI
5. **ClimaX** — Nguyen et al., ICML 2023. "ClimaX: A Foundation Model for Weather and Climate" (Microsoft)
6. **Earthformer** — Gao et al., NeurIPS 2022. "Earthformer: Exploring Space-Time Transformers for Earth System Forecasting"
7. **FourCastNet** — Pathak et al., 2022. "FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model"
8. **Pangu-Weather** — Bi et al., Nature 2023. "Accurate medium-range global weather forecasting with 3D neural networks"
9. **W-MAE** — Man et al., 2023. "W-MAE: Pre-trained Weather Model with Masked Autoencoder for Multi-variable Weather Forecasting"

### 降阶模型 (SVD/DMD + DL)
10. **DMD + Neural Networks** — 相关文献见 Dynamic Mode Decomposition 综述
11. **Physics-Informed Neural Networks** — Raissi et al., JCP 2019.

---

## 七、GitHub 仓库快速链接

| 仓库 | 链接 | License | 推荐度 |
|------|------|---------|--------|
| V-JEPA (Meta) | https://github.com/facebookresearch/jepa | CC-BY-NC 4.0 | ⭐⭐⭐⭐⭐ |
| I-JEPA (Meta) | https://github.com/facebookresearch/ijepa | CC-BY-NC 4.0 | ⭐⭐⭐ |
| VideoMAE (南大) | https://github.com/MCG-NJU/VideoMAE | CC-BY-NC 4.0 | ⭐⭐⭐⭐⭐ |
| MAE (Meta) | https://github.com/facebookresearch/mae | MIT | ⭐⭐⭐ |
| ClimaX (微软) | https://github.com/microsoft/ClimaX | MIT | ⭐⭐⭐⭐ |
| VideoMAE v2 | https://github.com/MCG-NJU/VideoMAEv2 | - | ⭐⭐⭐⭐ |

---

## 八、第一步行动清单

### 在好电脑 (RTX 5070) 上立刻开始:

```bash
# 1. 克隆 VideoMAE (快速验证用)
git clone https://github.com/MCG-NJU/VideoMAE.git
cd VideoMAE

# 2. 安装依赖
pip install timm==0.4.12 decord einops

# 3. 准备雷达数据 (把已有的 .npy 数据转为 VideoMAE 能读的格式)
# → 写一个 radar_dataset.py

# 4. 修改 in_chans 和 num_frames
# → 编辑 modeling_pretrain.py

# 5. RTX 5070 单卡跑起来 (不用 DDP!)
python run_mae_pretraining.py \
    --model pretrain_videomae_tiny_patch16 \
    --mask_ratio 0.9 \
    --batch_size 2 \
    --num_frames 8 \
    --input_size 128 \
    --epochs 100 \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 10 \
    --fp16
# 预计显存: ~4 GB / 8.55 GB
```

```bash
# 6. V-JEPA (核心方案, 在 RTX 5070 上)
git clone https://github.com/facebookresearch/jepa.git
cd jepa
# → 修改配置、数据加载
# → ViT-Tiny + 128×128 + 8帧 + 85%掩码
# → 预计显存: ~5 GB / 8.55 GB
```

### 在笔记本 (RTX 3050Ti 4GB) 上可以做:

```bash
# 调试数据pipeline (CPU模式)
python -c "from radar_dataset import RadarDataset; ds = RadarDataset(...); print(ds[0].shape)"

# SVD/小波 baseline (基本不吃显存)
python svd_baseline.py --n_components 64

# Phase 4 时序预测 (隐空间极小, ~1GB)
python train_latent_transformer.py --device cuda --fp16

# 模型评估/推理 (冻结参数, no_grad)
python evaluate.py --checkpoint best.pt --device cuda
```

### 如果能用服务器 (TITAN RTX 24GB):

```bash
# 所有阶段直接升级
python -m torch.distributed.launch --nproc_per_node=4 \
    run_mae_pretraining.py \
    --model pretrain_videomae_small_patch16 \
    --mask_ratio 0.9 \
    --batch_size 4 \
    --num_frames 16 \
    --input_size 192 \
    --epochs 200
```

---

*Created: 2026-03-13*
*Last Updated: 2026-03-13*
*Status: 计划阶段 — 待执行 Phase 0 数据准备与稀疏度验证*
