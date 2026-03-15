# 第五代: 物理先验引导 / 领域知识注入 → 隐空间映射

> **核心思想**: 将**领域物理知识**直接编码进模型架构和训练过程，
> 打破"通用 AI 架构 + 数据驱动"的范式，用物理约束减少搜索空间、提高样本效率。

---

## 一、代表论文

| 论文 | 会议 | 核心创新 | 与我们的关系 |
|------|------|---------|------------|
| **FourCastNet** (Pathak et al., 2022) | arXiv / NVIDIA | AFNO (自适应傅里叶神经算子) 做全球天气预报 | 频域物理先验 |
| **Pangu-Weather** (Bi et al., 2023) | Nature 2023 | 3D Earth-Specific Transformer + 气压面分层 | **大气物理直接编码进架构** |
| **GenCast** (Price et al., 2024) | Nature 2024 | 条件扩散模型 + 球面几何 + 集合预报 | 概率天气预测的物理约束 |
| **GraphCast** (Lam et al., 2023) | Science 2023 | GNN on icosahedral mesh + 多步自回归 | 网格物理 + 消息传递 |
| **ClimaX** (Nguyen et al., 2023) | ICML 2023 | ViT + 多变量融合 + 迁移学习 | 气候基础模型 |
| **NowcastNet** (Zhang et al., 2023) | Nature 2023 | 物理演化 + 深度生成 (混合模型) | **唯一的临近预报论文** |
| **PreDiff** (Gao et al., 2024) | ICLR 2024 | 条件隐扩散 + 物理约束对齐 | 扩散模型 + 物理 loss |
| **CasCast** (Gong et al., 2024) | CVPR 2024 | 级联确定性-扩散框架做降水预报 | 多阶段预测架构 |
| **I-JEPA** (Assran et al., 2023) | CVPR 2023 | 联合嵌入预测 (不做像素重建) | JEPA 范式 |
| **V-JEPA** (Bardes et al., 2024) | TMLR 2024 | 视频 JEPA: 在特征空间预测被遮挡区域 | **我们最终要做的方向** |

## 二、参考 GitHub 仓库

| 仓库 | Stars | 说明 | 推荐度 |
|------|-------|------|--------|
| [NVIDIA/FourCastNet](https://github.com/NVlabs/FourCastNet) | 700+ | AFNO 全球天气预报 | ⭐⭐⭐⭐ |
| [198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather) | 800+ | 盘古气象模型 | ⭐⭐⭐⭐ |
| [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) | 4k+ | GraphCast (JAX) | ⭐⭐⭐⭐ |
| [microsoft/ClimaX](https://github.com/microsoft/ClimaX) | 900+ | ClimaX 气候基础模型 | ⭐⭐⭐⭐⭐ |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | 3k+ | V-JEPA 官方 (CC-BY-NC) | ⭐⭐⭐⭐⭐ |
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | 2.5k+ | I-JEPA 官方 | ⭐⭐⭐⭐⭐ |
| [tung-nd/climax](https://github.com/tung-nd/climax) | 同上 | ClimaX 更新版 | ⭐⭐⭐⭐ |
| [amazon-science/prediff](https://github.com/gaozhihan/prediff) | 200+ | PreDiff 条件隐扩散 | ⭐⭐⭐⭐ |
| [OpenEarthLab/CasCast](https://github.com/OpenEarthLab/CasCast) | 100+ | CasCast 级联预报 | ⭐⭐⭐⭐ |

## 三、核心理念: 从"数据驱动"到"物理引导"

### 3.1 五代范式演进

```
Gen1 ViT/MAE:       通用架构 + 数据驱动
Gen2 Swin:          局部性先验 (图像的空间局部性)
Gen3 Mamba:         序列先验 (时间因果性)
Gen4 动态路由:      数据自适应先验 (复杂度不均匀)
Gen5 物理引导:      ★ 领域物理先验 (气象学定律)
```

### 3.2 物理先验可注入的位置

| 注入位置 | 方式 | 例子 |
|----------|------|------|
| **数据层** | 物理量标准化、坐标编码 | VIL 对数变换, 经纬度编码 |
| **架构层** | 等变卷积、物理对称性 | GMR Conv (旋转等变) |
| **注意力层** | 物理掩码、密度加权 | 15dBZ 阈值掩码 |
| **损失函数** | 物理约束 loss | 守恒律、CSI loss |
| **预训练层** | 物理引导的掩码策略 | 只 mask 降水区域 |
| **后处理** | 物理一致性修正 | 非负约束、空间平滑 |

## 四、架构设计 (Fashion-MNIST 验证方案)

Fashion-MNIST 虽无"物理"，但有**领域先验**:
- 衣服/鞋子有对称性 → 等变特征
- 不同类别有不同纹理复杂度 → 自适应掩码
- 边缘比背景重要 → 密度加权

### 4.1 Physics-JEPA (物理引导联合嵌入)

```
Fashion-MNIST 28×28
       │
  [Context Encoder]──────────────────┐
  (可见 patches, 25%)                │     特征空间对齐
       │                             │     (不做像素重建!)
  Encoder 输出                       │
  z_context ∈ R^{K×D}           [Target Encoder (EMA)]
       │                         (被遮挡 patches, 75%)
  ┌────▼──────────┐                  │
  │  Predictor    │                  │
  │  (轻量 ViT)  │──→ ẑ_target     ↓
  └───────────────┘       ↔       z_target
                     L2 loss
                   (只在 masked 区域)
```

**关键差异 vs MAE**:
- MAE 在**像素空间**重建 → 学低级纹理
- JEPA 在**特征空间**预测 → 学高级语义

### 4.2 领域引导掩码策略

```python
# MAE: 随机掩码 (所有 patch 等概率)
mask = random_mask(49, ratio=0.75)

# 第五代: 语义引导掩码
# 策略1: 信息密度引导 — mask 更多"有信息"的区域 (更难的任务)
density = compute_edge_density(image)   # 边缘密度
p_mask = density / density.sum()        # 信息密集区域更高概率被 mask
mask = weighted_random_mask(49, ratio=0.75, weights=p_mask)

# 策略2: 结构化掩码 — mask 整个"语义区域" (如裙子下摆)
# → 强迫模型理解物体结构，而非逐 patch 记忆

# 策略3: 对称感知掩码 — 只 mask 一半，预测另一半
# → 利用衣服的对称性先验
```

### 4.3 关键设计选择

| 设计点 | JEPA 方案 | 说明 |
|--------|----------|------|
| Target Encoder | EMA 更新 (τ=0.999) | 缓慢追踪 context encoder |
| Predictor | 2层 ViT, dim=64 | 轻量，只做特征预测 |
| Loss | L2 + VICReg 正则 | 防止 representation collapse |
| 掩码 | multi-block masking (I-JEPA) | 2-4 个语义区域块 |
| 显存 | ~2× MAE (双 encoder) | RTX 3050Ti 仍可 (~0.2GB) |

## 五、与 MAE/Swin/Mamba/Dynamic 的全面对比

| 特性 | Gen1 MAE | Gen2 Swin | Gen3 Mamba | Gen4 动态 | **Gen5 物理** |
|------|----------|-----------|------------|-----------|---------------|
| 重建空间 | 像素 | 像素 | 像素 | 像素 | **特征空间** |
| 先验类型 | 无 | 局部性 | 因果性 | 自适应 | **物理/领域** |
| 掩码策略 | 随机 | 随机 | 随机 | 学习 | **物理引导** |
| 表征质量 | 中 | 中高 | 中 | 高 | **最高** |
| 下游任务迁移 | 良 | 良 | 中 | 良 | **优** |
| 实现复杂度 | 低 | 中 | 中 | 中高 | 高 |

## 六、从 Fashion-MNIST → 雷达的直接映射

| Fashion 概念 | 雷达对应 | 物理先验 |
|-------------|---------|---------|
| 衣服轮廓 | 降水区域边界 | 15dBZ 阈值 = 降水/非降水 |
| 纹理区域 | 回波强度分布 | VIL 密度 = 降水强度 |
| 背景 | 晴空区域 (66%) | 可安全掩码/合并 |
| 对称性 | 气象涡旋/锋面 | 旋转等变 (GMR) |
| 类别 (T-shirt vs Coat) | 天气系统类型 (对流 vs 层状) | 不同物理机制 |

### 雷达 JEPA 最终目标

```
阶段1: Fashion-MNIST JEPA (本实验)
  → 验证特征空间预测的可行性
  → 验证物理引导掩码 vs 随机掩码的差异

阶段2: SEVIR VIL JEPA
  → 将掩码策略换成 15dBZ 物理阈值
  → Encoder 输入 37帧 (选可见), Predictor 预测 12帧 (被 mask)
  → 在特征空间做预测，不在像素空间

阶段3: Radar-JEPA 完整模型
  → 21层3D雷达 + 物理引导掩码 + GMR等变
  → Encoder: 学习4D时空物理特征
  → Predictor: 在隐空间预测未来
  → 无需 Decoder 做像素重建 → 隐空间直接服务下游
```

## 七、⚠️ 注意事项

1. **Representation Collapse**: JEPA 最大风险 — encoder 输出常量即可让 loss=0。
   **解决**: VICReg loss (方差+不变性+协方差正则)，或 DINO 风格的 centering
2. **EMA 系数**: τ 太大 → target 更新太慢 → 学不到新知识；τ 太小 → 不稳定。
   **推荐**: 从 0.996 cosine 增到 0.999
3. **Predictor 容量**: 太大 → predictor 自己就能记住，encoder 不学东西；
   太小 → 预测不准。I-JEPA 推荐 predictor 远小于 encoder
4. **Multi-block masking**: I-JEPA 论文验证过，mask 4 个连续块 (而非随机 patch) 学到更好的语义特征

## 八、实现路线图

```
Week 1: Fashion-MNIST MAE baseline (已完成 ✅)
Week 2: Fashion-MNIST JEPA (特征空间预测, 不做像素重建)
  - Context Encoder + Target Encoder (EMA) + Predictor
  - Multi-block masking
  - VICReg 防崩塌

Week 3: 对比实验
  - MAE vs JEPA: 隐空间分离度, linear probe accuracy
  - 随机掩码 vs 领域引导掩码
  - 消融: predictor 大小, EMA τ, mask block 数量

Week 4: 迁移到 SEVIR
  - 将 Fashion-MNIST JEPA 的架构移植到 SEVIR VIL
  - 15dBZ 物理掩码替换随机掩码
  - 评估隐空间质量 → 用于下游时序预测
```

---

## 九、参考文献

```bibtex
@inproceedings{assran2023ijepa,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle={CVPR},
  year={2023}
}

@article{bardes2024vjepa,
  title={V-JEPA: Latent Video Prediction for Visual Representation Learning},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={TMLR},
  year={2024}
}

@article{pathak2022fourcastnet,
  title={FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model using Adaptive Fourier Neural Operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and others},
  journal={arXiv:2202.11214},
  year={2022}
}

@article{bi2023pangu,
  title={Accurate medium-range global weather forecasting with 3D neural networks},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Hengheng and Chen, Xin and Gu, Xiaotao and Tian, Qi},
  journal={Nature},
  year={2023}
}

@article{zhang2023nowcastnet,
  title={Skilful nowcasting of extreme precipitation with NowcastNet},
  author={Zhang, Yuchen and Long, Mingsheng and others},
  journal={Nature},
  year={2023}
}

@article{price2024gencast,
  title={GenCast: Diffusion-based ensemble forecasting for medium-range weather},
  author={Price, Ilan and Sanchez-Gonzalez, Alvaro and Alet, Ferran and others},
  journal={Nature},
  year={2024}
}
```
