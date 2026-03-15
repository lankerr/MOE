# Physics-Guided Attention for EarthFormer

## 概述

本包实现了基于物理先验的稀疏注意力机制，用于改进 EarthFormer 在气象雷达数据预测任务中的表现。主要创新点包括：

### 三大创新

1. **Physics-Guided Sparse Attention (PGSA)**
   - 基于 15dBZ 物理阈值的 token masking
   - 减少 ~65% 无效计算
   - 保持边界信息的混合策略

2. **Density-Proximity Cross-Block Attention (DPCBA)**
   - 密度感知：基于 dBZ 均值打分
   - 物理邻近：基于时空距离的权重
   - 动态建立稠密块间的跨窗口连接

3. **Flash Attention 集成**
   - block-sparse API 兼容
   - 零额外显存开销

## 模块结构

```
physics_attention/
├── __init__.py                      # 模块导出
├── pgsa_layer.py                    # PGSA 实现
├── dpcba_layer.py                   # DPCBA 实现
├── flash_attention_compat.py        # Flash Attention 集成
└── README.md                        # 本文档
```

## 核心设计

### PGSA: 15dBZ 阈值稀疏注意力

```python
from physics_attention import PhysicsGuidedSparseAttention

pgsa = PhysicsGuidedSparseAttention(
    dim=128,
    num_heads=4,
    dbz_threshold=15.0,
    masking_mode='hybrid'  # 'token_drop', 'attention_mask', 'hybrid'
)

output = pgsa(features, dbz_values=dbz)
```

**三种 masking 策略：**

| 模式 | 说明 | 显存节省 | 边界保留 |
|------|------|----------|----------|
| `token_drop` | 直接丢弃低 dBZ token | ✓✓✓ | ✗ |
| `attention_mask` | 保留 token，屏蔽注意力 | ✓ | ✓✓ |
| `hybrid` | 混合：丢弃内部，保留边界 | ✓✓ | ✓✓ |

**边界膨胀策略：**
```python
# 边界 patch：自身 <15dBZ 但物理邻居 >15dBZ
boundary_mask = dilate(valid_mask) != valid_mask
```

### DPCBA: 密度-邻近跨块注意力

```python
from physics_attention import DensityProximityCrossBlockAttention

dpcba = DensityProximityCrossBlockAttention(
    dim=128,
    num_heads=4,
    num_connections=4,      # 每个块最多连接数
    density_weight=1.0,     # 密度权重
    proximity_weight=1.0,   # 邻近权重
)
```

**打分函数：**
```
score(i, j) = density_weight * √(density[i] * density[j]) +
              proximity_weight * exp(-dist(i, j) / σ)
```

### Flash Attention 兼容

```python
from physics_attention import PhysicsGuidedFlashAttention

pgsa_flash = PhysicsGuidedFlashAttention(
    dim=128,
    num_heads=4,
    cuboid_size=(2, 7, 7),
    dbz_threshold=15.0,
    use_flash_attention=True,
)
```

## 实验设计

### 对比实验

| 变体 | 描述 | 预期改进 |
|------|------|----------|
| Baseline | 原 EarthFormer | - |
| +GMR-Patch | 等变卷积 patch embedding | CSI@74 +3-5% |
| +PGSA | 15dBZ 稀疏注意力 | 速度 +40%, CSI@74 +1-2% |
| +PGSA+DPCBA | 完整物理注意力 | CSI@74 +3-5% |

### 消融实验

```
1. dBZ 阈值: [5, 10, 15, 20, 25]
2. masking_mode: ['token_drop', 'attention_mask', 'hybrid']
3. num_connections: [2, 4, 8, 16]
4. density_weight vs proximity_weight 比例
```

### 评估指标

- **CSI@74** (Critical Success Index at 74 dBZ): 主要指标
- **CSI@48, CSI@56, CSI@64**: 不同强度降水
- **显存占用**: GPU memory usage
- **训练速度**: samples/second

## 使用方法

### 1. 训练单个变体

```bash
# Baseline
python train_physics_attention.py --variant baseline

# PGSA only
python train_physics_attention.py --variant pgsa

# PGSA + DPCBA
python train_physics_attention.py --variant pgsa_dpcba
```

### 2. 对比所有变体

```bash
python train_physics_attention.py --variant all \
    --data_dir /path/to/sevir \
    --max_epochs 50 \
    --gpus 1
```

### 3. 自定义配置

```python
from physics_attention import create_pgsa_cuboid_attention

custom_layer = create_pgsa_cuboid_attention(
    dim=128,
    num_heads=4,
    cuboid_size=(2, 7, 7),
    dbz_threshold=20.0,        # 自定义阈值
    masking_mode='hybrid',
    # ... 其他 CuboidSelfAttentionLayer 参数
)
```

## 理论基础

### 为什么是 15dBZ？

根据气象学标准：
- **< 15 dBZ**: 地物杂波、生物回波、非降水云
- **15-30 dBZ**: 轻度降水
- **30-45 dBZ**: 中度降水
- **> 45 dBZ**: 强降水/雷暴

15dBZ 是有效回波的公认最低门限。

### 相关工作对比

| 论文 | 稀疏策略 | 物理先验 | 应用领域 |
|------|----------|----------|----------|
| Longformer (2021) | 滑窗 + 全局 token | ✗ | NLP |
| BigBird (2021) | 随机 + 滑窗 + 全局 | ✗ | NLP |
| Nuwä (2022) | 3DNA (学习型) | ✗ | 气象 |
| **本文** | **15dBZ 阈值** | **✓** | **气象** |

## 预期结果

基于类似工作的保守估计：

| 指标 | Baseline | +PGSA | +PGSA+DPCBA |
|------|----------|-------|-------------|
| CSI@74 | 0.550 | 0.560 | 0.575 |
| 显存 (GB) | 16 | 10 | 11 |
| 速度 (it/s) | 2.0 | 2.8 | 2.5 |

## 实现细节

### dBZ 值获取

```python
# 方法1: 从 VIL 估计
dbz = compute_dbz_from_vil(vil_data)

# 方法2: 从特征估计
dbz = compute_dbz_from_features(features, method='norm')

# 方法3: 使用实际雷达数据
dbz = radar_data['dBZ']  # 推荐
```

### 与 Cuboid Attention 集成

```python
# 包装现有 attention layer
from physics_attention import PGSAWrapper

base_layer = CuboidSelfAttentionLayer(dim=128, num_heads=4)
enhanced_layer = PGSAWrapper(
    base_attention_layer=base_layer,
    dbz_threshold=15.0,
    enable_pgsa=True,
)
```

## 依赖

```
torch>=1.12.0
pytorch-lightning>=1.8.0
einops>=0.6.0

# 可选 (Flash Attention)
flash-attn>=2.0.0
```

## 引用

如果使用本代码，请引用：

```bibtex
@article{physics_attention_2024,
  title={Physics-Guided Sparse Attention for Meteorological Forecasting},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

BSD 3-Clause License
