# Baseline Experiments 使用指南

## 实验优先级

### 高优先级实验（主要实验）

| 实验 | 数据集 | 配置文件 | 说明 |
|------|--------|----------|------|
| SEVIR 20帧 | SEVIR | `cfg_sevir_20frame_mae_mse.yaml` | EarthFormer基准，MAE+MSE loss |
| 重庆 1h→1h | 重庆 | `cfg_chongqing_1h_to_1h.yaml` | 10帧→10帧，MAE+MSE loss |
| 重庆 3h→3h | 重庆 | `cfg_chongqing_3h_to_3h.yaml` | 30帧→30帧，MAE+MSE loss |

### 低优先级实验（探索性）

| 实验 | 数据集 | 配置文件 | 说明 |
|------|--------|----------|------|
| SEVIR 2h→2h | SEVIR | `cfg_sevir_baseline_2h_to_2h.yaml` | 24帧→24帧探索 |
| SEVIR 4h→4h | SEVIR | `cfg_sevir_baseline_4h_to_4h.yaml` | 48帧→48帧探索 |

## 数据集说明

### 1. SEVIR 数据集 (美国雷达)
- 帧间隔: 5分钟
- 时间范围: 2017年6月13日 - 2017年10月15日
- 训练集: ~1738 样本
- 验证集: ~600 样本
- 测试集: 2017年9月15日后数据
- 覆盖: 美国东南部

### 2. 重庆雷达数据
- 帧间隔: 6分钟
- 格式: `day_simple_YYYYMMDD.npy` (240, 384, 384)
- 文件数: 224个
- 训练集: ~1400 样本
- 验证集: ~300 样本
- 测试集: ~300 样本
- 覆盖: 重庆地区 (山地地形)

## 运行命令

### 高优先级实验

```bash
# 1. SEVIR 基准实验 (与EarthFormer论文一致)
python train_baseline_experiments.py --exp sevir_20frame

# 2. 重庆 1h→1h
python train_baseline_experiments.py --exp chongqing_1h_to_1h

# 3. 重庆 3h→3h
python train_baseline_experiments.py --exp chongqing_3h_to_3h

# 所有高优先级实验
python train_baseline_experiments.py --priority high
```

### 快速测试 (1 epoch)

```bash
python train_baseline_experiments.py --exp sevir_20frame --test_run
python train_baseline_experiments.py --exp chongqing_1h_to_1h --test_run
```

## 实验配置说明

### SEVIR 20帧基准
- **输入**: 8帧 (40分钟)
- **输出**: 12帧 (60分钟)
- **Loss**: MAE + MSE (1:1)
- **架构**: EarthFormer原始配置
- **学习率**: 0.001 + cosine scheduler

### 重庆 1h→1h
- **输入**: 10帧 (60分钟，6分钟/帧)
- **输出**: 10帧 (60分钟)
- **Loss**: MAE + MSE (1:1)
- **架构**: 与SEVIR相同
- **数据路径**: `C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple`

### 重庆 3h→3h
- **输入**: 30帧 (180分钟)
- **输出**: 30帧 (180分钟)
- **Loss**: MAE + MSE (1:1)
- **架构**: 与SEVIR相同
- **Batch size**: 4 (更长序列)

## 重要注意事项

1. **不修改架构**: 实验使用EarthFormer原始架构，不修改Flash Attention、CNN等
2. **Loss函数**: 所有实验使用 MAE + MSE (1:1)
3. **数据范围**: SEVIR使用2017年6-9月数据，与论文一致
4. **评估指标**: CSI @ 16/74/133 dBZ

## 输出目录

```
outputs/
├── baseline_sevir_20frame_20260316_123456/
│   ├── config.yaml
│   ├── checkpoints/
│   └── logs/
├── baseline_chongqing_1h_to_1h_20260316_123456/
│   ├── config.yaml
│   └── ...
└── baseline_chongqing_3h_to_3h_20260316_123456/
    ├── config.yaml
    └── ...
```

## RTX 5070 优化

- batch_size: 1-8 (根据序列长度调整)
- bf16 mixed precision
- gradient_clip_val: 1.0
- early_stopping patience: 20
