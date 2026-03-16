# Baseline Experiments 使用指南

## 数据集支持

### 1. SEVIR 数据集 (美国雷达)
- 帧间隔: 5分钟
- 格式: SEVIR 标准格式
- 覆盖: 美国东南部
- 配置: `cfg_sevir_baseline_*.yaml`

### 2. 重庆雷达数据
- 帧间隔: 6分钟
- 格式: `day_simple_YYYYMMDD.npy` (240, H, W)
- 覆盖: 重庆地区 (山地地形)
- 原始尺寸: 约 500x500 (根据雷达PPI扫描)
- 预处理: resize 到目标尺寸
- 配置: `cfg_chongqing_baseline_*.yaml`

### 图像尺寸说明

**检查实际数据尺寸**:
```bash
python check_chongqing_data.py --data_dir /path/to/chongqing/data
```

**修改配置尺寸** (在 `cfg_chongqing_baseline_*.yaml`):
```yaml
dataset:
  img_height: 384   # 可选: 256, 384, 512
  img_width: 384

model:
  input_shape: [24, 384, 384, 1]
  target_shape: [24, 384, 384, 1]
```

**自动尺寸检测**: DataModule 支持自动检测数据尺寸

## 运行实验

### SEVIR 数据实验
```bash
# 高优先级: 2h→2h
python train_baseline_experiments.py --exp 2h_to_2h

# 高优先级: 4h→4h
python train_baseline_experiments.py --exp 4h_to_4h

# 所有高优先级实验
python train_baseline_experiments.py --priority high
```

### 重庆数据实验
```bash
# 重庆 2h→2h
python train_baseline_experiments.py --exp chongqing_2h_to_2h
```

### 快速测试
```bash
# 仅1 epoch测试
python train_baseline_experiments.py --exp 2h_to_2h --test_run
```

## 数据预处理

### 重庆数据预处理
如果还没有预处理重庆数据，运行:
```bash
cd preprocess
python chongqing_to_vil_gpu_daily_240_simple.py \
    --input_dir /path/to/chongqing/radar \
    --output_dir /path/to/output
```

输出: `day_simple_YYYYMMDD.npy` 文件

## 实验配置

### 输入-输出帧配置

| 实验 | 输入帧 | 输出帧 | 比例 | 说明 |
|------|--------|--------|------|------|
| 2h_to_2h | 24 | 24 | 1:1 | 标准设置 |
| 4h_to_4h | 48 | 48 | 1:1 | 可预测性上限 |
| 2h_to_3h | 24 | 36 | 1:1.5 | 输出>输入 |
| 4h_to_3h | 48 | 36 | 4:3 | 输入>输出 |

### 自适应学习率
- LRFinder: 自动寻找最优初始LR
- ReduceLROnPlateau: 验证loss不降时自动降低LR
- AutoStop: 综合早停 (loss + LR + gradient norm)

## 输出目录
```
outputs/
├── baseline_2h_to_2h_20260315_123456/
│   ├── config.yaml
│   ├── checkpoints/
│   └── logs/
└── baseline_chongqing_2h_to_2h_20260315_123456/
    ├── config.yaml
    └── ...
```

## 注意事项

1. **RTX 5070 8GB**: 已针对显存优化
   - batch_size: 1-4
   - gradient_accumulation: 4
   - bf16 mixed precision

2. **重庆数据路径**: 修改配置文件中的 `data_dir`

3. **时间对应**:
   - SEVIR: 24帧 = 2小时 (5分钟/帧)
   - 重庆: 24帧 ≈ 2.4小时 (6分钟/帧)
