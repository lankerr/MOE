# CSI 微调使用指南

## 📋 目录

1. [流程概述](#流程概述)
2. [准备工作](#准备工作)
3. [微调步骤](#微调步骤)
4. [使用示例](#使用示例)
5. [参数说明](#参数说明)
6. [常见问题](#常见问题)

---

## 流程概述

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 预训练 (Pre-training)                                │
│   目标: 使用 L1+MSE 训练到收敛                              │
│   Epochs: 50-100                                           │
│   输出: checkpoints/expXX/best_model.pt                    │
├─────────────────────────────────────────────────────────────┤
│ 阶段2: CSI 微调 (SFT)                                       │
│   目标: 优化 CSI 指标                                       │
│   输入: 阶段1 的 best_model.pt                             │
│   Epochs: 10-20                                            │
│   输出: checkpoints/sft/expXX_sft_multi/best_model.pt      │
├─────────────────────────────────────────────────────────────┤
│ 阶段3: 评估对比                                             │
│   对比预训练 vs 微调后的 CSI/HSS/POD/FAR 指标              │
└─────────────────────────────────────────────────────────────┘
```

---

## 准备工作

### 检查预训练模型状态

```powershell
cd c:\Users\97290\Desktop\MOE\datswinlstm_memory

# 检查 exp7-12 的训练状态
for exp in exp7_moe_flash exp8_swiglu_moe_flash exp9_balanced_moe_flash exp10_moe_rope_flash exp11_swiglu_moe_rope_flash exp12_balanced_moe_rope_flash; do
    echo "=== $exp ==="
    cat "checkpoints/$exp/training_log.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'Epochs: {len(d)}, Best Val: {min([e[\"val_loss\"] for e in d]):.4f}')"
done
```

**预期结果**: 所有模型应该训练 50+ epochs 且验证损失收敛 (< 0.04)

### 如果模型未收敛

继续训练直到 50-100 epochs：

```powershell
# 使用原始训练脚本继续训练
conda run -n rtx5070_cu128 python train_experiment_fast.py \\
    --exp_name exp11_swiglu_moe_rope_flash \\
    --epochs 100 \\
    --resume
```

---

## 微调步骤

### Step 1: 确定基础模型

根据验证损失和训练稳定性选择最佳模型：

| 模型 | Val Loss | 推荐度 | 说明 |
|------|----------|--------|------|
| exp11_swiglu_moe_rope_flash | ~0.043 | ⭐⭐⭐ | 最佳，SwiGLU+RoPE+Flash |
| exp12_balanced_moe_rope_flash | ~0.044 | ⭐⭐⭐ | 均衡，有负载均衡 |
| exp8_swiglu_moe_flash | ~0.044 | ⭐⭐ | SwiGLU+Flash |
| exp7_moe_flash | ~0.044 | ⭐⭐ | 基础 MoE+Flash |

**推荐**: `exp11_swiglu_moe_rope_flash` 或 `exp12_balanced_moe_rope_flash`

### Step 2: 运行 CSI 微调

#### 2.1 基础用法 (20帧模型)

```powershell
cd c:\Users\97290\Desktop\MOE\datswinlstm_memory

# 对 exp11 进行 CSI 微调
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --preset 20frame \\
    --epochs 20 \\
    --lr 1e-5 \\
    --csi_type multi
```

#### 2.2 36帧模型微调

```powershell
# 对 36帧模型进行 CSI 微调
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --preset 36frame \\
    --epochs 20 \\
    --lr 1e-5 \\
    --csi_type multi
```

#### 2.3 自定义参数

```powershell
# 自定义微调参数
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --preset 20frame \\
    --epochs 30 \\
    --lr 5e-6 \\
    --warmup_epochs 5 \\
    --csi_start_epoch 8 \\
    --final_csi_weight 0.9 \\
    --csi_type fuzzy \\
    --temperature_init 0.4 \\
    --temperature_final 0.03
```

### Step 3: 监控训练

训练日志保存在 `checkpoints/sft/{base_model}_sft_{csi_type}/training_log.json`

```powershell
# 实时监控训练
watch -n 10 "cat checkpoints/sft/exp11_swiglu_moe_rope_flash_sft_multi/training_log.json | tail -20"
```

关注以下指标：
- `val.total_loss`: 验证总损失（应该持续下降）
- `val.CSI_0.57`: CSI@30dBZ（主要优化目标）
- `base_weight` / `csi_weight`: 损失权重变化

### Step 4: 评估对比

```powershell
# 对比微调前后的 CSI 指标
conda run -n rtx5070_cu128 python evaluate_metrics.py
```

---

## 使用示例

### 示例 1: 快速开始 (推荐新手)

```powershell
# 1. 确认 exp11 已训练 50+ epochs
cat checkpoints/exp11_swiglu_moe_rope_flash/training_log.json | python -c "import sys,json; print(f'Epochs: {len(json.load(sys.stdin))}')"

# 2. 运行 CSI 微调 (使用默认参数)
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --preset 20frame

# 3. 等待完成 (~2-4 小时，取决于 GPU)
# 4. 检查结果
cat checkpoints/sft/exp11_swiglu_moe_rope_flash_sft_multi/training_log.json
```

### 示例 2: 批量微调多个模型

```powershell
# 创建批量微调脚本
cat > batch_sft.sh << 'EOF'
#!/bin/bash
models=(
    "exp11_swiglu_moe_rope_flash"
    "exp12_balanced_moe_rope_flash"
    "exp8_swiglu_moe_flash"
)

for model in "${models[@]}"; do
    echo "微调: $model"
    conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
        --base_model $model \\
        --preset 20frame \\
        --epochs 20 \\
        --lr 1e-5 \\
        --csi_type multi
done
EOF

chmod +x batch_sft.sh
./batch_sft.sh
```

### 示例 3: 对比不同 CSI 损失类型

```powershell
# 测试 single CSI
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --csi_type single

# 测试 multi CSI (推荐)
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --csi_type multi

# 测试 fuzzy CSI
conda run -n rtx5070_cu128 python sft/train_csi_sft.py \\
    --base_model exp11_swiglu_moe_rope_flash \\
    --csi_type fuzzy
```

---

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base_model` | 基础模型名称 | `exp11_swiglu_moe_rope_flash` |

### 预设配置

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--preset` | 帧数预设 | `20frame`, `36frame` | `20frame` |

**预设详情**:
- `20frame`: seq_len=20, input=8, output=12
- `36frame`: seq_len=36, input=13, output=23

### 训练参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--epochs` | 微调 epochs | 20 | 10-30 |
| `--lr` | 学习率 | 1e-5 | 5e-6 - 2e-5 |
| `--warmup_epochs` | 预热 epochs | 3 | 2-5 |
| `--accumulation_steps` | 梯度累积步数 | 4 | 2-8 |

### CSI 损失参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--csi_type` | CSI 类型 | `multi` | `multi` |
| `--csi_start_epoch` | 开始引入 CSI 的 epoch | 5 | 3-8 |
| `--final_csi_weight` | 最终 CSI 权重 | 0.8 | 0.7-0.9 |
| `--temperature_init` | 初始温度 | 0.3 | 0.2-0.5 |
| `--temperature_final` | 最终温度 | 0.05 | 0.03-0.1 |

**CSI 类型说明**:
- `single`: 单阈值 (30dBZ)，简单直接
- `multi`: 多阈值 (16/30/40/50 dBZ)，**推荐**
- `fuzzy`: 模糊 CSI，适合边界模糊情况

---

## 常见问题

### Q1: 微调需要多少时间？

**A**: 取决于配置和 GPU：
- RTX 5070 Laptop: ~2-3 小时/20 epochs
- RTX 4090: ~1-1.5 小时/20 epochs

### Q2: 如何判断微调是否成功？

**A**: 检查以下指标：
1. ✅ 验证 CSI@30dBZ 提升 > 2%
2. ✅ 验证总损失稳定或下降
3. ❌ 如果训练损失下降但验证 CSI 不变，可能是过拟合

### Q3: 微调后 CSI 反而下降了？

**A**: 可能原因：
1. 学习率过大 → 降低到 5e-6
2. CSI 权重增加太快 → 延后 `--csi_start_epoch`
3. 温度过小导致梯度消失 → 增加 `--temperature_init` 到 0.4

### Q4: 可以用 10 epochs 的模型微调吗？

**A**: 可以但效果不佳。建议：
- 最低: 20 epochs 的预训练模型
- 推荐: 50-100 epochs 的收敛模型
- 最佳: 验证损失不再下降的模型

### Q5: 20帧和36帧可以混合微调吗？

**A**: 不可以。确保：
- 预训练和微调使用相同的 `--preset`
- 模型架构配置一致

### Q6: 如何选择最佳微调模型？

**A**: 综合考虑：
1. **主指标**: CSI@30dBZ (最重要)
2. **辅助指标**: HSS@30dBZ, POD@30dBZ
3. **稳定性**: 训练/验证曲线平滑

---

## 输出结果

### 文件结构

```
checkpoints/sft/
└── exp11_swiglu_moe_rope_flash_sft_multi/
    ├── best_model.pt              # 最佳模型 (按验证损失)
    ├── epoch_5.pt
    ├── epoch_10.pt
    ├── epoch_15.pt
    ├── epoch_20.pt
    ├── latest_model.pt            # 最新检查点
    ├── training_log.json          # 训练日志
    └── sft_config.json            # 微调配置
```

### 训练日志示例

```json
{
  "epoch": 15,
  "lr": 1e-5,
  "base_weight": 0.2,
  "csi_weight": 0.8,
  "temperature": 0.08,
  "train": {
    "total_loss": 0.0412,
    "base_loss": 0.0398,
    "csi_loss": 0.0465
  },
  "val": {
    "total_loss": 0.0401,
    "base_loss": 0.0389,
    "csi_loss": 0.0442,
    "CSI_0.37": 0.5234,   // CSI@16dBZ
    "CSI_0.57": 0.4456,   // CSI@30dBZ ← 主要目标
    "CSI_0.71": 0.3345,   // CSI@40dBZ
    "CSI_0.86": 0.2012    // CSI@50dBZ
  },
  "time": 520
}
```

### 预期改进

| 指标 | 预训练 (exp11) | 微调后 | 改进 |
|------|----------------|--------|------|
| Val Loss | 0.0428 | 0.0401 | ▼ 6% |
| CSI@16dBZ | ~0.48 | ~0.52 | ▲ 8% |
| **CSI@30dBZ** | ~0.40 | **~0.45** | **▲ 12%** |
| CSI@40dBZ | ~0.30 | ~0.33 | ▲ 10% |
| CSI@50dBZ | ~0.18 | ~0.20 | ▲ 11% |

---

## 下一步

微调完成后，可以：

1. **部署最佳模型**: 使用 `best_model.pt` 进行推理
2. **对比实验**: 评估预训练 vs 微调 vs 其他方法
3. **论文实验**: 记录消融实验结果
4. **生产部署**: 集成到降水预测系统

---

## 相关文件

- `sft/differentiable_csi.py` - 可微 CSI 损失实现
- `sft/train_csi_sft.py` - 微调训练脚本
- `sft/README.md` - 技术细节文档
- `evaluate_metrics.py` - 评估脚本
