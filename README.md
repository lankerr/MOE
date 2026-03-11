# MOE - Mixture of Experts 增强的时空降水预测实验框架

> 基于 MoE (Mixture of Experts) + Flash Attention + RoPE 的时空降水临近预报模型对比实验

## 项目概述

本项目在 **SEVIR** (Storm EVent ImagRy) 数据集上，对两个时空预测模型进行 MoE 增强实验对比：

| 模型 | 架构 | 实验数量 | 帧配置 |
|------|------|---------|--------|
| **DATSwinLSTM-Memory** | 可变形注意力 + SwinLSTM + 记忆库 | 12 个实验 (Exp1-12) | 20f / 36f / 49f |
| **Earthformer** | Cuboid Transformer | 8 个实验 (baseline + Exp1-6 + Exp1.5) | 20f / 49f |

### 核心技术

- **MoE (Mixture of Experts)**: Top-2 路由，4 专家，支持 GELU / SwiGLU 激活
- **Flash Attention**: 基于 PyTorch SDPA 的高效注意力
- **Temporal RoPE**: 时序旋转位置编码
- **辅助损失**: Load Balancing Loss + Orthogonalization Loss

---

## 目录结构

```
MOE/
├── datswinlstm_memory/          # DATSwinLSTM-Memory 模型
│   ├── models/                  # 模型定义
│   │   ├── DATSwinLSTM_D_Memory.py   # 主模型
│   │   └── dat_blocks.py              # 可变形注意力块
│   ├── modules/                 # 共享模块 (两个模型共用)
│   │   ├── moe_layer.py         # ⭐ MoE 核心: MoELayer, TopKRouter, Experts
│   │   ├── swiglu.py            # SwiGLU FFN
│   │   └── temporal_rope.py     # Temporal RoPE 1D/2D
│   ├── experiments/
│   │   └── experiment_factory.py  # 12 个实验配置 + 注入逻辑
│   ├── train_experiment_fast.py   # 加速训练脚本
│   ├── run_all.py                 # 批量运行器
│   ├── config.py                  # 数据路径配置
│   ├── sevir_torch_wrap.py        # SEVIR 数据加载
│   ├── EXPERIMENTS_DETAIL.md      # 12 个实验详细说明
│   └── *.md                       # 各类文档
│
├── earth-forecasting-transformer/ # Earthformer 模型
│   ├── src/earthformer/           # Earthformer 源码
│   │   └── cuboid_transformer/
│   │       └── cuboid_transformer.py  # Cuboid Transformer 核心
│   ├── scripts/cuboid_transformer/sevir/
│   │   ├── experiment_factory_earthformer.py  # ⭐ Earthformer 实验工厂
│   │   ├── train_experiment_earthformer.py    # PL 训练脚本
│   │   ├── run_all_earthformer_full.py        # 批量运行器
│   │   ├── cfg_sevir_20frame.yaml             # 20帧配置
│   │   ├── cfg_sevir_49frame.yaml             # 49帧配置
│   │   └── debug/
│   │       ├── CODE_MIGRATION_AUDIT.md        # ⭐ 迁移审计报告
│   │       ├── MOE_NAN_ROOT_CAUSE.md          # NaN 根因分析
│   │       └── nan_debug_train.py             # NaN 调试工具
│   └── MIGRATION_GUIDE.md
│
├── nanoGPT/                       # nanoGPT + MoE 实验 (NLP 基线对比)
│   ├── model_moe.py               # MoE 增强的 GPT
│   ├── train_moe.py               # MoE 训练脚本
│   └── run_all_experiments.py     # 5 组对比实验
│
├── nanoMoE/                       # nanoMoE 参考实现
│
├── msrnn/                         # MS-RNN 基线模型 (ConvLSTM/MIM)
│
└── run_earthformer_20f.py         # 快速启动 Earthformer 20f 训练
```

---

## 实验设计

### DATSwinLSTM-Memory: 12 个实验

| 实验 | MoE | SwiGLU | Balance | Ortho | RoPE | Flash |
|------|:---:|:------:|:-------:|:-----:|:----:|:-----:|
| Exp1 | ✓ | | | | | |
| Exp2 | ✓ | ✓ | | | | |
| Exp3 | ✓ | ✓ | ✓ | ✓ | | |
| Exp4 | ✓ | | | | ✓ | |
| Exp5 | ✓ | ✓ | | | ✓ | |
| Exp6 | ✓ | ✓ | ✓ | ✓ | ✓ | |
| Exp7 | ✓ | | | | | ✓ |
| Exp8 | ✓ | ✓ | | | | ✓ |
| Exp9 | ✓ | ✓ | ✓ | ✓ | | ✓ |
| Exp10 | ✓ | | | | ✓ | ✓ |
| Exp11 | ✓ | ✓ | | | ✓ | ✓ |
| Exp12 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Earthformer: 8 个实验

基于 DATSwinLSTM Exp7-12 迁移，全部启用 Flash Attention：

| 实验 | 对应 DATS | 特性 |
|------|-----------|------|
| baseline | — | 原始 Earthformer |
| exp1_moe_flash | Exp7 | MoE(GELU) + Flash |
| exp1.5_moe_balanced_flash | *新增* | MoE(GELU) + Balance + Flash |
| exp2_swiglu_moe_flash | Exp8 | SwiGLU-MoE + Flash |
| exp3_balanced_moe_flash | Exp9 | SwiGLU-MoE + Balance + Ortho + Flash |
| exp4_moe_rope_flash | Exp10 | MoE(GELU) + RoPE + Flash |
| exp5_swiglu_moe_rope_flash | Exp11 | SwiGLU-MoE + RoPE + Flash |
| exp6_balanced_moe_rope_flash | Exp12 | 全配置 |

---

## 关键技术细节

### MoE 层架构

```
输入 x [B, seq_len, dim]
    ↓
TopKRouter: Linear(dim, num_experts) → Top-2 选择
    ↓
分发到 Expert_i: Linear(dim, hidden) → Act → Linear(hidden, dim)
    ↓
加权合并 → 输出 [B, seq_len, dim]
    ↓
辅助损失: balance_loss + ortho_loss (仅训练时)
```

- **StandardExpert**: `fc1 → GELU → drop → fc2 → drop`
- **SwiGLUExpert**: `fc_gate → SiLU ⊙ fc_up → drop → fc_down → drop` (带 autocast 防 NaN)

### Earthformer MoEFFNWrapper

Earthformer 的 `PositionwiseFFN` 内置了 LayerNorm + 残差连接，直接替换会丢失这些关键组件导致 NaN。
解决方案是 `MoEFFNWrapper`：

```python
class MoEFFNWrapper(nn.Module):
    def forward(self, data):
        residual = data               # 保留残差
        data = self.layer_norm(data)   # 保留 LayerNorm
        out = self.moe(data)           # MoE 替代 fc1+fc2
        out = self.dropout(out)        # 保留 dropout
        out = out + residual           # 残差连接
        return out
```

详见 [MOE_NAN_ROOT_CAUSE.md](earth-forecasting-transformer/scripts/cuboid_transformer/sevir/debug/MOE_NAN_ROOT_CAUSE.md)

### Flash Attention (SDPA)

- DATSwinLSTM: 内置 SDPA 路径 (`WindowAttention.use_flash`)
- Earthformer: Monkey-patch `CuboidSelfAttentionLayer.forward`，支持 relative position bias + shift mask

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| PyTorch | 2.x (支持 SDPA) |
| CUDA | 12.8+ |
| GPU | NVIDIA RTX 5070 Laptop (8GB VRAM) |
| RAM | 16GB |
| 精度 | bf16-mixed |

```bash
# 创建环境
conda create -n rtx5070_cu128 python=3.11
conda activate rtx5070_cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pytorch-lightning omegaconf einops torchmetrics
cd earth-forecasting-transformer && pip install -e .
```

---

## 快速开始

### DATSwinLSTM-Memory

```bash
cd datswinlstm_memory

# 单个实验
python -u train_experiment_fast.py --exp exp7_moe_flash --epochs 100

# 全部 12 个实验
python -u run_all.py
```

### Earthformer

```bash
cd earth-forecasting-transformer/scripts/cuboid_transformer/sevir

# 单个实验
python -u train_experiment_earthformer.py --cfg cfg_sevir_20frame.yaml --exp exp1_moe_flash --epochs 10

# 全部实验 (20f + 49f)
python -u run_all_earthformer_full.py
```

---

## 数据集

### SEVIR (Storm EVent ImagRy)

- 384×384 像素，5 分钟/帧，VIL 雷达数据
- 下载: `aws s3 sync --no-sign-request s3://sevir/data/vil ./sevir_data/vil`
- 约 200GB (VIL only)

---

## 评估指标

| 指标 | 说明 |
|------|------|
| **CSI** (Critical Success Index) | 主要指标，阈值: 16/74/133/160/181/219 |
| **POD** (Probability of Detection) | 检测率 |
| **SUCR** (Success Ratio) | 成功率 |
| **MSE / MAE** | 像素级误差 |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [EXPERIMENTS_DETAIL.md](datswinlstm_memory/EXPERIMENTS_DETAIL.md) | DATSwinLSTM 12 个实验详细说明 |
| [CODE_MIGRATION_AUDIT.md](earth-forecasting-transformer/scripts/cuboid_transformer/sevir/debug/CODE_MIGRATION_AUDIT.md) | Earthformer 迁移审计报告 |
| [MOE_NAN_ROOT_CAUSE.md](earth-forecasting-transformer/scripts/cuboid_transformer/sevir/debug/MOE_NAN_ROOT_CAUSE.md) | MoE NaN 根因分析 |
| [MIGRATION_GUIDE.md](earth-forecasting-transformer/MIGRATION_GUIDE.md) | Earthformer 迁移指南 |
| [FRAME_49_DESIGN.md](datswinlstm_memory/FRAME_49_DESIGN.md) | 49 帧设计文档 |

---

## License

- DATSwinLSTM-Memory: MIT License
- Earthformer: Apache 2.0 License (NVIDIA)
- nanoGPT / nanoMoE: MIT License
