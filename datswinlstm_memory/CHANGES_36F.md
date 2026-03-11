# 36帧 Flash Attention 实验方案

## 概述

在 20帧基线 (`train_384.py`) 基础上，扩展到 **36帧**序列长度，并以 Flash Attention (SDPA) 版本作为基线。同时重新跑 exp7-12（Flash Attention + MoE 实验变体）。

## 新增文件

| 文件 | 用途 |
|------|------|
| `train_36f_baseline.py` | 36帧 Flash Attention 基线训练脚本（无 MoE） |
| `run_36f_queue.py` | 36帧实验队列（基线 + exp7-12 顺序执行） |
| `CHANGES_36F.md` | 本文档 |

## 核心修改对比

### 数据参数变化

| 参数 | 20帧基线 (`train_384.py`) | 36帧基线 (`train_36f_baseline.py`) |
|------|:---:|:---:|
| `seq_len` (总序列长度) | 20 | **36** |
| `in_len` (输入帧数) | 8 | **12** |
| `out_len` (预测帧数) | 12 | **24** |
| `memory_len` (长期记忆) | 24 | **36** |

### 模型参数（不变）

| 参数 | 值 |
|------|------|
| `embed_dim` | 64 |
| `depths` | [2, 2] |
| `heads` | [4, 4] |
| `memory_channel_size` | 256 |
| `patch_size` | 4 |
| `window_size` | 4 |
| `input_img_size` | 384×384 |

### Flash Attention 状态

| 脚本 | WindowAttention | Memory Attention | 
|------|:---:|:---:|
| `train_384.py` (20帧基线) | ✅ SDPA (默认) | ✅ SDPA (默认) |
| `train_36f_baseline.py` (36帧基线) | ✅ SDPA (默认) | ✅ SDPA (默认) |
| `train_experiment.py` exp1-6 | ❌ 被 `apply_experiment` 关闭 | ❌ 被关闭 |
| `train_experiment.py` exp7-12 | ✅ 被 `apply_experiment` 开启 | ✅ 被开启 |

> **注意**: `WindowAttention` 和 `Attention` 类的构造函数中 `use_flash=True` 是默认值。`experiment_factory.py` 的 `apply_experiment()` 会**总是**显式调用 `_inject_flash_attention(model, enable=config.use_flash)` 来统一控制，确保 exp1-6 关闭、exp7-12 开启。

## MotionEncoder2D 分块处理

36帧序列会触发 `MotionEncoder2D.forward` 的分块逻辑（`chunk_size=20`）：

```
36帧 → 分块处理:
  Chunk 1: 帧 [0..20] (21帧, 含1帧 overlap) → MS → 保留前20帧
  Chunk 2: 帧 [20..35] (16帧, 末块) → MS → 全部保留
  拼接: 20 + 16 = 36帧 ✓
```

**为什么需要分块？**
- `MotionSqueeze` 的 `flow_computation()` 计算相邻帧对 (i, i+1) 的光流
- 2D 卷积的中间激活占用大量显存（36帧一次性处理会 OOM）
- 分块后每次处理 ≤21 帧，显存和 20帧基线相当

**重叠策略保证精确等价：**
- 非末块多取 1 帧作为 overlap → 边界帧对 (19, 20) 被正确计算
- 丢弃非末块的末帧（MS padding 复制帧，非真实光流）
- `flow_refine_conv` 全是 2D 卷积，每帧独立，分块后结果与全量处理 **100% 一致**

## 实验队列

`run_36f_queue.py` 按顺序执行以下实验：

| # | 名称 | 特性 | 脚本 |
|---|------|------|------|
| 1 | `baseline_36f_flash` | 纯基线 + Flash Attention | `train_36f_baseline.py` |
| 2 | `exp7_moe_flash` | MoE + Flash | `train_experiment.py --exp exp7_moe_flash` |
| 3 | `exp8_swiglu_moe_flash` | SwiGLU-MoE + Flash | `train_experiment.py --exp exp8_swiglu_moe_flash` |
| 4 | `exp9_balanced_moe_flash` | Balanced-MoE + Flash | `train_experiment.py --exp exp9_balanced_moe_flash` |
| 5 | `exp10_moe_rope_flash` | MoE + RoPE + Flash | `train_experiment.py --exp exp10_moe_rope_flash` |
| 6 | `exp11_swiglu_moe_rope_flash` | SwiGLU-MoE + RoPE + Flash | `train_experiment.py --exp exp11_swiglu_moe_rope_flash` |
| 7 | `exp12_balanced_moe_rope_flash` | Balanced-MoE + RoPE + Flash | `train_experiment.py --exp exp12_balanced_moe_rope_flash` |

### 36帧实验 CLI 参数

```bash
# 基线
python -u train_36f_baseline.py --epochs 200 --batch_size 1 --num_workers 0

# exp7-12 (36帧版本)
python -u train_experiment.py \
  --exp exp7_moe_flash \
  --seq_len 36 --input_frames 12 --output_frames 24 \
  --epochs 100 --batch_size 1 --num_workers 0 --no_amp \
  --checkpoint_dir ./checkpoints_36f
```

## 检查点保存位置

| 实验 | 路径 |
|------|------|
| 20帧基线 | `./checkpoints/baseline_20f/` |
| 36帧基线 | `./checkpoints/baseline_36f_flash/` |
| 36帧 exp7-12 | `./checkpoints_36f/exp7_moe_flash/` 等 |
| 20帧 exp1-10 | `./checkpoints/exp1_moe/` 等（之前的实验） |

## 监控命令

```powershell
# 查看 20帧基线训练进度
Get-Content .\checkpoints\_runlogs\baseline_20f.log -Wait

# 查看 36帧训练队列进度
Get-Content .\checkpoints\_runlogs\queue_36f.log -Wait

# GPU 状态
nvidia-smi -l 5
```
