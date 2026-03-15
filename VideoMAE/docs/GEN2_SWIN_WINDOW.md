# 第二代: Swin/局部窗口注意力 → 隐空间映射

> **核心思想**: 将 ViT 的全局注意力替换为**层级化局部窗口注意力**，通过 Shifted Window 实现跨窗口信息流动，
> 大幅降低计算量 (O(N²) → O(N·w²))，同时保持强表征能力。

---

## 一、代表论文

| 论文 | 会议 | 核心创新 | 与我们的关系 |
|------|------|---------|------------|
| **Swin Transformer** (Liu et al., 2021) | ICCV 2021 Best Paper | Shifted Window 多尺度 ViT | 奠基性工作，EarthFormer 的 Cuboid Attention 借鉴于此 |
| **Swin Transformer V2** (Liu et al., 2022) | CVPR 2022 | Log-CPB 相对位置编码 + 余弦注意力 → 更大模型/更高分辨率 | 位置编码可参考 |
| **Video Swin Transformer** (Liu et al., 2022) | CVPR 2022 | 3D Shifted Window → 时空局部注意力 | 直接用于视频/雷达时序 |
| **SimMIM** (Xie et al., 2022) | CVPR 2022 | Swin + 随机掩码预训练 (比 MAE 更简单的 decode) | **Swin 做 MAE 的最佳参考** |
| **SwinMAE** (未正式发表, 社区实现) | — | Swin Encoder + 轻量 Decoder 做 masked reconstruction | 社区验证 Swin+MAE 可行 |

## 二、参考 GitHub 仓库

| 仓库 | Stars | 说明 | 推荐度 |
|------|-------|------|--------|
| [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | 12k+ | 官方实现，PyTorch，代码清晰 | ⭐⭐⭐⭐⭐ |
| [SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) | 1k+ | 视频版 Swin，3D Window Attention | ⭐⭐⭐⭐ |
| [microsoft/SimMIM](https://github.com/microsoft/SimMIM) | 800+ | Swin+掩码预训练，decode 只需 1 层线性 | ⭐⭐⭐⭐⭐ |
| [berniwal/swin-transformer-pytorch](https://github.com/berniwal/swin-transformer-pytorch) | 700+ | 极简第三方实现，单文件，容易改 | ⭐⭐⭐⭐ |
| [huggingface/transformers (SwinModel)](https://huggingface.co/docs/transformers/model_doc/swin) | — | HuggingFace 集成，可直接加载预训练权重 | ⭐⭐⭐ |

## 三、架构设计 (Fashion-MNIST 验证方案)

### 3.1 Swin-MAE 架构

```
Fashion-MNIST 28×28
       │
  [Patch Embed]  4×4 patch → 7×7 = 49 tokens, dim=96
       │
  ┌────▼────┐
  │ Stage 1 │  Swin Block ×2 (window=7, 全覆盖, 无需shift)
  │ 7×7×96  │  W-MSA → MLP → SW-MSA → MLP
  └────┬────┘
  [Patch Merge]  7×7 → 4×4 (padding+merge), dim=192
       │
  ┌────▼────┐
  │ Stage 2 │  Swin Block ×2 (window=4, shift=2)
  │ 4×4×192 │
  └────┬────┘
       │
  [平均池化] → CLS token (dim=192) ← 隐空间表示
       │
  ┌────▼────────┐
  │  Decoder     │ Linear 192→49×16 → unpatchify → 28×28
  └──────────────┘
```

### 3.2 关键设计选择

| 设计点 | 选择 | 理由 |
|--------|------|------|
| 窗口大小 | 7 (Stage1), 4 (Stage2) | 28÷4=7, 正好一个窗口覆盖全图 |
| Shift | 仅 Stage2 | Stage1 窗口=全图，shift 无意义 |
| Patch Merging | 2×2 → 1 (通道2×) | 标准 Swin 下采样 |
| 位置编码 | 相对位置偏置 (RPB) | Swin 标准做法 |
| Decoder | SimMIM 风格: 1层线性 | 验证 encoder 隐空间质量 |
| 掩码策略 | 随机 patch mask 75% | 和 MAE baseline 保持一致 |

### 3.3 与 MAE Baseline 的对比实验

| 对比维度 | MAE (ViT) | Swin-MAE | 预期 |
|----------|-----------|----------|------|
| Encoder | 全局注意力 O(N²) | 窗口注意力 O(N·w²) | Swin 更高效 |
| 多尺度 | 无 | 有 (2 stage) | Swin 捕获层级特征 |
| 隐空间维度 | 128 (CLS) | 192 (pool) | Swin 更丰富 |
| 参数量 | ~1.36M | ~1.5M | 相当 |
| 训练速度 | 42s/epoch | 预计 ~50s/epoch | MAE 稍快 |
| 重建质量 | baseline | **目标: ≥baseline** | |
| 隐空间分离度 | 1.55 | **目标: >1.55** | |

## 四、实现计划

```python
# 核心模块:
class WindowAttention(nn.Module):    # 窗口内注意力 + 相对位置偏置
class SwinBlock(nn.Module):          # W-MSA → FFN → SW-MSA → FFN
class PatchMerging(nn.Module):       # 2×2 合并下采样
class SwinEncoder(nn.Module):        # [Stage1(7×7) → Merge → Stage2(4×4)]
class SwinMAE(nn.Module):            # SwinEncoder + mask + SimMIM-Decoder
```

### 掩码策略

Swin 的掩码比 ViT-MAE 复杂，因为窗口注意力要求所有 token 都在固定网格位置:
- **SimMIM 做法**: 不删除 masked token（和 MAE 不同），而是替换为可学习的 `[MASK]` embedding。Encoder 处理所有 token (含 mask)，但只在 masked 位置计算 loss。
- **优点**: 无需改变窗口划分逻辑
- **缺点**: 不像 MAE 那样省计算 (所有 token 都过 attention)

### 与雷达的衔接

Swin 的局部窗口 → EarthFormer 的 Cuboid 本质上是同一思想:
- Swin: 2D 空间窗口 (H_w, W_w)
- EarthFormer Cuboid: 3D 时空窗口 (T_w, H_w, W_w)
- 我们的实验: 验证**窗口内注意力 + mask 预训练**的组合效果

## 五、关键创新点 (论文可写)

1. **Swin 窗口与物理阈值协同**: 窗口划分与 15dBZ 物理边界对齐
2. **层级隐空间**: Stage1 → 局部纹理特征, Stage2 → 全局结构特征
3. **SimMIM vs MAE 在气象数据上的对比**: 气象数据稀疏，哪种掩码策略更优？

---

## 六、参考文献

```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={ICCV},
  year={2021}
}

@inproceedings{xie2022simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhu and Dai, Qi and Hu, Han},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{liu2022video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={CVPR},
  year={2022}
}
```
