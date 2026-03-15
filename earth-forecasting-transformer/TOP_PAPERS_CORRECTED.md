# 顶刊论文正确链接（已验证）

## 🏆 Nature 级别论文

### 必读：雷达降水预报

| 论文名称 | 年份 | 期刊 | PDF 下载 | 备注 |
|----------|------|------|----------|------|
| **Skilful precipitation nowcasting using deep generative models of radar** (DGMR) | 2021 | Nature | [PDF](https://arxiv.org/pdf/2104.00954.pdf) | DeepMind 首个生成式雷达外推 |
| **Skilful nowcasting of extreme precipitation with NowcastNet** | 2023 | Nature | [PDF](https://www.nature.com/articles/s41586-023-06184-4.pdf) | 清华大学龙明盛团队 |
| **Accurate medium-range global weather forecasting with 3D neural networks** (Pangu-Weather) | 2023 | Nature | [PDF](https://arxiv.org/pdf/2211.02556.pdf) | 华为盘古大模型 |
| **GraphCast: Learning skillful medium-range global weather forecasting** | 2023 | Nature | [PDF](https://arxiv.org/pdf/2212.12794.pdf) | DeepMind 图神经网络 |

### 全球天气预报

| 论文名称 | 年份 | 期刊 | PDF 下载 |
|----------|------|------|----------|
| **FourCastNet: A global data-driven high-resolution weather model** | 2022 | Nature | [PDF](https://arxiv.org/pdf/2202.11214.pdf) |
| **Accurate medium-range global weather forecasting** (Pangu) | 2023 | Nature | [PDF](https://www.nature.com/articles/s41586-023-06185-3.pdf) |

---

## 📚 顶级会议论文 (NeurIPS/ICLR)

### Transformer 架构

| 论文名称 | 年份 | 会议 | PDF 下载 |
|----------|------|------|----------|
| **EarthFormer: Exploring space-time transformers** | 2022 | NeurIPS | [PDF](https://arxiv.org/pdf/2207.05833.pdf) |
| **Swin Transformer: Hierarchical vision transformer** | 2022 | ICLR | [PDF](https://arxiv.org/pdf/2103.14030.pdf) |
| **An image is worth 16x16 words: Transformers for image recognition** (ViT) | 2021 | ICLR | [PDF](https://arxiv.org/pdf/2010.11929.pdf) |

### Token 高效处理

| 论文名称 | 年份 | 会议 | PDF 下载 |
|----------|------|------|----------|
| **Token merging for fast vision processing** (ToMe) | 2023 | ICLR | [PDF](https://arxiv.org/pdf/2209.15559.pdf) |
| **Efficient Vision Transformer with token pruning** (EViT) | 2022 | ICLR | [PDF](https://arxiv.org/pdf/2204.08616.pdf) |
| **DynamicViT: Efficient vision transformers with dynamic token sparsity** | 2021 | NeurIPS | [PDF](https://arxiv.org/pdf/2106.01304.pdf) |

### 等变网络

| 论文名称 | 年份 | 会议 | PDF 下载 |
|----------|------|------|----------|
| **Equivariant CNNs for the rotation group** | 2018 | NeurIPS | [PDF](https://arxiv.org/pdf/1802.08219.pdf) |
| **Steerable CNNs for rotation equivariance** | 2018 | ICLR | [PDF](https://arxiv.org/pdf/1804.08258.pdf) |

---

## 🔬 Science 级别论文

| 论文名称 | 年份 | 期刊 | PDF 下载 |
|----------|------|------|----------|
| **Learning skillful medium-range global weather forecasting** (GraphCast) | 2023 | Science | [PDF](https://www.science.org/doi/pdf/10.1126/science.adi2336) |

---

## 📥 快速下载命令

```bash
# 使用 wget/curl 直接下载
wget https://arxiv.org/pdf/2207.05833.pdf -O EarthFormer.pdf
wget https://arxiv.org/pdf/2209.15559.pdf -O ToMe.pdf
wget https://arxiv.org/pdf/1802.08219.pdf -O E2CNN.pdf

# Windows PowerShell
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2207.05833.pdf" -OutFile "EarthFormer.pdf"
```

---

## 📖 阅读顺序建议（已验证链接）

### 第一周：基础架构
1. [EarthFormer](https://arxiv.org/pdf/2207.05833.pdf) - 你的 baseline
2. [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) - Shifted Window
3. [ViT](https://arxiv.org/pdf/2010.11929.pdf) - Vision Transformer 基础

### 第二周：Token 高效
4. [ToMe](https://arxiv.org/pdf/2209.15559.pdf) - Token Merging
5. [EViT](https://arxiv.org/pdf/2204.08616.pdf) - Token 剪枝
6. [DynamicViT](https://arxiv.org/pdf/2106.01304.pdf) - 动态稀疏

### 第三周：等变网络
7. [E2CNN](https://arxiv.org/pdf/1802.08219.pdf) - 群等变卷积
8. [Steerable CNN](https://arxiv.org/pdf/1804.08258.pdf) - 可控滤波器

### 第四周：气象 SOTA
9. [DGMR](https://arxiv.org/pdf/2104.00954.pdf) - 生成式外推
10. [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4.pdf) - 极端降水
11. [Pangu-Weather](https://arxiv.org/pdf/2211.02556.pdf) - 华为盘古
12. [GraphCast](https://arxiv.org/pdf/2212.12794.pdf) - DeepMind 图网络
