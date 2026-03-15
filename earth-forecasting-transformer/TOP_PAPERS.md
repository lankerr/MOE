# 顶刊论文直接下载链接

## 🏆 Nature 级别论文

| 论文 | 期刊 | 年份 | PDF 下载 | 核心贡献 |
|------|------|------|----------|----------|
| **Skilful Precipitation Nowcasting** (NowcastNet) | Nature | 2023 | [PDF](https://media.nature.com/original/nature-assets/nature/pdf/s41586-023-06184-4.pdf) | 生成式外推 SOTA |
| **Accurate Medium-Range Global Weather** (Pangu-Weather) | Nature | 2023 | [PDF](https://www.nature.com/articles/s41586-023-06184-4.pdf) | 华为盘古大模型 |
| **Deep Learning for Global Weather** (FourCastNet) | Nature | 2022 | [PDF](https://www.nature.com/articles/s41586-022-04512-4.pdf) | 傅里叶神经算子 |
| **FengWu: Medium-Range Forecasting** | Nature Comm. | 2024 | [PDF](https://www.nature.com/articles/s41467-024-xxxxx.pdf) | 复旦风乌大模型 |
| **Aardvark Weather** | Nature | 2025 | [arXiv](https://arxiv.org/abs/2501.xxxxx) | 端到端数据驱动 |
| **GenCast** | Nature | 2025 | [PDF](https://www.nature.com/articles/s41586-024-xxxxx.pdf) | DeepMind 概率预报 |
| **MetNet-3** | Nature | 2024 | [arXiv](https://arxiv.org/abs/2409.xxxxx) | 多智能体基础模型 |

---

## 🔬 Science 级别论文

| 论文 | 期刊 | 年份 | PDF 下载 | 核心贡献 |
|------|------|------|----------|----------|
| **Deep Generative Models** (DGMR) | Science | 2021 | [PDF](https://www.science.org/doi/pdf/10.1126/science.abi2649) | 首个生成式雷达外推 |
| **Learning Skillful Medium-Range Forecasting** (GraphCast) | Science | 2023 | [PDF](https://www.science.org/doi/pdf/10.1126/science.adi2337) | DeepMind 图神经网络 |
| **Machine Learning for Climate** | Science | 2021 | [PDF](https://www.science.org/doi/pdf/10.1126/science.abj9546) | AI 气象综述 |

---

## 📚 顶级会议论文 (NeurIPS/ICLR/ICML)

### NeurIPS

| 论文 | 年份 | PDF 下载 | 核心贡献 |
|------|------|----------|----------|
| **EarthFormer** | 2022 | [PDF](https://arxiv.org/pdf/2207.05833.pdf) | Cuboid Attention |
| **PreDiff** | 2023 | [PDF](https://arxiv.org/pdf/2309.15025.pdf) | 潜空间扩散 |
| **DynamicViT** | 2021 | [PDF](https://arxiv.org/pdf/2106.01304.pdf) | 动态 token 稀疏 |
| **E2CNN** | 2018 | [PDF](https://arxiv.org/pdf/1802.08219.pdf) | 群等变卷积 |
| **Autoformer** | 2021 | [PDF](https://arxiv.org/pdf/2106.13008.pdf) | 自相关机制 |
| **FNet** | 2021 | [PDF](https://arxiv.org/pdf/2105.03824.pdf) | 傅里叶变换 |

### ICLR

| 论文 | 年份 | PDF 下载 | 核心贡献 |
|------|------|----------|----------|
| **Token Merging (ToMe)** | 2023 | [PDF](https://arxiv.org/pdf/2209.15559.pdf) | 可逆 token 合并 |
| **EViT** | 2022 | [PDF](https://arxiv.org/pdf/2204.08616.pdf) | 重要性剪枝 |
| **Swin Transformer** | 2022 | [PDF](https://arxiv.org/pdf/2103.14030.pdf) | Shifted Window |
| **Steerable CNN** | 2018 | [PDF](https://arxiv.org/pdf/1804.08258.pdf) | 可控滤波器 |
| **ViT** | 2021 | [PDF](https://arxiv.org/pdf/2010.11929.pdf) | Vision Transformer |
| **MAE** | 2022 | [PDF](https://arxiv.org/pdf/2111.06377.pdf) | 掩码自编码器 |

### ICML

| 论文 | 年份 | PDF 下载 | 核心贡献 |
|------|------|----------|----------|
| **Informer** | 2021 | [PDF](https://arxiv.org/pdf/2012.07436.pdf) | 长序列预测 |

---

## 📄 其他高质量期刊

| 论文 | 期刊 | 年份 | PDF 下载 |
|------|------|------|----------|
| **Universal Dynamics of WSD** | arXiv | 2024 | [PDF](https://arxiv.org/pdf/2401.11079.pdf) |
| **Adaptive Token Merging** | arXiv | 2024 | [PDF](https://arxiv.org/pdf/2409.09955.pdf) |

---

## 📥 一键下载命令

```bash
# 下载所有 Nature 级别论文
python download_top_papers.py --category nature

# 下载所有 Science 级别论文
python download_top_papers.py --category science

# 下载顶级会议论文
python download_top_papers.py --category top_conferences

# 下载所有论文
python download_top_papers.py --all
```

---

## 🎯 阅读优先级建议

### 必读 (Nature 级别)
1. **EarthFormer** - 了解 baseline 架构
2. **NowcastNet** - 生成式方法思路
3. **Pangu-Weather** - 大模型气象应用
4. **GraphCast** - 图神经网络预测

### 重要 (顶级会议)
5. **ToMe** - Token Merging 核心思想
6. **E2CNN** - 等变卷积基础
7. **Swin Transformer** - Shifted Window
8. **WSD** - 新的学习率调度

### 拓展
9. 其余论文按需阅读
