# 核心论文快速访问

## 一键下载（需要 requests 库）

```bash
pip install requests
python download_papers.py --all
```

## 直接链接（点击下载）

### 学习率调度器

| 论文 | PDF | 核心贡献 |
|------|-----|----------|
| **Universal Dynamics of Warmup Stable Decay** | [PDF](https://arxiv.org/pdf/2401.11079.pdf) | WSD 三阶段调度器 |
| WHEN WHY AND HOW MUCH | [OpenReview](https://openreview.net) | 自适应学习率 |

### Token 高效处理

| 论文 | PDF | 核心贡献 |
|------|-----|----------|
| **Token Merging (ToMe)** | [PDF](https://arxiv.org/pdf/2209.15559.pdf) | 合并非丢弃 |
| **EViT** | [PDF](https://arxiv.org/pdf/2204.08616.pdf) | 重要性打分 |
| **DynamicViT** | [PDF](https://arxiv.org/pdf/2106.01304.pdf) | 逐层剪枝 |
| Adaptive Token Merging | [PDF](https://arxiv.org/pdf/2409.09955.pdf) | 自适应合并 |

### 气象 AI

| 论文 | PDF | 核心贡献 |
|------|-----|----------|
| **EarthFormer** | [PDF](https://arxiv.org/pdf/2207.05833.pdf) | Cuboid Attention |
| **NowcastNet** | [Nature](https://www.nature.com/articles/s41586-023-06184-4) | 生成式预测 |
| **PreDiff** | [PDF](https://arxiv.org/pdf/2309.15025.pdf) | 扩散模型 |
| **Pangu-Weather** | [Nature](https://www.nature.com/articles/s41586-023-06184-4) | 3D 神经网络 |

### 等变网络

| 论文 | PDF | 核心贡献 |
|------|-----|----------|
| **E2CNN** | [PDF](https://arxiv.org/pdf/1802.08219.pdf) | 群等变 CNN |
| **Steerable CNN** | [PDF](https://arxiv.org/pdf/1804.08258.pdf) | 可控滤波器 |

## 阅读顺序建议

### 第一阶段（必读）
1. EarthFormer - 了解 baseline 架构
2. ToMe - 理解 token merging 思想
3. WSD - 学习新的学习率调度

### 第二阶段（深入）
4. EViT - token 重要性打分
5. E2CNN - 等变卷积原理
6. DynamicViT - 动态剪枝策略

### 第三阶段（拓展）
7. NowcastNet - 生成式方法
8. PreDiff - 扩散模型应用
9. 其余论文按需阅读
