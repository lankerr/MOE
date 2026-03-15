# 顶刊论文库

## 📁 目录结构

```
top_papers/
├── 🏆 nature/              # Nature 级别论文
│   ├── dgmr/              # DeepMind 生成式外推
│   ├── nowcastnet/        # 清华大学极端降水
│   ├── pangu_weather/     # 华为盘古大模型
│   ├── graphcast/         # DeepMind 图神经网络
│   └── fourcastnet/       # NVIDIA 傅里叶神经算子
│
├── 🔬 science/             # Science 级别论文
│   └── graphcast/         # GraphCast Science 版
│
├── 📚 top_conferences/     # 顶级会议论文
│   ├── transformer/       # Transformer 架构
│   ├── token_efficient/   # Token 高效处理
│   └── equivariant/       # 等变网络
│
└── 📄 others/             # 其他高质量期刊
```

## 📥 一键下载脚本

### PowerShell (Windows)
```powershell
# 进入目录
cd top_papers

# Nature 级别
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2104.00954.pdf" -OutFile "nature/dgmr/DGMR_Nature2021.pdf"
Invoke-WebRequest -Uri "https://www.nature.com/articles/s41586-023-06184-4.pdf" -OutFile "nature/nowcastnet/NowcastNet_Nature2023.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2211.02556.pdf" -OutFile "nature/pangu_weather/PanguWeather_Nature2023.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2212.12794.pdf" -OutFile "nature/graphcast/GraphCast_Nature2023.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2202.11214.pdf" -OutFile "nature/fourcastnet/FourCastNet_Nature2022.pdf"

# 顶级会议 - Transformer
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2207.05833.pdf" -OutFile "top_conferences/transformer/EarthFormer_NeurIPS2022.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2103.14030.pdf" -OutFile "top_conferences/transformer/SwinTransformer_ICLR2022.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2010.11929.pdf" -OutFile "top_conferences/transformer/ViT_ICLR2021.pdf"

# 顶级会议 - Token 高效
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2209.15559.pdf" -OutFile "top_conferences/token_efficient/ToMe_ICLR2023.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2204.08616.pdf" -OutFile "top_conferences/token_efficient/EViT_ICLR2022.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2106.01304.pdf" -OutFile "top_conferences/token_efficient/DynamicViT_NeurIPS2021.pdf"

# 顶级会议 - 等变网络
Invoke-WebRequest -Uri "https://arxiv.org/pdf/1802.08219.pdf" -OutFile "top_conferences/equivariant/E2CNN_NeurIPS2018.pdf"
Invoke-WebRequest -Uri "https://arxiv.org/pdf/1804.08258.pdf" -OutFile "top_conferences/equivariant/SteerableCNN_ICLR2018.pdf"

# Science 级别
Invoke-WebRequest -Uri "https://www.science.org/doi/pdf/10.1126/science.adi2336" -OutFile "science/graphcast/GraphCast_Science2023.pdf"
```

### Bash (Linux/Mac)
```bash
cd top_papers

# Nature 级别
wget -O nature/dgmr/DGMR_Nature2021.pdf https://arxiv.org/pdf/2104.00954.pdf
wget -O nature/nowcastnet/NowcastNet_Nature2023.pdf https://www.nature.com/articles/s41586-023-06184-4.pdf
wget -O nature/pangu_weather/PanguWeather_Nature2023.pdf https://arxiv.org/pdf/2211.02556.pdf
wget -O nature/graphcast/GraphCast_Nature2023.pdf https://arxiv.org/pdf/2212.12794.pdf
wget -O nature/fourcastnet/FourCastNet_Nature2022.pdf https://arxiv.org/pdf/2202.11214.pdf

# 顶级会议
wget -O top_conferences/transformer/EarthFormer_NeurIPS2022.pdf https://arxiv.org/pdf/2207.05833.pdf
wget -O top_conferences/token_efficient/ToMe_ICLR2023.pdf https://arxiv.org/pdf/2209.15559.pdf
wget -O top_conferences/equivariant/E2CNN_NeurIPS2018.pdf https://arxiv.org/pdf/1802.08219.pdf
```

## 📖 论文速查表

### Nature 级别 (5篇核心)
| 论文 | 文件名 | 链接 |
|------|--------|------|
| DGMR | `DGMR_Nature2021.pdf` | [arxiv](https://arxiv.org/pdf/2104.00954.pdf) |
| NowcastNet | `NowcastNet_Nature2023.pdf` | [Nature](https://www.nature.com/articles/s41586-023-06184-4.pdf) |
| Pangu-Weather | `PanguWeather_Nature2023.pdf` | [arxiv](https://arxiv.org/pdf/2211.02556.pdf) |
| GraphCast | `GraphCast_Nature2023.pdf` | [arxiv](https://arxiv.org/pdf/2212.12794.pdf) |
| FourCastNet | `FourCastNet_Nature2022.pdf` | [arxiv](https://arxiv.org/pdf/2202.11214.pdf) |

### 顶级会议 (必读 8 篇)
| 论文 | 文件名 | 链接 |
|------|--------|------|
| EarthFormer | `EarthFormer_NeurIPS2022.pdf` | [arxiv](https://arxiv.org/pdf/2207.05833.pdf) |
| ToMe | `ToMe_ICLR2023.pdf` | [arxiv](https://arxiv.org/pdf/2209.15559.pdf) |
| E2CNN | `E2CNN_NeurIPS2018.pdf` | [arxiv](https://arxiv.org/pdf/1802.08219.pdf) |
| Swin Transformer | `SwinTransformer_ICLR2022.pdf` | [arxiv](https://arxiv.org/pdf/2103.14030.pdf) |
| ViT | `ViT_ICLR2021.pdf` | [arxiv](https://arxiv.org/pdf/2010.11929.pdf) |
| EViT | `EViT_ICLR2022.pdf` | [arxiv](https://arxiv.org/pdf/2204.08616.pdf) |
| DynamicViT | `DynamicViT_NeurIPS2021.pdf` | [arxiv](https://arxiv.org/pdf/2106.01304.pdf) |
| Steerable CNN | `SteerableCNN_ICLR2018.pdf` | [arxiv](https://arxiv.org/pdf/1804.08258.pdf) |

## 🎯 阅读优先级

### 第一优先级 (必读，直接相关)
1. [EarthFormer](https://arxiv.org/pdf/2207.05833.pdf) - 你的 baseline
2. [ToMe](https://arxiv.org/pdf/2209.15559.pdf) - Token Merging
3. [E2CNN](https://arxiv.org/pdf/1802.08219.pdf) - 等变卷积

### 第二优先级 (重要参考)
4. [DGMR](https://arxiv.org/pdf/2104.00954.pdf) - 生成式方法
5. [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) - Shifted Window
6. [GraphCast](https://arxiv.org/pdf/2212.12794.pdf) - 图神经网络

### 第三优先级 (拓展阅读)
7. 其余论文按需阅读
