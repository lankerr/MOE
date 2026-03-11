# DATSwinLSTM-Memory：基于记忆增强的时空降水临近预报模型

## 📋 项目概述

本项目是 **DATSwinLSTM-D-Memory** 模型的实现代码，该模型结合了：
- **DAT (Deformable Attention Transformer)**：可变形注意力机制
- **SwinLSTM**：基于Swin Transformer的时序建模
- **Memory Bank**：记忆库增强机制，用于长短序列的特征记忆

### 核心创新点
1. **两阶段训练策略**：Phase 1 存储长序列特征到记忆库，Phase 2 利用记忆库进行短序列预测
2. **可变形注意力**：自适应关注降水区域的关键特征
3. **多尺度特征融合**：上下采样架构捕获不同尺度的时空特征

---

## 📊 数据集要求

### 主要数据集：SEVIR (Storm EVent ImagRy)

| 项目 | 详情 |
|------|------|
| **数据集大小** | 约 **1 TB** (完整版) |
| **事件数量** | 20,393+ 个天气事件 |
| **图像尺寸** | 384 × 384 像素 |
| **时间跨度** | 每个事件 4 小时 |
| **时间分辨率** | 5 分钟/帧 |
| **序列长度** | 49 帧/事件 |

### 数据内容
- **VIL (Vertically Integrated Liquid)**：垂直积分液态水含量（主要用于本模型）
- **C02, C09, C13**：GOES-16 卫星多通道图像
- **GLM**：闪电定位数据

### 下载方式

#### 方式一：AWS CLI（推荐）
```bash
# 安装 AWS CLI
pip install awscli

# 下载完整数据集（约1TB）
aws s3 sync --no-sign-request s3://sevir ./sevir_data

# 仅下载 VIL 数据（约200GB，本模型主要使用）
aws s3 sync --no-sign-request s3://sevir/data/vil ./sevir_data/vil
```

#### 方式二：Python boto3
```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
# 下载 CATALOG.csv
s3.download_file('sevir', 'CATALOG.csv', 'CATALOG.csv')
```

#### 官方链接
- **AWS Open Data Registry**: https://registry.opendata.aws/sevir/
- **官方文档**: https://sevir.mit.edu/

---

## 🔧 环境配置

### 依赖项
```bash
# 创建conda环境
conda create -n datswinlstm python=3.8
conda activate datswinlstm

# 安装PyTorch (CUDA 11.3示例)
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip install pytorch-lightning==1.6.0
pip install numpy matplotlib h5py pandas
```

### 硬件需求
| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| **GPU显存** | 16 GB | 32+ GB |
| **系统内存** | 32 GB | 64+ GB |
| **存储空间** | 200 GB (仅VIL) | 1.5 TB (完整) |
| **GPU型号** | RTX 3090 | A100 / H100 |

---

## 📁 目录结构

```
datswinlstm_memory/
├── README.md                 # 本文档
├── docs/                     # 文档和论文
│   └── paper.pdf            # 相关论文（如有）
├── configs/
│   └── config.py            # 模型和训练配置
├── models/
│   ├── DATSwinLSTM_D_Memory.py  # 主模型实现
│   ├── MotionSqueeze.py         # 运动特征提取
│   └── dat_blocks.py            # DAT注意力模块
└── run_sevir_multiGpu_distributedSampler.py  # 训练脚本
```

---

## 🚀 训练流程

### Phase 1：长序列记忆存储
```python
# 启用记忆库梯度更新
model.memory_bank.requires_grad = True

# 使用完整序列作为记忆输入
outputs = model(inputs, memory_x=train_data, phase=1)
loss_phase_1 = loss(outputs, targets)
loss_phase_1.backward()
optimizer.step()
```

### Phase 2：短序列预测
```python
# 冻结记忆库
model.memory_bank.requires_grad = False

# 仅使用输入序列进行预测
outputs = model(inputs, memory_x=inputs, phase=2)
loss_phase_2 = loss(outputs, targets)
loss_phase_2.backward()
optimizer.step()
```

---

## ⚠️ 已知问题

> **注意**：本仓库代码可能不完整，缺少以下文件：
> - `sevir_dataloader.py`：SEVIR数据加载器
> - `sevir_torch_wrap.py`：PyTorch数据集封装
> - `sevir.py`：评估指标（CSI、POD等）
> - 根目录 `config.py`：全局配置文件
>
> 这些文件可能属于 MetNow 项目的私有部分。

---

## 📚 参考文献

如使用本代码，请引用：
```bibtex
@article{datswinlstm2024,
  title={DATSwinLSTM-D-Memory: Memory-Enhanced Deformable Attention Transformer for Precipitation Nowcasting},
  author={Zhang, Wei et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}

@inproceedings{sevir2020,
  title={SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology},
  author={Veillette, Mark and others},
  booktitle={NeurIPS 2020},
  year={2020}
}
```

---

## 📞 联系方式

如有问题，请通过 GitHub Issues 联系或参考原始论文。
