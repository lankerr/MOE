# 第三代: SSM/Mamba 线性扫描 → 隐空间映射

> **核心思想**: 用**状态空间模型 (SSM)** 替代注意力机制，将序列建模从 O(N²) 降到 **O(N)**，
> 通过选择性扫描 (Selective Scan) 实现数据依赖的动态过滤。

---

## 一、代表论文

| 论文 | 会议 | 核心创新 | 与我们的关系 |
|------|------|---------|------------|
| **S4** (Gu et al., 2022) | ICLR 2022 | 结构化状态空间模型，长程依赖建模 | SSM 奠基工作 |
| **Mamba** (Gu & Dao, 2023) | NeurIPS 2023 (oral) | **选择性 SSM** + 硬件感知并行扫描算法 | 核心参考，线性复杂度的 Transformer 替代 |
| **Mamba-2** (Dao & Gu, 2024) | ICML 2024 | SSM = 结构化注意力，SSD 算法 (2-8× 加速) | 理论统一 SSM 与 Attention |
| **Vision Mamba (Vim)** (Zhu et al., 2024) | ICML 2024 | 双向 Mamba + 位置编码 → 替代 ViT | 视觉 SSM 的首个工作 |
| **VMamba** (Liu et al., 2024) | NeurIPS 2024 | VSS Block + 四方向交叉扫描 (Cross-Scan) | **2D 图像最优扫描策略** |
| **VideoMamba** (Li et al., 2024) | ECCV 2024 | 双向 Mamba → 视频理解，线性复杂度 | 视频/时序 SSM |
| **MambaOut** (Yu et al., 2024) | arXiv | "Do We Really Need Mamba for Vision?" — 用纯 Conv 对标 | 重要消融参考 |
| **ARM** (Anonymous, 2024) | arXiv | 自回归 Mamba 做图像生成 (MAE-like) | Mamba + 自回归重建 |

## 二、参考 GitHub 仓库

| 仓库 | Stars | 说明 | 推荐度 |
|------|-------|------|--------|
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | 13k+ | 官方 Mamba 实现 (CUDA kernel) | ⭐⭐⭐⭐⭐ |
| [hustvl/Vim](https://github.com/hustvl/Vim) | 2.5k+ | Vision Mamba，官方代码 | ⭐⭐⭐⭐⭐ |
| [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) | 2k+ | VMamba，四方向交叉扫描 | ⭐⭐⭐⭐⭐ |
| [OpenGVLab/VideoMamba](https://github.com/OpenGVLab/VideoMamba) | 600+ | VideoMamba，视频 SSM | ⭐⭐⭐⭐ |
| [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal) | 3k+ | **极简 Mamba (~200行 Python)**，最适合学习 | ⭐⭐⭐⭐⭐ |
| [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) | 1.5k+ | 纯 Python Mamba (不依赖 CUDA kernel) | ⭐⭐⭐⭐ |
| [radarSAR/Mamba-in-Mamba](https://github.com/FerneyOAmworworry/MiM-ISTD) | 200+ | Mamba-in-Mamba 红外小目标检测 | ⭐⭐⭐ |

## 三、SSM 核心原理

### 3.1 连续 SSM

$$
\dot{x}(t) = Ax(t) + Bu(t) \\\\
y(t) = Cx(t) + Du(t)
$$

- $x(t) \in \mathbb{R}^N$: 隐状态
- $u(t)$: 输入序列
- $A, B, C, D$: 系统矩阵

### 3.2 离散化 (ZOH)

$$
\bar{A} = e^{\Delta A}, \quad \bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B
$$

### 3.3 Mamba 的选择性机制

关键创新: **$B, C, \Delta$ 都是输入依赖的** (不再是固定参数)

```python
# 伪代码
B = Linear(x)           # (B, L, N) — 输入依赖
C = Linear(x)           # (B, L, N) — 输入依赖  
Δ = softplus(Linear(x)) # (B, L, D) — 输入依赖的步长

# 选择性扫描 (并行版本)
y = selective_scan(x, Δ, A, B, C)
```

这使得 Mamba 能像注意力一样**选择性地关注/忽略输入**，但时间复杂度为 O(L)。

## 四、架构设计 (Fashion-MNIST 验证方案)

### 4.1 Mamba-MAE 架构

```
Fashion-MNIST 28×28
       │
  [Patch Embed]  4×4 patch → 49 tokens, dim=128
       │
  ┌────▼──────────┐
  │  Mamba Block  │  ×6 层
  │  BiMamba +    │  双向扫描: 前向 + 后向
  │  FFN          │  选择性 SSM 替代 Attention
  └────┬──────────┘
       │
  [平均池化] → 隐空间 (dim=128)
       │
  ┌────▼────────┐
  │  Decoder     │ Linear (轻量) → unpatchify → 28×28
  └──────────────┘
```

### 4.2 2D 扫描策略

Mamba 原生是 1D 序列模型。要处理 2D 图像，需要选择扫描路径:

```
方案 A: Raster Scan (最简单)         方案 B: 蛇形扫描 (Snake)
→ → → → → → →                       → → → → → → →
→ → → → → → →                       ← ← ← ← ← ← ←
→ → → → → → →                       → → → → → → →

方案 C: 四方向交叉扫描 (VMamba, 最优)
→→→  +  ←←←  +  ↓↓↓  +  ↑↑↑
然后 merge 四个方向的输出
```

**推荐**: 对 Fashion-MNIST 用方案 A (Raster) 或方案 C (Cross-Scan)

### 4.3 关键设计选择

| 设计点 | 选择 | 理由 |
|--------|------|------|
| SSM 方式 | 纯 Python Mamba (不依赖 CUDA) | RTX 3050Ti 的 CUDA 兼容性问题 |
| 扫描方向 | 双向 (forward + backward) | 图像无因果关系，需双向 |
| 隐状态维度 N | 16 | 轻量，N 大了 SSM 反而慢 |
| 扩展比 expand | 2 | d_inner = 2 × d_model |
| dt_rank | auto (d_model // 16) | Mamba 默认 |
| 掩码策略 | SimMIM 风格 (不删除 token) | Mamba 不适合可变长度输入 |

### 4.4 与 MAE/Swin 的对比实验

| 对比维度 | MAE (ViT) | Swin-MAE | Mamba-MAE | 预期 |
|----------|-----------|----------|-----------|------|
| 注意力类型 | 全局 O(N²) | 局部窗口 O(N·w²) | SSM O(N) | Mamba 最快 |
| 长程依赖 | 好 | 需 shift | 通过隐状态传递 | 各有优劣 |
| 参数量 | 1.36M | ~1.5M | ~1.4M | 相当 |
| 序列长度扩展性 | 差 (N² 增长) | 好 | **极好 (线性)** | Mamba 优势场景 |
| 实现复杂度 | 低 | 中 | 中高 | |

## 五、核心代码框架

```python
class MambaBlock(nn.Module):
    """简化 Mamba Block — 不依赖 CUDA kernel"""
    def __init__(self, d_model=128, d_state=16, expand=2, dt_rank='auto'):
        super().__init__()
        d_inner = d_model * expand
        dt_rank = d_model // 16 if dt_rank == 'auto' else dt_rank
        
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # z + x
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3, 
                                padding=1, groups=d_inner)  # depthwise conv
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2)  # dt, B, C
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        
        # A 矩阵 (固定初始化)
        A = torch.arange(1, d_state + 1).float().repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        self.out_proj = nn.Linear(d_inner, d_model)
    
    def selective_scan(self, x, dt, A, B, C, D):
        """纯 Python 选择性扫描"""
        B_batch, L, d_inner = x.shape
        N = A.shape[1]
        
        # 离散化
        dtA = torch.einsum('bld,dn->bldn', dt, A)
        dA = torch.exp(dtA)
        dB_x = torch.einsum('bld,bln,bld->bldn', dt, B, x)
        
        # 递推扫描
        h = torch.zeros(B_batch, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB_x[:, i]
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y


class BiMambaEncoder(nn.Module):
    """双向 Mamba Encoder"""
    def __init__(self, d_model=128, depth=6, d_state=16):
        # forward_mamba + backward_mamba → 合并
        ...
```

## 六、⚠️ 注意事项

1. **CUDA Kernel 依赖**: 官方 Mamba 用 CUDA selective scan kernel，Windows 上编译困难。
   **解决**: 用纯 Python 版本 (`mamba-minimal` 或 `alxndrTL/mamba.py`)
2. **2D 扫描顺序**: Raster scan 会丢失空间局部性，VMamba 的 Cross-Scan 更好但实现复杂
3. **掩码策略**: Mamba 是序列模型，不能像 MAE 那样删除 token (会破坏扫描连续性)
   → 必须用 SimMIM 风格 (mask embedding 替换)
4. **MambaOut**: 有论文质疑 Mamba 在视觉任务上不如 Conv。需要在实验中验证。

## 七、与雷达应用的衔接

| 特性 | Mamba 优势 | 应用场景 |
|------|----------|---------|
| 线性复杂度 | 49帧×384×384 的长序列无压力 | 雷达全序列扫描 |
| 选择性机制 | 自动忽略空白区域 | 天然适合稀疏雷达数据 |
| 因果扫描 | 时间维度 causal → 自然适合预测 | 时序预测方向直接用 |
| 隐状态 | 压缩历史信息，形成紧凑表示 | 隐空间映射 |

**Radar-Mamba 思路**: 空间维度用双向 Mamba (或 Cross-Scan)，时间维度用单向 causal Mamba，
这样预测时只需要从时间方向续写隐状态即可。

---

## 八、参考文献

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv:2312.00752},
  year={2023}
}

@article{dao2024mamba2,
  title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  journal={ICML},
  year={2024}
}

@article{zhu2024vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  journal={ICML},
  year={2024}
}

@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={NeurIPS},
  year={2024}
}

@article{li2024videomamba,
  title={VideoMamba: State Space Model for Efficient Video Understanding},
  author={Li, Kunchang and Li, Xinhao and Wang, Yi and others},
  journal={ECCV},
  year={2024}
}
```
