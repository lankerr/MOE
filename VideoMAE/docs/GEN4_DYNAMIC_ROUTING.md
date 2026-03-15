# 第四代: 动态路由 / 可学习稀疏 → 隐空间映射

> **核心思想**: 模型自动学习将计算资源分配给信息密集区域，
> 通过**可学习门控/路由**实现 token 级别的动态计算，
> 结合 MoE (Mixture of Experts) 实现条件计算。

---

## 一、代表论文

| 论文 | 会议 | 核心创新 | 与我们的关系 |
|------|------|---------|------------|
| **Dynamic ViT** (Rao et al., 2021) | NeurIPS 2021 | 可学习门控渐进式剪枝 token | 直接参考: 不重要 token 在深层被丢弃 |
| **A-ViT** (Yin et al., 2022) | CVPR 2022 | 自适应 token halting (ACT 机制) | 每个 token 自适应决定在哪层停止 |
| **EViT** (Liang et al., 2022) | ICLR 2022 | Top-K attentive token + fuse inattentive | 保留重要 token，融合不重要 token |
| **ToMe** (Bolya et al., 2023) | ICLR 2023 | Token Merging — 合并相似 token | 无需训练的 token 减少方案 |
| **Q-Sparse** (Wang et al., 2024) | arXiv 2024 | Top-K 稀疏注意力 + STE 门控 | 注意力层级的动态稀疏 |
| **NSA** (Yuan et al., 2025) | arXiv 2025 | 分层稀疏注意力 (压缩+选择+滑窗) | 最新 native sparse attention |
| **Switch Transformer** (Fedus et al., 2022) | JMLR 2022 | MoE: 每 token 路由到 1 个 expert | 条件计算的经典范式 |
| **Soft MoE** (Puigcerver et al., 2024) | ICLR 2024 | 软路由: token 不二选一, 软组合 expert | 比硬路由更稳定 |
| **Mixture of Depths** (Raposo et al., 2024) | arXiv 2024 | 每层动态选择处理哪些 token | 深度维度的动态路由 |

## 二、参考 GitHub 仓库

| 仓库 | Stars | 说明 | 推荐度 |
|------|-------|------|--------|
| [raoyongming/DynamicViT](https://github.com/raoyongming/DynamicViT) | 300+ | Dynamic ViT 官方，可学习 token 剪枝 | ⭐⭐⭐⭐⭐ |
| [NVlabs/A-ViT](https://github.com/NVlabs/A-ViT) | 200+ | 自适应 token halting | ⭐⭐⭐⭐ |
| [youweiliang/evit](https://github.com/youweiliang/evit) | 200+ | EViT: attentive token 保留 | ⭐⭐⭐⭐ |
| [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe) | 1k+ | Token Merging，即插即用 | ⭐⭐⭐⭐⭐ |
| [google/flaxformer (MoE)](https://github.com/google/flaxformer) | 1k+ | Google Switch/MoE 实现 | ⭐⭐⭐ |
| [huggingface/transformers (Mixtral)](https://huggingface.co/docs/transformers/model_doc/mixtral) | — | Mixtral MoE 实现 | ⭐⭐⭐⭐ |
| [XuezheMax/megalodon](https://github.com/XuezheMax/megalodon) | 800+ | Megalodon: 无限序列长度 | ⭐⭐⭐ |
| [lucidrains/mixture-of-experts](https://github.com/lucidrains/mixture-of-experts) | 300+ | 极简 MoE 实现 (PyTorch 单文件) | ⭐⭐⭐⭐⭐ |

## 三、架构设计 (Fashion-MNIST 验证方案)

### 3.1 路线 A: Dynamic Token Pruning MAE

```
Fashion-MNIST 28×28
       │
  [Patch Embed]  4×4 → 49 tokens, dim=128
       │
  ┌────▼─────────┐
  │ ViT Block ×2 │  正常处理全部 49 tokens
  └────┬─────────┘
       │
  ┌────▼─────────────┐
  │ Pruning Gate     │  MLP(token) → sigmoid → binary (STE)
  │ keep top-K       │  K = 49 × (1 - prune_ratio)
  └────┬─────────────┘
       │
  ┌────▼─────────┐
  │ ViT Block ×4 │  只处理 K 个重要 token (省计算)
  └────┬─────────┘
       │
  ┌────▼────────┐
  │  Decoder     │ 恢复全部 49 → predict → 重建
  └──────────────┘
```

### 3.2 路线 B: MoE-MAE (每 token 分流到不同 Expert)

```
Fashion-MNIST 28×28
       │
  [Patch Embed]  49 tokens, dim=128
       │
  ┌────▼──────────────────────┐
  │ MoE Transformer Block ×6 │
  │  Self-Attn → Router →    │
  │  Expert1 (FFN-large)     │  ← 复杂 token (衣服纹理)
  │  Expert2 (FFN-small)     │  ← 简单 token (背景)
  │  Expert3 (Identity)      │  ← 空白 token (跳过)
  └────┬──────────────────────┘
       │
  [隐空间] → Decoder → 重建
```

### 3.3 路线 C: Mixture of Depths MAE

```
每一层 Transformer:
  Router(x) → 决定哪些 token 经过 Attention + FFN
                哪些 token 直接 skip (residual only)

优势: 不改变 token 数量，但减少实际计算量
      某些 token 可能只经过 2/6 层处理
```

### 3.4 关键设计选择

| 设计点 | 路线 A | 路线 B | 路线 C |
|--------|--------|--------|--------|
| token 数量变化 | 减少 | 不变 | 不变 |
| 计算分配 | 按重要性二选一 | 按复杂度分流 | 按层级跳过 |
| 额外参数 | Prune MLP | Router + 多FFN | Router (每层) |
| 训练难度 | 中 (STE) | 高 (负载均衡) | 中 |
| **推荐实验** | ✅ 先做 | 第二步 | 第二步 |

## 四、核心代码框架

### 4.1 Dynamic Token Pruning

```python
class TokenPruner(nn.Module):
    """可学习 token 剪枝门控"""
    def __init__(self, dim, prune_ratio=0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        self.prune_ratio = prune_ratio
    
    def forward(self, x):
        # x: (B, N, D)
        scores = self.gate(x).squeeze(-1)  # (B, N)
        scores = torch.sigmoid(scores)
        
        # Top-K 保留
        k = int(x.shape[1] * (1 - self.prune_ratio))
        topk_idx = scores.topk(k, dim=1).indices   # (B, K)
        topk_idx = topk_idx.sort(dim=1).values      # 保持原始顺序
        
        # STE: 前向用硬选择，反向通过 soft score 传梯度
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)
        mask_ste = scores + (mask - scores).detach()
        
        # 收集保留 token
        x_pruned = torch.gather(x, 1, 
            topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        return x_pruned, mask_ste, topk_idx
```

### 4.2 Simple MoE Layer

```python
class MoEFFN(nn.Module):
    """简单 2-Expert MoE FFN"""
    def __init__(self, dim, num_experts=2, capacity_factor=1.5):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            MLP(dim, dim * 4) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: (B, N, D)
        scores = F.softmax(self.router(x), dim=-1)  # (B, N, E)
        # Top-1 routing
        idx = scores.argmax(dim=-1)  # (B, N)
        
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (idx == e)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[e](expert_input)
                output[mask] = expert_output * scores[mask, e:e+1]
        
        return output
```

## 五、与 MAE/Swin/Mamba 的系统对比

| 特性 | Gen1 MAE | Gen2 Swin | Gen3 Mamba | **Gen4 动态路由** |
|------|----------|-----------|------------|-------------------|
| 注意力 | 全局 | 局部窗口 | 线性 SSM | 动态稀疏 |
| FLOPs 节省 | mask省encoder | 窗口省N² | O(N)替代O(N²) | **跳过不重要token** |
| 自适应性 | 无 (随机mask) | 无 (固定窗口) | 弱 (选择性SSM) | **强 (可学习路由)** |
| MoE 兼容 | ✗ | ✗ | ✗ | **✓ 天然适合** |
| 参数效率 | 全参数全用 | 全参数全用 | 全参数全用 | **条件参数: 按需激活** |

## 六、⚠️ 注意事项

1. **STE 训练不稳定**: Token pruning 的 hard selection 需要 Straight-Through Estimator，
   初期 loss 可能震荡。**建议**: 前 10 epoch 不剪枝 (warmup)
2. **负载均衡**: MoE 路由可能坍缩 (所有 token 选同一个 expert)。
   **解决**: 加 load balancing loss
3. **梯度问题**: 被剪枝的 token 没有梯度。
   **解决**: 保留 skip-connection (pruned token 可通过 residual 仍然获得梯度)
4. **Fashion-MNIST 太简单**: 49 tokens 中动态路由效果可能不明显。
   建议扩大到 14×14=196 tokens (patch_size=2)

## 七、与雷达应用的衔接

| 雷达场景 | 第四代优势 |
|----------|----------|
| 66% 空气区域 | Token pruning 自动丢弃 → 60%+ 计算节省 |
| 降水核心 (强回波) | MoE 中的 "heavy expert" 专门处理 |
| 边界区域 (不确定) | 软路由: 同时使用多个 expert 的加权组合 |
| 多尺度特征 | Mixture of Depths: 简单区域浅层即可，复杂区域深层处理 |

**核心洞察**: 气象数据天然适合动态路由 — 晴空区域无需深度处理，
只有降水区域需要全部计算资源。这和 NLP 中的 "easy/hard tokens" 完全一致。

---

## 八、参考文献

```bibtex
@inproceedings{rao2021dynamicvit,
  title={DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification},
  author={Rao, Yongming and Zhao, Wenliang and Liu, Benlin and Lu, Jiwen and Zhou, Jie and Hsieh, Cho-Jui},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{bolya2023tome,
  title={Token Merging: Your ViT But Faster},
  author={Bolya, Daniel and Fu, Cheng-Yang and Dai, Xiaoliang and Zhang, Peizhao and Feichtenhofer, Christoph and Hoffman, Judy},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{puigcerver2024softmoe,
  title={From Sparse to Soft Mixtures of Experts},
  author={Puigcerver, Joan and Riquelme, Carlos and Mustafa, Basil and Houlsby, Neil},
  booktitle={ICLR},
  year={2024}
}

@article{raposo2024mixtureofdepths,
  title={Mixture-of-Depths: Dynamically allocating compute in transformer-based language models},
  author={Raposo, David and Ritter, Sam and Richards, Blake and Lillicrap, Timothy and Humphreys, Peter C and Santoro, Adam},
  journal={arXiv:2404.02258},
  year={2024}
}

@article{fedus2022switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={JMLR},
  year={2022}
}
```
