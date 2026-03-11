"""
Sparse Mixture of Experts (MoE) Layer
======================================
包含:
- Top-K Router (门控网络)
- 稀疏 MoE 前馈层 (替代标准 Mlp/FeedForward)
- Load Balancing Loss (Switch Transformer 风格)
- Expert Orthogonalization Loss (专家多样性约束)

设计目标:
- 8GB VRAM 友好: 稀疏激活 (Top-2/4专家), 每个token只激活2个专家
- 即插即用: 可直接替换原模型中的 Mlp 类
- 辅助损失: 防止专家坍塌 + 鼓励专家多样性

Reference:
- Switch Transformer (Fedus et al., 2021)
- ST-MoE (Zoph et al., 2022)
- DeepSpeed MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MoEConfig:
    """MoE 配置"""
    num_experts: int = 4          # 专家数量 (4-8, 显存友好)
    top_k: int = 2                # 每个token激活的专家数
    expert_dim: int = 0           # 专家隐藏维度 (0=auto, 使用 mlp_ratio * dim)
    mlp_ratio: float = 2.0        # 专家FFN维度 = dim * mlp_ratio
    use_swiglu: bool = True       # 专家内部使用 SwiGLU 激活
    # 辅助损失
    balance_loss_weight: float = 0.01   # Load balancing loss 权重
    ortho_loss_weight: float = 0.001    # Orthogonalization loss 权重
    # Router
    router_jitter: float = 0.01   # Router 输入噪声 (训练时)
    capacity_factor: float = 1.25 # 容量因子 (缓冲区)
    drop_tokens: bool = False     # 是否丢弃超出容量的token
    expert_chunk_size: int = 2048 # 每个专家一次处理的token数 (0=不分块)


# ===================== Auxiliary Losses =====================

def load_balancing_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor, 
                         num_experts: int) -> torch.Tensor:
    """
    Switch Transformer 风格的负载均衡损失
    
    鼓励每个专家处理大致相同数量的 token，防止专家坍塌。
    
    L_balance = N * Σ_i(f_i · P_i)
    其中:
        f_i = 分配给专家i的token比例 (离散,不可微)
        P_i = 路由到专家i的平均概率 (连续,可微)
        N = 专家数量
    
    Args:
        router_probs: (batch*seq, num_experts) 路由概率
        expert_indices: (batch*seq, top_k) 选中的专家索引
        num_experts: 专家数
    
    Returns:
        scalar loss
    """
    # f_i: 每个专家的 token 分配比例 (不可微)
    # 对所有 top-k 选择求 one-hot 然后求平均
    one_hot = F.one_hot(expert_indices, num_experts).float()  # (B*S, top_k, E)
    tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)  # (E,) 每个专家的平均token密度
    
    # P_i: 每个专家的平均路由概率 (可微)
    router_prob_per_expert = router_probs.mean(dim=0)  # (E,)
    
    # L = N * Σ(f_i * P_i)
    balance_loss = num_experts * (tokens_per_expert * router_prob_per_expert).sum()
    
    return balance_loss


def orthogonalization_loss(experts: nn.ModuleList) -> torch.Tensor:
    """
    专家正交化损失
    
    鼓励不同专家学习不同的特征表示，避免冗余。
    最小化专家权重矩阵之间的余弦相似度。
    
    L_ortho = Σ_{i≠j} |cos(W_i, W_j)|²
    
    Args:
        experts: nn.ModuleList of expert modules
    
    Returns:
        scalar loss
    """
    # 收集每个专家的第一层权重作为特征指纹
    weight_list = []
    for expert in experts:
        # 获取专家的第一层权重 (最能代表专家特化方向)
        if hasattr(expert, 'w1'):
            w = expert.w1.weight.flatten()  # SwiGLU expert
        elif hasattr(expert, 'fc1'):
            w = expert.fc1.weight.flatten()  # Standard expert
        elif hasattr(expert, 'net'):
            # Sequential expert
            for layer in expert.net:
                if hasattr(layer, 'weight'):
                    w = layer.weight.flatten()
                    break
        else:
            continue
        weight_list.append(w)
    
    if len(weight_list) < 2:
        return torch.tensor(0.0, device=weight_list[0].device if weight_list else 'cpu')
    
    # 堆叠权重
    W = torch.stack(weight_list)  # (E, D)
    W_norm = F.normalize(W, dim=1)  # L2 normalization
    
    # 计算余弦相似度矩阵
    cosine_sim = W_norm @ W_norm.T  # (E, E)
    
    # 去掉对角线 (自身相似度=1)
    E = cosine_sim.shape[0]
    mask = ~torch.eye(E, dtype=torch.bool, device=cosine_sim.device)
    off_diag = cosine_sim[mask]
    
    # L_ortho = mean(|cos_sim|²)
    ortho_loss = (off_diag ** 2).mean()
    
    return ortho_loss


# ===================== Router =====================

class TopKRouter(nn.Module):
    """
    Top-K 专家选择路由器
    
    对每个 token 独立计算路由概率，选择 top-k 个专家。
    训练时添加噪声以增加探索性。
    
    Args:
        dim: 输入特征维度
        num_experts: 专家总数
        top_k: 每个token选择的专家数
        jitter_noise: 训练时的噪声强度
    """
    
    def __init__(self, dim: int, num_experts: int, top_k: int = 2, 
                 jitter_noise: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch*seq, dim)
        
        Returns:
            gate_weights: (batch*seq, top_k)  选中专家的权重
            expert_indices: (batch*seq, top_k) 选中的专家索引
            router_probs: (batch*seq, num_experts) 完整路由概率 (用于aux loss)
        """
        # 训练时加噪声
        if self.training and self.jitter_noise > 0:
            x = x * (1.0 + torch.randn_like(x) * self.jitter_noise)
        
        # 计算 logits 和概率
        logits = self.gate(x)  # (B*S, E)
        router_probs = F.softmax(logits.float(), dim=-1)  # float32 for stability
        
        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 归一化 top-k 权重 (使其和为1)
        gate_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return gate_weights, top_k_indices, router_probs


# ===================== Expert FFN =====================

class StandardExpert(nn.Module):
    """标准 FFN 专家: Linear → GELU → Dropout → Linear → Dropout"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUExpert(nn.Module):
    """
    SwiGLU 门控激活专家 (Qwen/LLaMA 风格)
    
    SwiGLU(x) = W₂ · (SiLU(W_gate · x) ⊙ W₁ · x)
    
    参数量修正: 为保持与标准FFN相近的参数量, hidden_dim = dim * mlp_ratio * 2/3
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # 参数量修正: SwiGLU 有3个矩阵 (w1, w_gate, w2) vs 标准FFN 2个
        # 所以 hidden_dim 乘以 2/3 保持总参数量相当
        adjusted_hidden = int(hidden_dim * 2 / 3)
        # 确保能被8整除 (GPU对齐优化)
        adjusted_hidden = (adjusted_hidden + 7) // 8 * 8
        
        self.w1 = nn.Linear(dim, adjusted_hidden, bias=False)     # up projection
        self.w_gate = nn.Linear(dim, adjusted_hidden, bias=False)  # gate projection
        self.w2 = nn.Linear(adjusted_hidden, dim, bias=False)      # down projection
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU 中 gate*up 乘积在 fp16 下极易溢出 (fp16 max=65504)
        # 必须禁用 autocast 才能真正以 fp32 执行，否则 autocast 会偷偷 recast 回 fp16
        input_dtype = x.dtype
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            gate = F.silu(self.w_gate(x))
            up = self.w1(x)
            x = gate * up
            x = self.drop(x)
            x = self.w2(x)
            x = self.drop(x)
        return x.to(input_dtype)


# ===================== MoE Layer =====================

class MoELayer(nn.Module):
    """
    稀疏混合专家前馈层 (Sparse MoE FFN)
    
    即插即用替换原模型中的 Mlp / FeedForward。
    
    流程:
    1. Router 计算每个 token 的 top-k 专家权重
    2. 每个 token 被发送到选中的 k 个专家
    3. 专家输出按权重加权求和
    4. 同时计算辅助损失 (balance + ortho)
    
    Args:
        dim: 输入/输出特征维度
        config: MoEConfig 配置
        drop: Dropout rate
    
    用法:
        # 替代 Mlp(in_features=256, hidden_features=512, ...)
        moe = MoELayer(dim=256, config=MoEConfig(num_experts=4, top_k=2))
    """
    
    def __init__(self, dim: int, config: MoEConfig = None, drop: float = 0.0):
        super().__init__()
        if config is None:
            config = MoEConfig()
        
        self.config = config
        self.dim = dim
        self.num_experts = config.num_experts
        
        # 计算专家隐藏维度
        hidden_dim = config.expert_dim if config.expert_dim > 0 else int(dim * config.mlp_ratio)
        
        # Router
        self.router = TopKRouter(dim, config.num_experts, config.top_k, config.router_jitter)
        
        # 创建专家
        ExpertClass = SwiGLUExpert if config.use_swiglu else StandardExpert
        self.experts = nn.ModuleList([
            ExpertClass(dim, hidden_dim, drop) for _ in range(config.num_experts)
        ])
        
        # 存储辅助损失 (每次forward后可读取)
        self._aux_loss = torch.tensor(0.0)
        self._balance_loss = torch.tensor(0.0)
        self._ortho_loss = torch.tensor(0.0)
        self._expert_counts = None  # 用于监控
    
    @property
    def aux_loss(self) -> torch.Tensor:
        """获取最近一次 forward 的辅助损失"""
        return self._aux_loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) 或 (B*N, C) 或 (B, T, H, W, C) 等任意形状，最后一维为特征维
        
        Returns:
            out: 与 x 同形状的输出
        """
        orig_shape = x.shape
        C = x.shape[-1]
        if x.dim() >= 3:
            x_flat = x.reshape(-1, C)  # (..., C) -> (N_total, C)
        else:
            x_flat = x
        
        # 1) Router
        gate_weights, expert_indices, router_probs = self.router(x_flat)
        # gate_weights: (B*N, top_k), expert_indices: (B*N, top_k)
        
        # 2) 稀疏分发 + 专家计算
        # 使用 simple loop (显存友好, 适合少量专家)
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # 找到路由到第 i 个专家的 token
            # expert_indices: (B*N, top_k), 检查哪些 token 在 top_k 中包含专家 i
            expert_mask = (expert_indices == i)  # (B*N, top_k)
            has_expert = expert_mask.any(dim=-1)  # (B*N,) 布尔掩码
            
            if not has_expert.any():
                continue

            # 获取对应权重
            # 对于 top-k > 1, 同一token可能多次路由到同一专家, 取加和后的权重
            token_weights = (gate_weights * expert_mask.float()).sum(dim=-1)  # (B*N,)

            # 用 index + chunk 减少峰值显存
            selected_idx = has_expert.nonzero(as_tuple=False).squeeze(-1)  # (n_tokens,)
            chunk = self.config.expert_chunk_size if self.config.expert_chunk_size > 0 else selected_idx.numel()

            for s in range(0, selected_idx.numel(), chunk):
                idx_chunk = selected_idx[s:s + chunk]
                expert_input = x_flat.index_select(0, idx_chunk)  # (chunk, C)
                selected_weights = token_weights.index_select(0, idx_chunk).unsqueeze(-1)  # (chunk, 1)

                # 专家计算
                expert_output = expert(expert_input)

                # 加权累加到输出（避免布尔索引写回造成额外临时张量）
                output.index_add_(0, idx_chunk, (expert_output * selected_weights).to(output.dtype))
        
        # 3) 计算辅助损失
        if self.training:
            if self.config.balance_loss_weight > 0:
                self._balance_loss = load_balancing_loss(
                    router_probs, expert_indices, self.num_experts
                ) * self.config.balance_loss_weight
            else:
                self._balance_loss = torch.tensor(0.0, device=x_flat.device)

            if self.config.ortho_loss_weight > 0:
                self._ortho_loss = orthogonalization_loss(
                    self.experts
                ) * self.config.ortho_loss_weight
            else:
                self._ortho_loss = torch.tensor(0.0, device=x_flat.device)
            
            self._aux_loss = self._balance_loss + self._ortho_loss
            
            # 统计专家使用情况
            with torch.no_grad():
                self._expert_counts = torch.zeros(self.num_experts, device=x_flat.device)
                for i in range(self.num_experts):
                    self._expert_counts[i] = (expert_indices == i).sum().float()
        
        # 恢复形状
        output = output.reshape(orig_shape)
        return output
    
    def get_expert_stats(self) -> dict:
        """获取专家使用统计 (用于日志/监控)"""
        if self._expert_counts is None:
            return {}
        counts = self._expert_counts
        total = counts.sum()
        if total == 0:
            return {}
        ratios = counts / total
        return {
            'expert_counts': counts.tolist(),
            'expert_ratios': ratios.tolist(),
            'balance_score': 1.0 - ratios.std().item() / (ratios.mean().item() + 1e-8),
            'balance_loss': self._balance_loss.item(),
            'ortho_loss': self._ortho_loss.item(),
        }


# ===================== 便捷接口 =====================

def create_moe_from_mlp(dim: int, hidden_dim: int = None, drop: float = 0.0,
                         num_experts: int = 4, top_k: int = 2, 
                         use_swiglu: bool = True) -> MoELayer:
    """
    便捷函数: 创建一个与现有 Mlp 维度匹配的 MoE 层
    
    用于替换:
        mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    改为:
        mlp = create_moe_from_mlp(dim=dim, hidden_dim=mlp_hidden_dim, drop=drop)
    """
    mlp_ratio = (hidden_dim / dim) if hidden_dim else 2.0
    config = MoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        use_swiglu=use_swiglu,
    )
    return MoELayer(dim=dim, config=config, drop=drop)


def collect_moe_aux_losses(model: nn.Module) -> torch.Tensor:
    """
    遍历模型中所有 MoELayer, 收集辅助损失之和。
    
    用法 (training loop):
        loss = criterion(pred, target)
        aux_loss = collect_moe_aux_losses(model)
        total_loss = loss + aux_loss
        total_loss.backward()
    """
    total_aux = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, MoELayer):
            aux = module.aux_loss
            if aux.device != total_aux.device:
                total_aux = total_aux.to(aux.device)
            total_aux = total_aux + aux
    return total_aux


def get_all_expert_stats(model: nn.Module) -> dict:
    """收集模型中所有 MoE 层的专家统计"""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            s = module.get_expert_stats()
            if s:
                stats[name] = s
    return stats


if __name__ == "__main__":
    """测试 MoE 组件"""
    print("=" * 60)
    print("测试 MoE Layer")
    print("=" * 60)
    
    # 测试配置
    dim = 128
    seq_len = 64
    batch = 2
    
    x = torch.randn(batch, seq_len, dim)
    
    # 默认 MoE (4专家, Top-2, SwiGLU)
    config = MoEConfig(num_experts=4, top_k=2, use_swiglu=True, mlp_ratio=2.0)
    moe = MoELayer(dim=dim, config=config)
    
    # 参数统计
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"MoE 参数量: {total_params:,}")
    print(f"  - Router: {sum(p.numel() for p in moe.router.parameters()):,}")
    print(f"  - 每个专家: {sum(p.numel() for p in moe.experts[0].parameters()):,}")
    
    # 标准 Mlp 对比
    from torch.nn import Linear, GELU, Dropout, Sequential
    std_mlp = Sequential(
        Linear(dim, int(dim * 2)), GELU(), Dropout(0.0),
        Linear(int(dim * 2), dim), Dropout(0.0)
    )
    std_params = sum(p.numel() for p in std_mlp.parameters())
    print(f"\n标准 Mlp 参数量: {std_params:,}")
    print(f"MoE/Mlp 参数比: {total_params/std_params:.2f}x")
    
    # 前向传播
    moe.train()
    out = moe(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"辅助损失: {moe.aux_loss.item():.6f}")
    print(f"  - Balance Loss: {moe._balance_loss.item():.6f}")
    print(f"  - Ortho Loss: {moe._ortho_loss.item():.6f}")
    
    # 专家统计
    stats = moe.get_expert_stats()
    print(f"\n专家使用统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # 测试 collect 函数
    aux = collect_moe_aux_losses(moe)
    print(f"\ncollect_moe_aux_losses: {aux.item():.6f}")
    
    print("\n✅ MoE Layer 测试通过!")
