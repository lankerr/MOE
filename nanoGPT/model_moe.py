"""
nanoGPT + Mixture of Experts (MoE) 模型
========================================
基于 nanoGPT 原始模型扩展，融合 nanoMoE 的 MoE 组件。
支持以下变体：
  - Dense (n_exp=1): 标准 Transformer
  - Vanilla MoE: Top-K 路由，无辅助损失
  - Full MoE: Top-K 路由 + aux_loss 负载均衡 + router_z_loss 防爆
  - 不同激活函数: GELU, ReLU², SwiGLU

兼容原始 nanoGPT 的 train.py 训练循环。
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # === MoE 配置 ===
    n_exp: int = 1            # 专家数量 (1=Dense, >1=MoE)
    moe_top_k: int = 2        # 每个 token 选择的专家数
    moe_start_layer: int = 2  # 从第几层开始用 MoE (前几层保持Dense)
    moe_stride: int = 1       # 每隔几层放一个 MoE (1=每层都是MoE)

    # 辅助损失
    use_aux_loss: bool = False       # Switch Transformer 负载均衡损失
    aux_loss_weight: float = 0.01
    use_router_z_loss: bool = False  # ST-MoE z-loss 防止 logits 爆炸
    router_z_loss_weight: float = 0.001

    # 路由器设置
    use_noisy_top_k: bool = False    # 路由器加噪声
    train_capacity: float = 1.25     # 训练时容量因子
    eval_capacity: float = 2.0       # 推理时容量因子

    # 激活函数: 'gelu', 'relu2', 'swiglu'
    activation: str = 'gelu'


# ═══════════════════════════════════════════════════════════
#  辅助损失管理器 (全局收集各层的辅助损失)
# ═══════════════════════════════════════════════════════════

class AuxLossManager:
    """简易辅助损失管理器，收集各 MoE 层的辅助损失"""
    def __init__(self):
        self.losses = {}
        self.load_stats = {}  # 用于负载均衡监控

    def add(self, name, value):
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)

    def add_stats(self, name, value):
        if name not in self.load_stats:
            self.load_stats[name] = []
        self.load_stats[name].append(value)

    def get_total_loss(self, config):
        total = 0.0
        if 'aux_loss' in self.losses and config.use_aux_loss:
            total += config.aux_loss_weight * sum(self.losses['aux_loss']) / len(self.losses['aux_loss'])
        if 'router_z_loss' in self.losses and config.use_router_z_loss:
            total += config.router_z_loss_weight * sum(self.losses['router_z_loss']) / len(self.losses['router_z_loss'])
        return total

    def get_load_balance_info(self):
        """返回各层的负载均衡信息"""
        return self.load_stats.copy()

    def reset(self):
        self.losses.clear()
        self.load_stats.clear()

# 全局实例
AUX_MANAGER = AuxLossManager()


# ═══════════════════════════════════════════════════════════
#  激活函数
# ═══════════════════════════════════════════════════════════

class ReLUSquared(nn.Module):
    """ReLU² 激活: ReLU(x)²"""
    def forward(self, x):
        return F.relu(x).square()


# ═══════════════════════════════════════════════════════════
#  MLP 变体
# ═══════════════════════════════════════════════════════════

class MLP(nn.Module):
    """标准 MLP (单个专家或 Dense 层)"""
    def __init__(self, config):
        super().__init__()
        act = config.activation
        if act == 'swiglu':
            # SwiGLU: gate_proj + up_proj -> activation * gate -> down_proj
            self.gate_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.act = nn.SiLU()
            self.is_swiglu = True
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            if act == 'relu2':
                self.act = ReLUSquared()
            else:  # gelu
                self.act = nn.GELU()
            self.is_swiglu = False
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.is_swiglu:
            x = self.act(self.gate_proj(x)) * self.c_fc(x)
        else:
            x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MLPExperts(nn.Module):
    """批量化的多专家 MLP (BMM实现，高效)"""
    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
        self.n_embd = config.n_embd
        self.intermediate = 4 * config.n_embd
        self.activation = config.activation

        if config.activation == 'swiglu':
            self.gate_proj = nn.Parameter(torch.empty(config.n_exp, config.n_embd, self.intermediate))
            self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, self.intermediate))
            self.c_proj = nn.Parameter(torch.empty(config.n_exp, self.intermediate, config.n_embd))
            self.act = nn.SiLU()
        else:
            self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, self.intermediate))
            self.c_proj = nn.Parameter(torch.empty(config.n_exp, self.intermediate, config.n_embd))
            if config.activation == 'relu2':
                self.act = ReLUSquared()
            else:
                self.act = nn.GELU()

        # 初始化
        for p in [self.c_fc, self.c_proj]:
            nn.init.normal_(p, mean=0.0, std=0.02)
        if config.activation == 'swiglu':
            nn.init.normal_(self.gate_proj, mean=0.0, std=0.02)

    def forward(self, x):
        """x: [n_exp, capacity, n_embd] -> [n_exp, capacity, n_embd]"""
        if self.activation == 'swiglu':
            gate_out = torch.bmm(x, self.gate_proj)
            fc_out = torch.bmm(x, self.c_fc)
            x = self.act(gate_out) * fc_out
        else:
            x = torch.bmm(x, self.c_fc)
            x = self.act(x)
        x = torch.bmm(x, self.c_proj)
        return x


# ═══════════════════════════════════════════════════════════
#  路由器
# ═══════════════════════════════════════════════════════════

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        self.n_exp = config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity

        # 路由权重 (无 bias)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        if self.use_noisy_top_k:
            self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        num_tokens = B * T
        x_flat = x.view(num_tokens, C)

        # 1. 路由 logits
        logits = self.w_g(x_flat)  # [B*T, n_exp]
        if self.training and self.use_noisy_top_k:
            noise = F.softplus(self.w_noise(x_flat))
            noise *= torch.randn_like(noise)
            logits += noise

        # 2. 辅助损失
        if self.training and self.use_router_z_loss:
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            AUX_MANAGER.add("router_z_loss", z_loss)

        # 3. Top-K 选择
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        if self.training and self.use_aux_loss:
            # Switch Transformer 辅助损失: 鼓励负载均衡
            # 统一在 float32 下计算，避免 bf16 autocast 下 scatter dtype 不匹配
            logits_f32 = logits.float()
            all_probs = torch.zeros_like(logits_f32)
            top_k_softmax = F.softmax(top_k_logits.float(), dim=-1)
            all_probs.scatter_(-1, top_k_indices, top_k_softmax)
            aux_loss = self._compute_aux_loss(all_probs, top_k_indices)
            AUX_MANAGER.add("aux_loss", aux_loss)

        # 4. 路由概率 (在 top-k 上归一化)
        router_probs = F.softmax(top_k_logits, dim=-1)

        # 5. 容量限制 + 排名
        exp_capacity = self._get_capacity(num_tokens)
        expert_mask = F.one_hot(top_k_indices, self.n_exp)  # [B*T, k, n_exp]

        # 计算每个 expert 的 cumsum 确定排名
        reshaped = expert_mask.permute(1, 0, 2).reshape(self.top_k * num_tokens, self.n_exp)
        cumsum = torch.cumsum(reshaped, dim=0)
        position = cumsum.reshape(self.top_k, num_tokens, self.n_exp).permute(1, 0, 2)
        rank = (position - 1) * expert_mask

        # 容量过滤
        capacity_mask = rank < exp_capacity
        final_mask = expert_mask * capacity_mask
        probs_mask = (final_mask.sum(dim=-1) > 0)
        router_probs_masked = router_probs * probs_mask
        final_rank = torch.sum(rank, dim=-1)

        # 记录负载统计
        if not self.training:
            with torch.no_grad():
                expert_counts = torch.bincount(top_k_indices.view(-1), minlength=self.n_exp).float()
                AUX_MANAGER.add_stats("expert_counts", expert_counts.cpu())

        return final_mask, router_probs_masked, top_k_indices, final_rank, exp_capacity

    def _compute_aux_loss(self, probs, indices):
        """Switch Transformer 辅助损失 (鼓励均匀分配)"""
        one_hot = F.one_hot(indices, self.n_exp).float().sum(dim=1)  # [B*T, n_exp]
        tokens_per_expert = one_hot.mean(dim=0)
        prob_per_expert = probs.mean(dim=0)
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)

    def _get_capacity(self, num_tokens):
        cap_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = int(self.top_k * cap_factor * num_tokens / self.n_exp)
        capacity = max(capacity, 4)
        return capacity


# ═══════════════════════════════════════════════════════════
#  MoE 层
# ═══════════════════════════════════════════════════════════

class MOELayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.router = Router(config)
        self.experts = MLPExperts(config)
        self.n_exp = config.n_exp
        self.top_k = config.moe_top_k
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(config.dropout)
        # 用于专家分析：记录每个专家被选中的次数
        self.register_buffer('expert_selection_counts', torch.zeros(config.n_exp), persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        final_mask, router_probs, top_k_indices, rank, exp_capacity = self.router(x)

        x_flat = x.view(B * T, C)
        expert_inputs = torch.zeros(self.n_exp, exp_capacity, C, dtype=x.dtype, device=x.device)

        # Dispatch tokens to experts
        flat_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_idx = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        valid = flat_rank < exp_capacity
        v_tokens = flat_token_idx[valid]
        v_experts = flat_indices[valid]
        v_ranks = flat_rank[valid]

        expert_inputs[v_experts, v_ranks] = x_flat[v_tokens]

        # Run all experts (batched)
        expert_outputs = self.experts(expert_inputs)

        # Gather & combine
        output_flat = torch.zeros_like(x_flat)
        gated_out = expert_outputs[v_experts, v_ranks]
        v_probs = router_probs.view(-1)[valid].unsqueeze(1)
        weighted_out = gated_out * v_probs
        output_flat.scatter_add_(0, v_tokens.unsqueeze(1).expand_as(weighted_out), weighted_out)

        # 更新专家选择统计
        with torch.no_grad():
            counts = torch.bincount(flat_indices, minlength=self.n_exp).float()
            self.expert_selection_counts += counts

        output = self.dropout(output_flat.view(B, T, C))
        return output


# ═══════════════════════════════════════════════════════════
#  Transformer 层
# ═══════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config, use_moe=False, layer_idx=0):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.use_moe = use_moe
        if use_moe:
            self.mlp = MOELayer(config, layer_idx=layer_idx)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ═══════════════════════════════════════════════════════════
#  GPT 主模型
# ═══════════════════════════════════════════════════════════

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 构建 Transformer 块
        blocks = []
        for i in range(config.n_layer):
            if config.n_exp > 1:
                use_moe = (i >= config.moe_start_layer) and ((i - config.moe_start_layer) % config.moe_stride == 0)
            else:
                use_moe = False
            blocks.append(Block(config, use_moe=use_moe, layer_idx=i))

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(blocks),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # 初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 统计
        total_params = self.get_num_params()
        active_params = self.get_num_active_params()
        moe_layers = sum(1 for b in blocks if b.use_moe)
        print(f"总参数: {total_params/1e6:.2f}M | 活跃参数: {active_params/1e6:.2f}M")
        print(f"层数: {config.n_layer} | MoE层: {moe_layers} | 专家: {config.n_exp} | Top-K: {config.moe_top_k}")
        print(f"激活函数: {config.activation} | 辅助损失: aux={config.use_aux_loss} z={config.use_router_z_loss}")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_active_params(self):
        """MoE 模型中每次前向传播实际使用的参数量"""
        n = 0
        seen = set()
        for name, p in self.named_parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if 'experts' in name and self.config.n_exp > 1:
                n += p.numel() * self.config.moe_top_k / self.config.n_exp
            else:
                n += p.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 每次前向前重置辅助损失
        AUX_MANAGER.reset()

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # 加入辅助损失
            aux_loss = AUX_MANAGER.get_total_loss(self.config)
            loss = lm_loss + aux_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_expert_stats(self):
        """获取所有 MoE 层的专家选择统计"""
        stats = {}
        for i, block in enumerate(self.transformer.h):
            if block.use_moe:
                counts = block.mlp.expert_selection_counts
                total = counts.sum()
                if total > 0:
                    pct = (counts / total * 100).cpu().tolist()
                    stats[f"layer_{i}"] = {
                        'counts': counts.cpu().tolist(),
                        'pct': pct,
                        'balance': min(pct) / max(pct) if max(pct) > 0 else 0,
                    }
        return stats

    def reset_expert_stats(self):
        """重置专家选择统计"""
        for block in self.transformer.h:
            if block.use_moe:
                block.mlp.expert_selection_counts.zero_()

    def ablate_expert(self, layer_idx, expert_idx):
        """消融指定层的指定专家 (将其权重清零)"""
        block = self.transformer.h[layer_idx]
        if not block.use_moe:
            print(f"Layer {layer_idx} is not MoE!")
            return
        experts = block.mlp.experts
        with torch.no_grad():
            experts.c_fc[expert_idx].zero_()
            experts.c_proj[expert_idx].zero_()
            if hasattr(experts, 'gate_proj') and isinstance(experts.gate_proj, nn.Parameter):
                experts.gate_proj[expert_idx].zero_()
        print(f"Ablated expert {expert_idx} in layer {layer_idx}")
