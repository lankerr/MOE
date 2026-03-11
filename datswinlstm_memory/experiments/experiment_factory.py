"""
实验变体工厂: 为 DATSwinLSTM-Memory 创建12种实验配置
====================================================

通过「后处理注入」策略，在模型构建后替换/增强指定模块:
- MoE 注入: 替换 DATSwinTransformerBlock.mlp 和 SwinTransformerBlock.mlp
- RoPE 注入: 增强 WindowAttention.forward 和 DATSwinDAttention.forward
- Flash Attention: 启用 SDPA (Memory Attention + WindowAttention)

12 个实验 (6 基础 × 2 Flash):
- Exp1:  MoE (GELU, Top-2, 4专家)
- Exp2:  SwiGLU-MoE (Qwen风格门控, Top-2, 4专家)  
- Exp3:  MoE + Load Balance + Orthogonalization
- Exp4:  Exp1 + Temporal RoPE
- Exp5:  Exp2 + Temporal RoPE
- Exp6:  Exp3 + Temporal RoPE
- Exp7:  Exp1 + Flash Attention
- Exp8:  Exp2 + Flash Attention
- Exp9:  Exp3 + Flash Attention
- Exp10: Exp4 + Flash Attention (MoE + RoPE + Flash)
- Exp11: Exp5 + Flash Attention (SwiGLU-MoE + RoPE + Flash)
- Exp12: Exp6 + Flash Attention (Balanced-MoE + RoPE + Flash)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import functools

# 导入自定义模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.moe_layer import MoELayer, MoEConfig, collect_moe_aux_losses, get_all_expert_stats
from modules.swiglu import SwiGLUFFN
from modules.temporal_rope import (
    TemporalRoPE2D, TemporalRoPE1D, 
    create_rope_for_window_attention, create_rope_for_memory_attention,
    apply_rotary_emb
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "baseline"
    
    # MoE 配置
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 2
    mlp_ratio: float = 2.0
    use_swiglu: bool = False       # False=GELU, True=SwiGLU
    
    # 辅助损失
    balance_loss_weight: float = 0.0
    ortho_loss_weight: float = 0.0
    router_jitter: float = 0.01
    
    # RoPE 配置
    use_rope: bool = False
    theta_long: float = 10000.0    # 长期记忆θ (低频)
    theta_short: float = 2000.0    # 短期记忆θ (高频)
    
    # Flash Attention 配置
    use_flash: bool = False         # 启用 SDPA (Attention + WindowAttention)
    
    # 训练参数
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 1
    
    def __repr__(self):
        parts = [f"Exp: {self.name}"]
        if self.use_moe:
            act = "SwiGLU" if self.use_swiglu else "GELU"
            parts.append(f"MoE({self.num_experts}E, top{self.top_k}, {act})")
        if self.balance_loss_weight > 0:
            parts.append(f"BalLoss={self.balance_loss_weight}")
        if self.ortho_loss_weight > 0:
            parts.append(f"OrthoLoss={self.ortho_loss_weight}")
        if self.use_rope:
            parts.append(f"RoPE(thetaL={self.theta_long}, thetaS={self.theta_short})")
        if self.use_flash:
            parts.append("FlashAttn(SDPA)")
        return " | ".join(parts)


# ===================== 6 个预设实验 =====================

EXPERIMENTS = {
    "exp1_moe": ExperimentConfig(
        name="Exp1_MoE_GELU",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=False,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
    ),
    "exp2_swiglu_moe": ExperimentConfig(
        name="Exp2_SwiGLU_MoE",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
    ),
    "exp3_balanced_moe": ExperimentConfig(
        name="Exp3_Balanced_MoE",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.01,
        ortho_loss_weight=0.001,
    ),
    "exp4_moe_rope": ExperimentConfig(
        name="Exp4_MoE_GELU_RoPE",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=False,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
    ),
    "exp5_swiglu_moe_rope": ExperimentConfig(
        name="Exp5_SwiGLU_MoE_RoPE",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
    ),
    "exp6_balanced_moe_rope": ExperimentConfig(
        name="Exp6_Balanced_MoE_RoPE",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.01,
        ortho_loss_weight=0.001,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
    ),
    
    # ---- Exp7~12: Flash Attention 版 (与 Exp1~6 并列对比) ----
    
    "exp7_moe_flash": ExperimentConfig(
        name="Exp7_MoE_GELU_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=False,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_flash=True,
    ),
    "exp8_swiglu_moe_flash": ExperimentConfig(
        name="Exp8_SwiGLU_MoE_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_flash=True,
    ),
    "exp9_balanced_moe_flash": ExperimentConfig(
        name="Exp9_Balanced_MoE_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.01,
        ortho_loss_weight=0.001,
        use_flash=True,
    ),
    "exp10_moe_rope_flash": ExperimentConfig(
        name="Exp10_MoE_GELU_RoPE_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=False,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
        use_flash=True,
    ),
    "exp11_swiglu_moe_rope_flash": ExperimentConfig(
        name="Exp11_SwiGLU_MoE_RoPE_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.0,
        ortho_loss_weight=0.0,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
        use_flash=True,
    ),
    "exp12_balanced_moe_rope_flash": ExperimentConfig(
        name="Exp12_Balanced_MoE_RoPE_Flash",
        use_moe=True,
        num_experts=4,
        top_k=2,
        use_swiglu=True,
        balance_loss_weight=0.01,
        ortho_loss_weight=0.001,
        use_rope=True,
        theta_long=10000.0,
        theta_short=2000.0,
        use_flash=True,
    ),
}


# ===================== MoE 注入 =====================

def _replace_mlp_with_moe(module: nn.Module, config: ExperimentConfig):
    """
    递归查找所有 Mlp 实例，替换为 MoE 层
    
    目标:
    - DATSwinTransformerBlock.mlp (dat_blocks.py)
    - SwinTransformerBlock.mlp (DATSwinLSTM_D_Memory.py)
    - Memory 中的 FeedForward (可选)
    """
    moe_config = MoEConfig(
        num_experts=config.num_experts,
        top_k=config.top_k,
        mlp_ratio=config.mlp_ratio,
        use_swiglu=config.use_swiglu,
        balance_loss_weight=config.balance_loss_weight,
        ortho_loss_weight=config.ortho_loss_weight,
        router_jitter=config.router_jitter,
    )
    
    replaced_count = 0
    
    for name, child in module.named_children():
        # 检查是否是 Mlp (通过属性判断，多种 Mlp 类可能存在)
        is_mlp = (hasattr(child, 'fc1') and hasattr(child, 'fc2') and 
                  hasattr(child, 'act') and isinstance(child, nn.Module))
        
        # 检查是否是 FeedForward (Memory 中的)
        is_ff = (hasattr(child, 'net') and isinstance(child.net, nn.Sequential) and
                 child.__class__.__name__ == 'FeedForward')
        
        if (is_mlp or is_ff) and name == 'mlp':
            # 获取维度信息
            if is_mlp:
                dim = child.fc1.in_features
            else:  # FeedForward
                # 找到 Sequential 中第一个 Linear
                for layer in child.net:
                    if isinstance(layer, nn.Linear):
                        dim = layer.in_features
                        break
            
            # 创建 MoE 替代
            moe_layer = MoELayer(dim=dim, config=moe_config, drop=0.0)
            setattr(module, name, moe_layer)
            replaced_count += 1
        else:
            # 递归子模块
            replaced_count += _replace_mlp_with_moe(child, config)
    
    return replaced_count


# ===================== Flash Attention 注入 =====================

def _inject_flash_attention(model: nn.Module, enable: bool = True) -> int:
    """
    设置模型中所有 Attention / WindowAttention 的 use_flash 标志
    
    目标模块:
    - Attention (Memory cross-attention): 已内置 SDPA 路径
    - WindowAttention (Swin self-attention): 已内置 SDPA 路径
    - DATSwinDAttention: 不转换 (Q/K 空间维度不对称)
    
    Args:
        model: Memory 模型实例
        enable: True=启用 SDPA, False=使用原始手动 attention
    
    Returns:
        被设置的 Attention 模块数量
    """
    count = 0
    has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        
        if cls_name == 'Attention' and hasattr(module, 'use_flash'):
            # Memory 中的 cross-attention
            module.use_flash = enable and has_sdpa
            count += 1
            
        elif cls_name == 'WindowAttention' and hasattr(module, 'use_flash'):
            # Swin Transformer 的 window self-attention
            module.use_flash = enable and has_sdpa
            count += 1
    
    return count


# ===================== RoPE 注入 =====================

def _inject_rope_to_window_attention(model: nn.Module, config: ExperimentConfig):
    """
    为 WindowAttention 和 DATSwinDAttention 注入 Temporal RoPE
    
    策略: monkey-patch forward 方法, 在 Q*K 之前对 Q/K 施加旋转
    """
    rope_modules = {}  # 存储创建的 RoPE 模块
    
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'WindowAttention':
            head_dim = module.dim // module.num_heads
            window_size = module.window_size[0] if isinstance(module.window_size, tuple) else module.window_size
            
            rope = create_rope_for_window_attention(
                head_dim=head_dim,
                window_size=window_size,
                theta_long=config.theta_long,
                theta_short=config.theta_short
            )
            
            # 作为子模块注册 (确保 parameters/buffers 被追踪)
            module.add_module('rope', rope)
            
            # 保存原始 forward
            module._orig_forward = module.forward
            
            # Monkey-patch forward
            module.forward = functools.partial(_window_attention_with_rope, module)
            rope_modules[name] = rope
        
        elif module.__class__.__name__ == 'DATSwinDAttention':
            head_dim = module.n_head_channels
            window_size = module.window_size[0] if isinstance(module.window_size, tuple) else module.window_size
            
            rope = create_rope_for_window_attention(
                head_dim=head_dim,
                window_size=window_size,
                theta_long=config.theta_long,
                theta_short=config.theta_short
            )
            
            module.add_module('rope', rope)
            module._orig_forward = module.forward
            module.forward = functools.partial(_dat_attention_with_rope, module)
            rope_modules[name] = rope
    
    return rope_modules


def _window_attention_with_rope(self, x, mask=None, temporal_type='short', timestep=0):
    """
    增强版 WindowAttention.forward: 加入 RoPE (+ 可选 Flash Attention)
    
    原始流程: QKV → split → Q*scale → Q@K^T → +bias → softmax → @V → proj
    增强流程: QKV → split → **RoPE(Q,K)** → Attention(手动 or SDPA) → proj
    """
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # === RoPE 注入点 ===
    if hasattr(self, 'rope'):
        q, k = self.rope(q, k, temporal_type=temporal_type, timestep=timestep)
    
    # === 构建 relative position bias (两条路径共用) ===
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.contiguous().view(-1)
    ].view(
        self.window_size[0] * self.window_size[1], 
        self.window_size[0] * self.window_size[1], -1
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
    
    # === Flash Attention (SDPA) 路径 ===
    if getattr(self, 'use_flash', False):
        # attn_bias: (B_, nH, N, N)  —— 包含 relative position bias + window mask
        attn_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)
        
        if mask is not None:
            nW = mask.shape[0]
            attn_bias = attn_bias.contiguous().view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn_bias = attn_bias.contiguous().view(-1, self.num_heads, N, N)
        
        drop_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=drop_p, is_causal=False
        )
        x = x.transpose(1, 2).reshape(B_, N, C)
    else:
        # === 原始手动 Attention 路径 ===
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.contiguous().view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.contiguous().view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _dat_attention_with_rope(self, x, window_size, mask=None, 
                              temporal_type='short', timestep=0):
    """
    增强版 DATSwinDAttention.forward: 加入 RoPE
    
    DATSwin 使用 Conv2D 做 Q/K/V projection，然后做 deformable sampling。
    RoPE 在 Q/K reshape 后、attn 计算前注入。
    """
    H = window_size
    W = window_size
    B, N, C = x.size()
    dtype, device = x.dtype, x.device
    
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # b (h w) c -> b c h w
    
    # 计算 query
    q = self.proj_q(x)
    q_off = q.reshape(B * self.n_group, self.n_group_channels, H, W)
    
    # 计算 offset
    offset = self.conv_offset(q_off)
    Hk, Wk = offset.size(2), offset.size(3)
    n_sample = Hk * Wk
    
    if self.offset_range_factor > 0 and not self.no_off:
        offset_range = torch.tensor([1.0 / (Hk-1), 1.0 / (Wk-1)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
    
    offset = offset.permute(0, 2, 3, 1)  # b p h w -> b h w p
    reference = self._get_ref_points(Hk, Wk, B, dtype, device)
    
    if self.no_off:
        offset = torch.zeros_like(offset)
    
    if self.offset_range_factor >= 0:
        pos = offset + reference
    else:
        pos = (offset + reference).clamp(-1, +1)
    
    x_sampled = F.grid_sample(
        input=x.reshape(B * self.n_group, self.n_group_channels, H, W),
        grid=pos[..., (1, 0)],
        mode='bilinear', align_corners=True
    )
    x_sampled = x_sampled.reshape(B, C, 1, n_sample)
    
    # Q/K/V
    q = q.reshape(B * self.n_head, self.n_head_channels, H * W)
    k = self.proj_k(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
    v = self.proj_v(x_sampled).reshape(B * self.n_head, self.n_head_channels, n_sample)
    
    # === RoPE 注入 ===
    # q: (B*nH, head_dim, HW) → (B*nH, HW, head_dim) → rope → 转回
    if hasattr(self, 'rope'):
        q_t = q.permute(0, 2, 1).unsqueeze(1)  # (B*nH, 1, HW, head_dim)
        k_t = k.permute(0, 2, 1).unsqueeze(1)  # (B*nH, 1, n_sample, head_dim)
        
        # 对 Q 用空间+时间 RoPE
        q_spatial_freqs = self.rope.get_spatial_freqs(H * W, device)
        t_freq = self.rope.get_temporal_freqs(timestep, temporal_type, device)
        
        # 组合频率
        s_cos = q_spatial_freqs[..., 0]
        s_sin = q_spatial_freqs[..., 1]
        t_cos = t_freq[..., 0]
        t_sin = t_freq[..., 1]
        combined_cos = s_cos * t_cos - s_sin * t_sin
        combined_sin = s_sin * t_cos + s_cos * t_sin
        combined_freqs = torch.stack([combined_cos, combined_sin], dim=-1)
        
        q_t = apply_rotary_emb(q_t, combined_freqs)
        
        # K 可能经过 deformable sampling，空间位置已变，只用时间 RoPE
        k_temporal_freqs = self.rope.get_temporal_freqs(timestep, temporal_type, device)
        from modules.temporal_rope import precompute_freqs_cis
        k_1d = precompute_freqs_cis(self.n_head_channels, n_sample, 
                                     self.rope.theta_short if temporal_type=='short' else self.rope.theta_long,
                                     device=device)
        k_t = apply_rotary_emb(k_t, k_1d)
        
        q = q_t.squeeze(1).permute(0, 2, 1)  # back to (B*nH, head_dim, HW)
        k = k_t.squeeze(1).permute(0, 2, 1)
    
    # Attention
    attn = torch.einsum('b c m, b c n -> b m n', q, k)
    attn = attn.mul(self.scale)
    
    # Position encoding
    if self.use_pe:
        if self.dwc_pe:
            residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                B * self.n_head, self.n_head_channels, H * W)
        elif self.fixed_pe:
            rpe_table = self.rpe_table
            attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            attn = attn + attn_bias.reshape(B * self.n_head, H * W, n_sample)
        else:
            rpe_table = self.rpe_table
            rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            q_grid = self._get_ref_points(H, W, B, dtype, device)
            displacement = (
                q_grid.reshape(B * self.n_group, H * W, 2).unsqueeze(2)
                - pos.reshape(B * self.n_group, n_sample, 2).unsqueeze(1)
            ).mul(0.5)
            attn_bias = F.grid_sample(
                input=rpe_bias.reshape(B * self.n_group, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                grid=displacement[..., (1, 0)],
                mode='bilinear', align_corners=True
            )
            attn_bias = attn_bias.reshape(B * self.n_head, H * W, n_sample)
            attn = attn + attn_bias
    
    if mask is not None:
        attn = attn.view(-1, self.n_head, H * W, n_sample)
        nW = mask.shape[0]
        attn = attn.view(B // nW, nW, self.n_head, H * W, n_sample) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.n_head, H * W, n_sample)
        attn = attn.view(-1, H * W, n_sample)
    
    attn = F.softmax(attn, dim=2)
    attn = self.attn_drop(attn)
    out = torch.einsum('b m n, b c n -> b c m', attn, v)
    
    if self.use_pe and self.dwc_pe:
        out = out + residual_lepe
    out = out.reshape(B, C, H, W)
    
    y = self.proj_drop(self.proj_out(out))
    
    return y, pos.reshape(B, self.n_group, Hk, Wk, 2), reference.reshape(B, self.n_group, Hk, Wk, 2)


# ===================== 模型变体工厂 =====================

def apply_experiment(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    """
    将实验配置应用到模型上
    
    按顺序执行:
    1. Flash Attention 设置 (总是执行, 显式控制 use_flash)
    2. MoE 替换 (如果启用)
    3. RoPE 注入 (如果启用)
    
    Args:
        model: 原始 Memory 模型实例
        config: 实验配置
    
    Returns:
        修改后的模型 (就地修改)
    """
    print(f"\n{'='*60}")
    print(f"应用实验配置: {config}")
    print(f"{'='*60}")
    
    # 1) Flash Attention — 总是显式设置, 确保实验间一致性
    n_flash = _inject_flash_attention(model, enable=config.use_flash)
    if config.use_flash:
        print(f"[OK] Flash Attention (SDPA): {n_flash} 个 Attention 模块已启用")
    else:
        print(f"[OFF] Flash Attention: 已禁用 ({n_flash} 个模块设为手动 attention)")
    
    # 2) MoE 替换
    if config.use_moe:
        n_replaced = _replace_mlp_with_moe(model, config)
        print(f"[OK] MoE 替换: {n_replaced} 个 Mlp -> MoELayer")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        moe_params = sum(
            sum(p.numel() for p in m.parameters()) 
            for m in model.modules() if isinstance(m, MoELayer)
        )
        print(f"  总参数: {total_params:,} | MoE参数: {moe_params:,} ({100*moe_params/total_params:.1f}%)")
    
    # 3) RoPE 注入
    if config.use_rope:
        rope_modules = _inject_rope_to_window_attention(model, config)
        print(f"[OK] RoPE 注入: {len(rope_modules)} 个 Attention 模块")
        print(f"  theta_long={config.theta_long} (低频), theta_short={config.theta_short} (高频)")
    
    # 总参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n最终模型: {total_params:,} 总参数 | {trainable_params:,} 可训练")
    
    return model


def create_experiment_model(model_args, exp_name: str, 
                             memory_channel_size=512, short_len=12, long_len=36):
    """
    创建带实验配置的模型
    
    Args:
        model_args: argparse.Namespace 模型参数
        exp_name: 实验名称 (exp1_moe, exp2_swiglu_moe, etc.)
        memory_channel_size: memory 通道数
        short_len: 短期长度
        long_len: 长期长度
    
    Returns:
        model: 配置好的模型
        config: 使用的实验配置
    """
    from models.DATSwinLSTM_D_Memory import Memory
    
    config = EXPERIMENTS[exp_name]
    
    # 创建基础模型
    model = Memory(
        model_args,
        memory_channel_size=memory_channel_size,
        short_len=short_len,
        long_len=long_len
    )
    
    # 应用实验配置
    model = apply_experiment(model, config)
    
    return model, config


# ===================== 训练辅助 =====================

def compute_total_loss(pred_loss: torch.Tensor, model: nn.Module, 
                        config: ExperimentConfig) -> Tuple[torch.Tensor, dict]:
    """
    计算总损失 = 预测损失 + MoE 辅助损失
    
    Args:
        pred_loss: 预测任务的核心损失 (如 L1/MSE)
        model: 含 MoE 的模型
        config: 实验配置
    
    Returns:
        total_loss: 总损失
        loss_dict: 分项损失字典 (用于日志)
    """
    loss_dict = {'pred_loss': pred_loss.item()}
    total_loss = pred_loss
    
    if config.use_moe and (config.balance_loss_weight > 0 or config.ortho_loss_weight > 0):
        aux_loss = collect_moe_aux_losses(model)
        total_loss = total_loss + aux_loss
        loss_dict['aux_loss'] = aux_loss.item()
    
    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


def get_experiment_expert_stats(model: nn.Module) -> dict:
    """获取所有 MoE 层的专家统计"""
    return get_all_expert_stats(model)


if __name__ == "__main__":
    """列出所有实验配置"""
    print("=" * 60)
    print("DATSwinLSTM-Memory MoE+RoPE+Flash 实验配置 (12个)")
    print("=" * 60)
    
    for key, config in EXPERIMENTS.items():
        print(f"\n{key}:")
        print(f"  {config}")
        features = []
        if config.use_moe:
            features.append("MoE")
        if config.use_swiglu:
            features.append("SwiGLU")
        if config.balance_loss_weight > 0:
            features.append("LoadBalance")
        if config.ortho_loss_weight > 0:
            features.append("Orthogonal")
        if config.use_rope:
            features.append("TemporalRoPE")
        if config.use_flash:
            features.append("FlashAttn")
        print(f"  特性: {' + '.join(features)}")
