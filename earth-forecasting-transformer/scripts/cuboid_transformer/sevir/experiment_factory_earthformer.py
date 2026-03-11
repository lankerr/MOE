import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from dataclasses import dataclass

# 借用 DATSwinLSTM 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, r"c:\Users\97290\Desktop\MOE\datswinlstm_memory")

from modules.moe_layer import MoELayer, MoEConfig
from modules.temporal_rope import apply_rotary_emb, TemporalRoPE2D

@dataclass
class ExperimentConfig:
    name: str = "baseline"
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 2
    mlp_ratio: float = 2.0
    use_swiglu: bool = False
    balance_loss_weight: float = 0.0
    ortho_loss_weight: float = 0.0
    router_jitter: float = 0.01
    use_rope: bool = False
    theta_long: float = 10000.0
    theta_short: float = 2000.0
    use_flash: bool = False
    lr: float = 1e-4

# 注意：对应关系分别是 DATSwinLSTM 的 exp7 -> exp1, exp8 -> exp2 ... exp12 -> exp6
EXPERIMENTS = {
    "exp1_moe_flash": ExperimentConfig(
        name="Exp1_Earthformer_MoE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=False, use_flash=True,
    ),
    "exp1_5_moe_balanced_flash": ExperimentConfig(
        name="Exp1.5_Earthformer_MoE_Balanced_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=False,
        balance_loss_weight=0.01, use_flash=True,
    ),
    "exp2_swiglu_moe_flash": ExperimentConfig(
        name="Exp2_Earthformer_SwiGLU_MoE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=True, use_flash=True,
    ),
    "exp3_balanced_moe_flash": ExperimentConfig(
        name="Exp3_Earthformer_Balanced_MoE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=True,
        balance_loss_weight=0.01, ortho_loss_weight=0.001, use_flash=True,
    ),
    "exp4_moe_rope_flash": ExperimentConfig(
        name="Exp4_Earthformer_MoE_RoPE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=False,
        use_rope=True, theta_long=10000.0, theta_short=2000.0, use_flash=True,
    ),
    "exp5_swiglu_moe_rope_flash": ExperimentConfig(
        name="Exp5_Earthformer_SwiGLU_MoE_RoPE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=True,
        use_rope=True, theta_long=10000.0, theta_short=2000.0, use_flash=True,
    ),
    "exp6_balanced_moe_rope_flash": ExperimentConfig(
        name="Exp6_Earthformer_Balanced_MoE_RoPE_Flash",
        use_moe=True, num_experts=4, top_k=2, use_swiglu=True,
        balance_loss_weight=0.01, ortho_loss_weight=0.001,
        use_rope=True, theta_long=10000.0, theta_short=2000.0, use_flash=True,
    ),
}

# ===================== MoE Wrapper (保留 LayerNorm + 残差) =====================

class MoEFFNWrapper(nn.Module):
    """
    包装 MoELayer，保留原 PositionwiseFFN 的 pre-LayerNorm + 残差连接。
    
    原始 PositionwiseFFN (pre_norm=True) 流程:
        residual = data
        data = layer_norm(data)
        out = activation(fc1(data))  (或 gated: act(fc1_gate(data)) * fc1(data))
        out = fc2(out) + dropout
        out = out + residual
    
    替换后:
        residual = data
        data = layer_norm(data)
        out = MoELayer(data)
        out = dropout(out)
        out = out + residual
    """
    def __init__(self, moe_layer, layer_norm, dropout_p=0.1):
        super().__init__()
        self.moe = moe_layer
        self.layer_norm = layer_norm
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, data):
        residual = data
        data = self.layer_norm(data)
        out = self.moe(data)
        out = self.dropout(out)
        out = out + residual
        return out


# ===================== MoE Injection =====================
def _replace_ffn_with_moe(module: nn.Module, config: ExperimentConfig):
    from earthformer.cuboid_transformer.cuboid_transformer import PositionwiseFFN
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
        if isinstance(child, PositionwiseFFN):
            dim = child.ffn_1.in_features
            moe_layer = MoELayer(dim=dim, config=moe_config, drop=0.0)
            # 复制原 FFN 的 LayerNorm 权重和 dropout 率
            layer_norm = child.layer_norm
            dropout_p = child.dropout_layer.p
            wrapped = MoEFFNWrapper(moe_layer, layer_norm, dropout_p)
            setattr(module, name, wrapped)
            replaced_count += 1
        else:
            replaced_count += _replace_ffn_with_moe(child, config)
    return replaced_count

# ===================== Flash & RoPE Injection =====================
def _inject_flash_and_rope(model: nn.Module, config: ExperimentConfig):
    from earthformer.cuboid_transformer.cuboid_transformer import CuboidSelfAttentionLayer, masked_softmax
    count_flash = 0
    count_rope = 0
    
    for name, module in model.named_modules():
        if isinstance(module, CuboidSelfAttentionLayer):
            if config.use_flash:
                module.use_flash = True
                count_flash += 1
            if config.use_rope:
                head_dim = module.dim // module.num_heads
                window_size = module.cuboid_size[0] * module.cuboid_size[1] * module.cuboid_size[2]
                # 简单复用 2D RoPE，实际上 Earthformer 是3D但是我们只需要对特征维度施加旋转
                # 我们这里可以将 sequence 的长度视为 window_size，并使用一维 RoPE
                from modules.temporal_rope import TemporalRoPE1D
                rope = TemporalRoPE1D(head_dim=head_dim, max_len=window_size)
                module.add_module('rope', rope)
                count_rope += 1

            # Monkey-patch forward
            if config.use_flash or config.use_rope:
                module._orig_forward = module.forward
                module.forward = functools.partial(_cuboid_attention_forward, module)
                
    return count_flash, count_rope

def _cuboid_attention_forward(self, x, global_vectors=None):
    # 当 use_global_vector=True 时，global vectors 的逻辑非常复杂（L2G, G2L, G2G 注意力），
    # Flash/RoPE 无法正确处理。直接回退到原始 forward。
    if self.use_global_vector:
        return self._orig_forward(x, global_vectors)

    from earthformer.cuboid_transformer.cuboid_transformer import update_cuboid_size_shift_size, _generalize_padding, cuboid_reorder, compute_cuboid_self_attention_mask, cuboid_reorder_reverse, masked_softmax, _generalize_unpadding
    
    x = self.norm(x)
    B, T, H, W, C_in = x.shape

    cuboid_size, shift_size = update_cuboid_size_shift_size((T, H, W), self.cuboid_size, self.shift_size, self.strategy)
    pad_t, pad_h, pad_w = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0], (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1], (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    x = _generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)
    if any(i > 0 for i in shift_size):
        shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    else:
        shifted_x = x
    reordered_x = cuboid_reorder(shifted_x, cuboid_size=cuboid_size, strategy=self.strategy)
    _, num_cuboids, cuboid_volume, _ = reordered_x.shape
    attn_mask = compute_cuboid_self_attention_mask((T, H, W), cuboid_size, shift_size=shift_size, strategy=self.strategy, padding_type=self.padding_type, device=x.device)
    
    head_C = C_in // self.num_heads
    qkv = self.qkv(reordered_x).reshape(B, num_cuboids, cuboid_volume, 3, self.num_heads, head_C).permute(3, 0, 4, 1, 2, 5)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # RoPE 注入
    if hasattr(self, 'rope'):
        q, k = self.rope(q, k, temporal_type='long')

    # Flash Attention / SDPA (仅 use_global_vector=False 时进入此函数)
    if getattr(self, 'use_flash', False) and hasattr(F, 'scaled_dot_product_attention'):
        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)]\
                .reshape(cuboid_volume, cuboid_volume, -1).permute(2, 0, 1)\
                .contiguous().unsqueeze(1)
        else:
            relative_position_bias = 0.0
            
        float_mask = relative_position_bias
        if attn_mask is not None:
            zero_mask = torch.zeros_like(attn_mask, dtype=q.dtype).unsqueeze(0)
            zero_mask = zero_mask.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))
            float_mask = float_mask + zero_mask

        q_sdpa = q.transpose(1, 2).reshape(B * num_cuboids, self.num_heads, cuboid_volume, head_C)
        k_sdpa = k.transpose(1, 2).reshape(B * num_cuboids, self.num_heads, cuboid_volume, head_C)
        v_sdpa = v.transpose(1, 2).reshape(B * num_cuboids, self.num_heads, cuboid_volume, head_C)

        if isinstance(float_mask, torch.Tensor):
            float_mask = float_mask.expand(B, self.num_heads, num_cuboids, cuboid_volume, cuboid_volume)
            float_mask = float_mask.transpose(1, 2).reshape(B * num_cuboids, self.num_heads, cuboid_volume, cuboid_volume)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True):
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=float_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        
        reordered_x = out.reshape(B, num_cuboids, self.num_heads, cuboid_volume, head_C).transpose(1, 2).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)
        
        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))

    else:
        # 手动注意力 (RoPE-only 或无加速)
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)
        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)].reshape(cuboid_volume, cuboid_volume, -1).permute(2, 0, 1).contiguous().unsqueeze(1)
            attn_score = attn_score + relative_position_bias
            
        attn_score = masked_softmax(attn_score, mask=attn_mask)
        attn_score = self.attn_drop(attn_score)
        reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)
        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))

    shifted_x = cuboid_reorder_reverse(reordered_x, cuboid_size=cuboid_size, strategy=self.strategy, orig_data_shape=(T + pad_t, H + pad_h, W + pad_w))
    if any(i > 0 for i in shift_size):
        shifted_x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
    out = _generalize_unpadding(shifted_x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
    return out

def apply_experiment(model: nn.Module, config_name: str) -> None:
    if config_name == "baseline":
        return
    config = EXPERIMENTS.get(config_name)
    if not config:
        raise ValueError(f"Unknown experiment: {config_name}")

    print(f"\n[Experiment Factory (Earthformer)] Applying {config.name} ...")
    
    if config.use_moe:
        c = _replace_ffn_with_moe(model, config)
        print(f"  -> Replaced {c} PositionwiseFFN layers with MoE")

    fc, rc = _inject_flash_and_rope(model, config)
    if config.use_flash:
        print(f"  -> Enabled Flash Attention (SDPA) on {fc} CuboidSelfAttentionLayer")
    if config.use_rope:
        print(f"  -> Injected RoPE on {rc} CuboidSelfAttentionLayer")
