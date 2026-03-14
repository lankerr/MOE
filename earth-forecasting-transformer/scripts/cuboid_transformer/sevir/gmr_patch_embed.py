"""
GMR-Conv Equi-ViT Patch Embedding 模块 — Earthformer 49F 用
============================================================

核心思想:
  传统 Earthformer: 3级 stack_conv (Conv+Conv+PatchMerge) × 3 → 逐步下采样
  Equi-ViT patch embed: 2级 大步长 GMR-Conv → 一步到位嵌入
  
  这是 **架构级** 的改变，不是简单的算子替换:
  - GMR baseline (gmr_layers.py): 保持原始 3级结构，只换 Conv2d→GMR_Conv2d
  - GMR patch embed (本文件): 用 ViT 风格的大步长嵌入重新设计编码器/解码器

架构:
  编码器: 384→96 (stride=4) → 96→32 (stride=3)    总计 12× 下采样
  解码器: 32→64 (×2) → 64→128 (×2) → 128→384 (×3) 总计 12× 上采样

优势:
  1. 更少卷积层 (2层 vs 原始6层)，更快推理
  2. 大感受野 patch embedding 直接捕获中尺度特征
  3. GMR-Conv 天然连续旋转等变 → patch embedding 阶段即保持对称性
  4. PatchMerging3D (nn.Linear) 不再需要 → 去除非等变的线性下采样

使用方法:
    from gmr_patch_embed import patch_model_with_gmr_embed, count_gmr_embed_layers
    model = CuboidTransformerModel(...)
    patch_model_with_gmr_embed(model)
"""

import torch
import torch.nn as nn
from GMR_Conv import GMR_Conv2d


class GMRPatchEmbedEncoder(nn.Module):
    """Equi-ViT 风格 GMR-Conv Patch Embedding 编码器。

    2 级大步长 GMR-Conv 将 384×384 图像嵌入为 32×32 × embed_dim 特征图。
    完全替代原始的 InitialStackPatchMergingEncoder (3级 stack_conv)。

    数据流:
        (B, T, 384, 384, 1)
        → reshape → (B*T, 1, 384, 384)
        → Stage1 GMR stride=4 → (B*T, mid_dim, 96, 96)
        → Stage2 GMR stride=3 → (B*T, embed_dim, 32, 32)
        → reshape → (B, T, 32, 32, embed_dim)
    """

    def __init__(self, in_chans=1, embed_dim=64, mid_dim=16):
        super().__init__()
        # Stage 1: 384 → 96 (stride=4, k=5, pad=2)
        # floor((384 + 2*2 - 5) / 4) + 1 = floor(383/4) + 1 = 96
        self.stage1 = nn.Sequential(
            GMR_Conv2d(in_chans, mid_dim, kernel_size=5, stride=4, padding=2),
            nn.GroupNorm(4, mid_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 2: 96 → 32 (stride=3, k=3, pad=0)
        # floor((96 - 3) / 3) + 1 = 32
        self.stage2 = nn.Sequential(
            GMR_Conv2d(mid_dim, embed_dim, kernel_size=3, stride=3, padding=0),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, H, W, C_in) e.g. (B, 37, 384, 384, 1)
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)   # (B*T, C, H, W)
        x = self.stage1(x)                                     # (B*T, mid_dim, 96, 96)
        x = self.stage2(x)                                     # (B*T, embed_dim, 32, 32)
        x = x.permute(0, 2, 3, 1)                              # (B*T, 32, 32, embed_dim)
        x = self.norm(x)
        _, H2, W2, D = x.shape
        x = x.reshape(B, T, H2, W2, D)                         # (B, T, 32, 32, embed_dim)
        return x


class GMRPatchUpsampleDecoder(nn.Module):
    """Equi-ViT 风格 GMR-Conv 上采样解码器。

    3 级上采样将 32×32 × embed_dim 特征图恢复为 384×384 × out_dim。
    完全替代原始的 FinalStackUpsamplingDecoder。

    数据流:
        (B, T, 32, 32, embed_dim)
        → reshape → (B*T, embed_dim, 32, 32)
        → Up×2 + GMR → (B*T, 16, 64, 64)
        → Up×2 + GMR → (B*T, out_dim, 128, 128)
        → Up×3 + GMR → (B*T, out_dim, 384, 384)
        → reshape → (B, T, 384, 384, out_dim)

    out_dim=4 以匹配原始 dec_final_proj = Linear(4, 1)。
    """

    def __init__(self, embed_dim=64, mid_dim=16, out_dim=4):
        super().__init__()
        # Stage 1: 32 → 64 (×2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            GMR_Conv2d(embed_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, mid_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 2: 64 → 128 (×2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            GMR_Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 3: 128 → 384 (×3)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='nearest'),
            GMR_Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # x: (B, T, H, W, C) e.g. (B, 12, 32, 32, 64)
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)   # (B*T, C, H, W)
        x = self.up1(x)                                        # (B*T, 16, 64, 64)
        x = self.up2(x)                                        # (B*T, 4, 128, 128)
        x = self.up3(x)                                        # (B*T, 4, 384, 384)
        x = x.permute(0, 2, 3, 1)                              # (B*T, 384, 384, 4)
        _, H2, W2, D = x.shape
        x = x.reshape(B, T, H2, W2, D)                         # (B, T, 384, 384, 4)
        return x


def patch_model_with_gmr_embed(model, in_chans=1, embed_dim=64, mid_dim=16, out_dim=4):
    """将 CuboidTransformerModel 的编码器/解码器替换为 Equi-ViT GMR patch embedding。

    替换:
    1. initial_encoder → GMRPatchEmbedEncoder (2级 GMR stride下采样)
    2. final_decoder → GMRPatchUpsampleDecoder (3级 GMR 上采样)
    
    不替换:
    - Transformer encoder/decoder blocks (attention, FFN, PatchMerging3D)
    - dec_final_proj (Linear(4,1) → 保留)
    - decoder.upsample_layers (transformer 内部上采样)
    """
    model.initial_encoder = GMRPatchEmbedEncoder(
        in_chans=in_chans, embed_dim=embed_dim, mid_dim=mid_dim)
    model.final_decoder = GMRPatchUpsampleDecoder(
        embed_dim=embed_dim, mid_dim=mid_dim, out_dim=out_dim)

    enc_params = sum(p.numel() for p in model.initial_encoder.parameters())
    dec_params = sum(p.numel() for p in model.final_decoder.parameters())
    print(f"[GMR-PatchEmbed] 编码器: GMRPatchEmbedEncoder ({enc_params} params)")
    print(f"[GMR-PatchEmbed] 解码器: GMRPatchUpsampleDecoder ({dec_params} params)")
    print(f"[GMR-PatchEmbed] dec_final_proj 保留: {model.dec_final_proj}")
    return model


def count_gmr_embed_layers(model):
    """统计 GMR patch embed 模型中各类层的数量"""
    n_gmr = sum(1 for m in model.modules() if isinstance(m, GMR_Conv2d))
    n_conv2d = sum(1 for m in model.modules()
                   if isinstance(m, nn.Conv2d) and not isinstance(m, GMR_Conv2d))
    n_total = sum(p.numel() for p in model.parameters())
    return {"GMR_Conv2d": n_gmr, "remaining_Conv2d": n_conv2d, "total_params": n_total}
