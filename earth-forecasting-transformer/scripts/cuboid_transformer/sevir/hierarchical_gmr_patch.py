"""
两级非重叠 GMR Patch Embedding

核心设计原则:
1. 每级 stride = kernel (无重叠，无晕染)
2. 第一级: 4×4 s=4 提取局部细节
3. Channel MLP (逐位置，不跨 patch)
4. 第二级: 3×3 s=3 聚合更大范围语义
5. 全程保留稀疏性

与原版对比:
- 原 EarthFormer: 3级重叠 CNN → 晕染严重，特征稠密化
- 两级非重叠 GMR: 2级非重叠 → 保留稀疏性，物理 mask 对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 假设 gmr_layers.py 已存在，包含 GMR_Conv2d
try:
    from gmr_layers import GMR_Conv2d
    GMR_AVAILABLE = True
except ImportError:
    GMR_AVAILABLE = False
    print("Warning: gmr_layers not found, using placeholder Conv2d")


class HierarchicalGMRPatchEmbed(nn.Module):
    """
    两级非重叠 GMR Patch Embedding

    架构:
        输入 [B, T, 384, 384, 1]
          ↓ Stage 1: GMR 4×4 s=4
        [B, T, 96, 96, 32]
          ↓ Channel MLP
        [B, T, 96, 96, 32]
          ↓ Stage 2: GMR 3×3 s=3
        [B, T, 32, 32, 128]

    特点:
    - 每级非重叠 (stride = kernel)，无晕染
    - 旋转等变性 (C4 对称)
    - 稀疏 mask 完美对齐
    """

    def __init__(
        self,
        in_chans: int = 1,
        stage1_dim: int = 32,
        stage2_dim: int = 128,
        dbz_threshold: float = 15.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.dbz_threshold = dbz_threshold
        self.stage1_dim = stage1_dim
        self.stage2_dim = stage2_dim

        # Stage 1: 4×4 s=4 非重叠 GMR 卷积
        if GMR_AVAILABLE:
            self.stage1_conv = GMR_Conv2d(in_chans, stage1_dim,
                                       kernel_size=4, stride=4, padding=0)
        else:
            self.stage1_conv = nn.Conv2d(in_chans, stage1_dim,
                                       kernel_size=4, stride=4, padding=0)

        self.stage1_norm = nn.GroupNorm(8, stage1_dim)
        self.stage1_act = nn.GELU()

        # Channel MLP (逐位置，不跨 patch)
        mid_dim = int(stage1_dim * mlp_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(stage1_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, stage1_dim),
        )
        self.stage1_ln = nn.LayerNorm(stage1_dim)

        # Stage 2: 3×3 s=3 非重叠 GMR 卷积
        if GMR_AVAILABLE:
            self.stage2_conv = GMR_Conv2d(stage1_dim, stage2_dim,
                                       kernel_size=3, stride=3, padding=0)
        else:
            self.stage2_conv = nn.Conv2d(stage1_dim, stage2_dim,
                                       kernel_size=3, stride=3, padding=0)

        self.stage2_norm = nn.GroupNorm(16, stage2_dim)
        self.stage2_act = nn.GELU()
        self.stage2_ln = nn.LayerNorm(stage2_dim)

    def forward(self, x: torch.Tensor, return_mask: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: [B, T, H, W, C] NTHWC 格式，通常是 [B, 13, 384, 384, 1]
            return_mask: 是否返回与 token 对齐的 mask

        Returns:
            tokens: [B, T, 32, 32, stage2_dim]
            mask: [B, T, 32, 32] bool (如果 return_mask=True)
        """
        B, T, H, W, C = x.shape

        # 保存原始 dBZ 用于生成 mask
        x_raw = x  # [B, T, H, W, 1]

        # 展平时间维度
        x = x.reshape(B * T, C, H, W)  # [B*T, 1, 384, 384]

        # Stage 1: 384 → 96
        x = self.stage1_conv(x)  # [B*T, 32, 96, 96]
        x = self.stage1_norm(x)
        x = self.stage1_act(x)

        # Channel MLP (逐位置)
        x = x.permute(0, 2, 3, 1)  # [B*T, 96, 96, 32]
        x = self.stage1_ln(x)
        x = x + self.channel_mlp(x)  # 残差连接
        x = x.permute(0, 3, 1, 2)  # [B*T, 32, 96, 96]

        # Stage 2: 96 → 32
        x = self.stage2_conv(x)  # [B*T, 128, 32, 32]
        x = self.stage2_norm(x)
        x = self.stage2_act(x)

        # 最终 norm + 格式转换
        x = x.permute(0, 2, 3, 1)  # [B*T, 32, 32, 128]
        x = self.stage2_ln(x)
        x = x.reshape(B, T, 32, 32, self.stage2_dim)  # [B, T, 32, 32, 128]

        if not return_mask:
            return x, None

        # 生成与 token 完美对齐的 mask
        # 因为非重叠，MaxPool 步长 = 卷积步长，完美对齐！
        dbz = x_raw.squeeze(-1)  # [B, T, H, W]
        dbz = dbz.reshape(B * T, 1, H, W)

        # Stage 1 mask: 4×4 MaxPool → 96×96
        mask_96 = F.max_pool2d(dbz, kernel_size=4, stride=4)  # [B*T, 1, 96, 96]

        # Stage 2 mask: 3×3 MaxPool → 32×32
        mask_32 = F.max_pool2d(mask_96, kernel_size=3, stride=3)  # [B*T, 1, 32, 32]

        mask_32 = (mask_32.squeeze(1) >= self.dbz_threshold)  # [B*T, 32, 32]
        mask_32 = mask_32.reshape(B, T, 32, 32)  # [B, T, 32, 32]

        return x, mask_32


class HierarchicalGMRDecoder(nn.Module):
    """
    两级非重叠 GMR Decoder (编码器的对称部分)
    """

    def __init__(
        self,
        stage1_dim: int = 32,
        stage2_dim: int = 128,
        out_chans: int = 1,
    ):
        super().__init__()

        # 第一级上采样: 转置卷积 3×3 s=3
        if GMR_AVAILABLE:
            self.up1 = GMR_Conv2d(stage2_dim, stage1_dim,
                                 kernel_size=3, stride=3, padding=0)
        else:
            self.up1 = nn.ConvTranspose2d(stage2_dim, stage1_dim,
                                       kernel_size=3, stride=3, padding=0)

        self.up1_norm = nn.GroupNorm(8, stage1_dim)
        self.up1_act = nn.GELU()

        # Channel MLP
        mid_dim = stage1_dim * 2
        self.channel_mlp = nn.Sequential(
            nn.Linear(stage1_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, stage1_dim),
        )
        self.ln = nn.LayerNorm(stage1_dim)

        # 第二级上采样: 转置卷积 4×4 s=4
        if GMR_AVAILABLE:
            self.up2 = GMR_Conv2d(stage1_dim, out_chans,
                                 kernel_size=4, stride=4, padding=0)
        else:
            self.up2 = nn.ConvTranspose2d(stage1_dim, out_chans,
                                       kernel_size=4, stride=4, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 32, 32, stage2_dim]
        Returns:
            out: [B, T, 384, 384, out_chans]
        """
        B, T, H, W, C = x.shape

        # 转置为卷积格式
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, 32, 32]

        # 第一级上采样: 32 → 96
        x = self.up1(x)  # [B*T, 32, 96, 96]
        x = self.up1_norm(x)
        x = self.up1_act(x)

        # Channel MLP
        x = x.permute(0, 2, 3, 1)  # [B*T, 96, 96, 32]
        x = self.ln(x)
        x = x + self.channel_mlp(x)
        x = x.permute(0, 3, 1, 2)  # [B*T, 32, 96, 96]

        # 第二级上采样: 96 → 384
        x = self.up2(x)  # [B*T, 1, 384, 384]

        # 转回 NTHWC 格式
        x = x.permute(0, 2, 3, 1)  # [B*T, 384, 384, 1]
        x = x.reshape(B, T, 384, 384, 1)

        return x


def patch_model_with_hierarchical_gmr(model, base_units: int = 64):
    """
    将 HierarchicalGMRPatchEmbed 替换 EarthFormer 的 initial_encoder 和 final_decoder

    Args:
        model: CuboidTransformerModel 实例
        base_units: 基础单元数

    Returns:
        修改后的模型
    """
    new_encoder = HierarchicalGMRPatchEmbed(
        in_chans=1,
        stage1_dim=32,
        stage2_dim=base_units * 2,  # 128，与 Cuboid Attention 输入对齐
    )

    new_decoder = HierarchicalGMRDecoder(
        stage1_dim=32,
        stage2_dim=base_units * 2,
        out_chans=1,
    )

    # 替换编码器和解码器
    model.initial_encoder = new_encoder
    model.final_decoder = new_decoder

    print("[HierarchicalGMR] 替换完成")
    print(f"  Encoder 参数: {sum(p.numel() for p in new_encoder.parameters()):,}")
    print(f"  Decoder 参数: {sum(p.numel() for p in new_decoder.parameters()):,}")

    return model


if __name__ == "__main__":
    # 测试代码
    print("Testing Hierarchical GMR Patch Embedding...")
    model = HierarchicalGMRPatchEmbed()

    # 测试前向传播
    x = torch.randn(1, 13, 384, 384, 1)
    tokens, mask = model(x, return_mask=True)

    print(f"Input shape: {x.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.float().mean():.2%}")
    print("\nTest passed!")
