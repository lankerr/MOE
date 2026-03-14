"""
E2CNN 等变卷积替换模块 — Earthformer 49F 用
=============================================
将 CuboidTransformerModel 的 InitialStackPatchMergingEncoder 和
FinalStackUpsamplingDecoder 中的 nn.Conv2d 替换为 e2cnn R2Conv (C4 旋转等变)。

Transformer (attention/FFN) 部分完全不改 — 只改 Conv 前/后端。

使用方法:
    from equivariant_layers import patch_model_with_e2cnn
    model = CuboidTransformerModel(...)
    patch_model_with_e2cnn(model)  # 原地替换 Conv2d → R2Conv wrapper
"""

import torch
import torch.nn as nn
import warnings
from e2cnn import gspaces, nn as enn


# C4 旋转群 (90°/180°/270°/360° 旋转对称)
GSPACE = gspaces.Rot2dOnR2(N=4)


def _make_field_type(channels):
    """为给定通道数创建 FieldType。
    channels 必须是 4 的倍数 (C4 regular repr dim=4)，
    除了 channels=1 时用 trivial repr。
    """
    if channels == 1:
        return enn.FieldType(GSPACE, [GSPACE.trivial_repr])
    assert channels % 4 == 0, f"channels={channels} 不是 4 的倍数，无法用 C4 regular_repr"
    return enn.FieldType(GSPACE, [GSPACE.regular_repr] * (channels // 4))


class EquivariantConv2dWrapper(nn.Module):
    """将 e2cnn R2Conv 包装为与 nn.Conv2d 完全兼容的接口。
    
    输入/输出都是普通 PyTorch tensor (B, C, H, W)。
    内部自动 wrap/unwrap GeometricTensor。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_type = _make_field_type(in_channels)
        self.out_type = _make_field_type(out_channels)
        self.conv = enn.R2Conv(
            self.in_type, self.out_type,
            kernel_size=kernel_size, padding=padding,
        )

    def forward(self, x):
        # x: (B, C, H, W) 普通 tensor
        geo_x = enn.GeometricTensor(x, self.in_type)
        geo_out = self.conv(geo_x)
        return geo_out.tensor  # 回到普通 tensor


class EquivariantGroupNormWrapper(nn.Module):
    """将 e2cnn InnerBatchNorm 包装为替代 GroupNorm 的接口。"""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.field_type = _make_field_type(num_channels)
        self.norm = enn.InnerBatchNorm(self.field_type)

    def forward(self, x):
        geo_x = enn.GeometricTensor(x, self.field_type)
        geo_out = self.norm(geo_x)
        return geo_out.tensor


def _replace_conv_block(conv_block_seq, activation_name='leaky'):
    """替换 nn.Sequential(Conv2d, GroupNorm, Act, Conv2d, GroupNorm, Act, ...)
    中的 Conv2d → EquivariantConv2dWrapper, GroupNorm → EquivariantGroupNormWrapper。
    Activation 保持不变。
    """
    new_layers = []
    for layer in conv_block_seq:
        if isinstance(layer, nn.Conv2d):
            new_layers.append(EquivariantConv2dWrapper(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size[0],
                padding=layer.padding[0],
            ))
        elif isinstance(layer, nn.GroupNorm):
            new_layers.append(EquivariantGroupNormWrapper(
                num_groups=layer.num_groups,
                num_channels=layer.num_channels,
            ))
        else:
            # 保持 activation 不变 (LeakyReLU 等 pointwise 操作对 regular repr 是等变的)
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def _replace_upsample_conv(upsample_layer):
    """替换 Upsample3DLayer 内部的 self.conv (nn.Conv2d)。"""
    old_conv = upsample_layer.conv
    upsample_layer.conv = EquivariantConv2dWrapper(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size[0],
        padding=old_conv.padding[0],
    )


def patch_model_with_e2cnn(model):
    """原地将 CuboidTransformerModel 的 Conv2d 替换为 e2cnn 等变卷积。
    
    替换范围:
    1. initial_encoder (InitialStackPatchMergingEncoder) 的 conv_block_list
    2. final_decoder (FinalStackUpsamplingDecoder) 的 conv_block_list + upsample_list
    3. decoder 内的 upsample_layers
    
    不替换:
    - Transformer 的 attention/FFN (全是 nn.Linear)
    - PatchMerging3D (nn.Linear)
    - dec_final_proj (nn.Linear)
    """
    replaced = 0

    # 1. Encoder conv blocks
    if hasattr(model, 'initial_encoder') and hasattr(model.initial_encoder, 'conv_block_list'):
        enc = model.initial_encoder
        for i in range(len(enc.conv_block_list)):
            enc.conv_block_list[i] = _replace_conv_block(enc.conv_block_list[i])
            replaced += 1

    # 2. Decoder final upsampling
    if hasattr(model, 'final_decoder') and hasattr(model.final_decoder, 'conv_block_list'):
        dec = model.final_decoder
        # conv blocks
        for i in range(len(dec.conv_block_list)):
            dec.conv_block_list[i] = _replace_conv_block(dec.conv_block_list[i])
            replaced += 1
        # upsample layers (each contains one Conv2d)
        for i in range(len(dec.upsample_list)):
            _replace_upsample_conv(dec.upsample_list[i])
            replaced += 1

    # 3. Decoder inter-block upsample layers
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'upsample_layers'):
        for i in range(len(model.decoder.upsample_layers)):
            _replace_upsample_conv(model.decoder.upsample_layers[i])
            replaced += 1

    print(f"[E2CNN] 已替换 {replaced} 个模块 (Conv2d → C4 等变 R2Conv)")
    return model


def count_e2cnn_layers(model):
    """统计模型中 e2cnn 等变层的数量"""
    n_equiv = sum(1 for m in model.modules() if isinstance(m, EquivariantConv2dWrapper))
    n_norm = sum(1 for m in model.modules() if isinstance(m, EquivariantGroupNormWrapper))
    n_conv2d = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    return {"R2Conv": n_equiv, "EquivNorm": n_norm, "remaining_Conv2d": n_conv2d}
