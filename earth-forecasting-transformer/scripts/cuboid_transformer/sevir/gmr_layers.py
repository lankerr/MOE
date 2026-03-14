"""
GMR-Conv 等变卷积替换模块 — Earthformer 49F 用
==============================================
将 CuboidTransformerModel 的 InitialStackPatchMergingEncoder 和
FinalStackUpsamplingDecoder 中的 nn.Conv2d 替换为 GMR_Conv2d
(Gaussian Mixture Ring Convolution, arXiv:2504.02819)。

GMR-Conv 优势:
  - 连续旋转等变 (任意角度，不限 90° 整数倍)
  - drop-in 替换 nn.Conv2d (无需 FieldType / GeometricTensor)
  - 参数更少: 高斯环分解 → O(C·n) 参数

Transformer (attention/FFN) 部分完全不改 — 只改 Conv 前/后端。

使用方法:
    from gmr_layers import patch_model_with_gmr, count_gmr_layers
    model = CuboidTransformerModel(...)
    patch_model_with_gmr(model)  # 原地替换 Conv2d → GMR_Conv2d
"""

import torch
import torch.nn as nn
from GMR_Conv import GMR_Conv2d


def _replace_conv2d_with_gmr(old_conv):
    """将单个 nn.Conv2d 替换为 GMR_Conv2d，保留所有配置。"""
    new_conv = GMR_Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size[0] if isinstance(old_conv.kernel_size, tuple) else old_conv.kernel_size,
        stride=old_conv.stride[0] if isinstance(old_conv.stride, tuple) else old_conv.stride,
        padding=old_conv.padding[0] if isinstance(old_conv.padding, tuple) else old_conv.padding,
        dilation=old_conv.dilation[0] if isinstance(old_conv.dilation, tuple) else old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    )
    return new_conv


def _replace_conv_block(conv_block_seq):
    """替换 nn.Sequential 中的所有 nn.Conv2d → GMR_Conv2d。
    GroupNorm、激活函数等保持不变。
    """
    new_layers = []
    for layer in conv_block_seq:
        if isinstance(layer, nn.Conv2d):
            new_layers.append(_replace_conv2d_with_gmr(layer))
        else:
            # GroupNorm, LeakyReLU 等保持原样
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def _replace_upsample_conv(upsample_layer):
    """替换 Upsample3DLayer 内部的 self.conv (nn.Conv2d) → GMR_Conv2d。"""
    old_conv = upsample_layer.conv
    upsample_layer.conv = _replace_conv2d_with_gmr(old_conv)


def patch_model_with_gmr(model):
    """原地将 CuboidTransformerModel 的 Conv2d 替换为 GMR_Conv2d。

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

    print(f"[GMR-Conv] 已替换 {replaced} 个模块 (Conv2d → GMR_Conv2d)")
    return model


def count_gmr_layers(model):
    """统计模型中 GMR-Conv 和剩余 Conv2d 的数量"""
    n_gmr = sum(1 for m in model.modules() if isinstance(m, GMR_Conv2d))
    n_conv2d = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and not isinstance(m, GMR_Conv2d))
    n_total_params = sum(p.numel() for p in model.parameters())
    return {"GMR_Conv2d": n_gmr, "remaining_Conv2d": n_conv2d, "total_params": n_total_params}
