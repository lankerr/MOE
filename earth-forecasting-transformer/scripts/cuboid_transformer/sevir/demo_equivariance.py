"""
旋转等变性直觉演示
==================
用圆环、方块等几何图案，直观展示:
  - 普通 Conv2d: 旋转输入后输出变了 (不等变)
  - e2cnn R2Conv: 旋转输入后输出只是跟着旋转 (等变!)

核心验证:
  f(R·x) ≈ R·f(x)   对 e2cnn 成立
  f(R·x) ≠ R·f(x)   对普通 Conv2d 不成立

用法:
  conda activate rtx3050ti_cu128
  python -u demo_equivariance.py
"""

import torch
import torch.nn as nn
import numpy as np
import os

# ============================================================
#  1. 创建几何测试图案
# ============================================================

def make_ring(size=64, r_outer=20, r_inner=12, center=None):
    """圆环"""
    if center is None:
        center = (size // 2, size // 2)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    ring = ((dist <= r_outer) & (dist >= r_inner)).astype(np.float32)
    return ring

def make_ring_with_notch(size=64, r_outer=20, r_inner=12):
    """带缺口的圆环 (有明确的方向性)"""
    ring = make_ring(size, r_outer, r_inner)
    cx, cy = size // 2, size // 2
    # 右侧缺口
    ring[cy-3:cy+3, cx+r_inner:cx+r_outer+1] = 0
    return ring

def make_L_shape(size=64):
    """L形"""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    img[cy-15:cy+15, cx-15:cx-5] = 1.0  # 竖线
    img[cy+5:cy+15, cx-15:cx+15] = 1.0   # 横线
    return img

def make_arrow(size=64):
    """箭头 (指向右)"""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    img[cy-2:cy+2, cx-20:cx+10] = 1.0  # 箭杆
    for i in range(8):
        img[cy-i-2:cy-i, cx+10-i:cx+12-i] = 1.0  # 上翼
        img[cy+i:cy+i+2, cx+10-i:cx+12-i] = 1.0    # 下翼
    return img

def rotate_image_90(img, k=1):
    """旋转图像 k×90° (逆时针)"""
    return np.rot90(img, k=k).copy()

def images_to_tensor(img):
    """(H,W) -> (1,1,H,W) tensor"""
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

# ============================================================
#  2. 构建对比网络
# ============================================================

def build_standard_conv(seed=42):
    """标准 Conv2d"""
    torch.manual_seed(seed)
    conv = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1),
    )
    return conv

def build_e2cnn_conv(seed=42):
    """e2cnn C4 等变 Conv"""
    from e2cnn import gspaces, nn as enn
    torch.manual_seed(seed)
    gspace = gspaces.Rot2dOnR2(N=4)
    in_type = enn.FieldType(gspace, [gspace.trivial_repr])      # 1 ch
    mid_type = enn.FieldType(gspace, [gspace.regular_repr] * 2)  # 8 ch
    out_type = enn.FieldType(gspace, [gspace.trivial_repr])      # 1 ch

    model = enn.SequentialModule(
        enn.R2Conv(in_type, mid_type, 3, padding=1),
        enn.ReLU(mid_type, inplace=True),
        enn.R2Conv(mid_type, mid_type, 3, padding=1),
        enn.ReLU(mid_type, inplace=True),
        enn.R2Conv(mid_type, out_type, 3, padding=1),
    )
    return model, in_type

# ============================================================
#  3. 等变性测试核心
# ============================================================

def test_equivariance(name, img, std_conv, e2_conv, e2_in_type):
    """
    测试等变性:
      1. 原图 → f(x)
      2. 旋转90° → R·x → f(R·x)
      3. 对比 f(R·x) vs rot90(f(x))
    """
    from e2cnn import nn as enn

    x_orig = images_to_tensor(img)
    x_rot = images_to_tensor(rotate_image_90(img, k=1))

    with torch.no_grad():
        # 标准 Conv
        y_std_orig = std_conv(x_orig)
        y_std_rot = std_conv(x_rot)
        y_std_orig_then_rot = torch.from_numpy(
            rotate_image_90(y_std_orig.squeeze().numpy(), k=1)
        ).unsqueeze(0).unsqueeze(0)

        # e2cnn Conv
        geo_orig = enn.GeometricTensor(x_orig, e2_in_type)
        geo_rot = enn.GeometricTensor(x_rot, e2_in_type)
        y_e2_orig = e2_conv(geo_orig).tensor
        y_e2_rot = e2_conv(geo_rot).tensor
        y_e2_orig_then_rot = torch.from_numpy(
            rotate_image_90(y_e2_orig.squeeze().numpy(), k=1)
        ).unsqueeze(0).unsqueeze(0)

    # 计算误差: f(R·x) vs R·f(x)
    err_std = (y_std_rot - y_std_orig_then_rot).abs().mean().item()
    err_e2 = (y_e2_rot - y_e2_orig_then_rot).abs().mean().item()

    max_std = y_std_orig.abs().mean().item()
    max_e2 = y_e2_orig.abs().mean().item()

    rel_err_std = err_std / (max_std + 1e-8) * 100
    rel_err_e2 = err_e2 / (max_e2 + 1e-8) * 100

    return {
        'name': name,
        'err_std_abs': err_std,
        'err_e2_abs': err_e2,
        'rel_err_std': rel_err_std,
        'rel_err_e2': rel_err_e2,
        'y_std_orig': y_std_orig.squeeze().numpy(),
        'y_std_rot': y_std_rot.squeeze().numpy(),
        'y_std_orig_then_rot': y_std_orig_then_rot.squeeze().numpy(),
        'y_e2_orig': y_e2_orig.squeeze().numpy(),
        'y_e2_rot': y_e2_rot.squeeze().numpy(),
        'y_e2_orig_then_rot': y_e2_orig_then_rot.squeeze().numpy(),
    }


def test_all_rotations(name, img, std_conv, e2_conv, e2_in_type):
    """测试 0°/90°/180°/270° 四个角度"""
    from e2cnn import nn as enn

    results = []
    for k in range(4):
        angle = k * 90
        x_rot = images_to_tensor(rotate_image_90(img, k=k))
        x_orig = images_to_tensor(img)

        with torch.no_grad():
            # Standard
            y_std_orig = std_conv(x_orig)
            y_std_rot = std_conv(x_rot)
            y_std_orig_then_rot = torch.from_numpy(
                rotate_image_90(y_std_orig.squeeze().numpy(), k=k)
            ).unsqueeze(0).unsqueeze(0)

            # E2CNN
            geo_orig = enn.GeometricTensor(x_orig, e2_in_type)
            geo_rot = enn.GeometricTensor(x_rot, e2_in_type)
            y_e2_orig = e2_conv(geo_orig).tensor
            y_e2_rot = e2_conv(geo_rot).tensor
            y_e2_orig_then_rot = torch.from_numpy(
                rotate_image_90(y_e2_orig.squeeze().numpy(), k=k)
            ).unsqueeze(0).unsqueeze(0)

        err_std = (y_std_rot - y_std_orig_then_rot).abs().mean().item()
        err_e2 = (y_e2_rot - y_e2_orig_then_rot).abs().mean().item()
        results.append((angle, err_std, err_e2))
    return results


# ============================================================
#  4. 可视化
# ============================================================

def save_comparison_figure(results_list, images_dict, save_path="demo_equivariance.png"):
    """保存对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过图片保存")
        return

    n = len(results_list)
    fig, axes = plt.subplots(n, 7, figsize=(21, 3*n))
    if n == 1:
        axes = [axes]

    for row, r in enumerate(results_list):
        name = r['name']
        img = images_dict[name]

        # Col 0: 原图
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_title(f'{name}\n原图 x', fontsize=9)
        axes[row][0].axis('off')

        # Col 1: 旋转90°
        axes[row][1].imshow(rotate_image_90(img, 1), cmap='gray')
        axes[row][1].set_title('R(x)\n旋转90°', fontsize=9)
        axes[row][1].axis('off')

        # Col 2: Conv(x)
        axes[row][2].imshow(r['y_std_orig'], cmap='RdBu_r')
        axes[row][2].set_title('Conv(x)', fontsize=9)
        axes[row][2].axis('off')

        # Col 3: Conv(Rx) - 先旋转再卷积
        axes[row][3].imshow(r['y_std_rot'], cmap='RdBu_r')
        axes[row][3].set_title(f'Conv(Rx)\nerr={r["rel_err_std"]:.1f}%', fontsize=9, color='red')
        axes[row][3].axis('off')

        # Col 4: R(Conv(x)) - 先卷积再旋转
        axes[row][4].imshow(r['y_std_orig_then_rot'], cmap='RdBu_r')
        axes[row][4].set_title('R(Conv(x))\n理论应=Col3', fontsize=9)
        axes[row][4].axis('off')

        # Col 5: E2(Rx) - e2cnn 先旋转再卷积
        axes[row][5].imshow(r['y_e2_rot'], cmap='RdBu_r')
        axes[row][5].set_title(f'E2(Rx)\nerr={r["rel_err_e2"]:.2f}%', fontsize=9, color='green')
        axes[row][5].axis('off')

        # Col 6: R(E2(x)) - e2cnn 先卷积再旋转
        axes[row][6].imshow(r['y_e2_orig_then_rot'], cmap='RdBu_r')
        axes[row][6].set_title('R(E2(x))\n理论应=Col5', fontsize=9)
        axes[row][6].axis('off')

    fig.suptitle('旋转等变性验证: Conv(Rx) vs R(Conv(x))\n'
                 '红色=普通Conv(不等变)  绿色=E2CNN(等变)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图片保存到: {save_path}")
    plt.close()


# ============================================================
#  5. 主程序
# ============================================================

def main():
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 70)
    print("旋转等变性直觉演示")
    print("=" * 70)
    print()
    print("核心问题: 将输入旋转 90° 后再送入网络，")
    print("         输出是否等于 '先送入网络再旋转输出'？")
    print()
    print("  等变: f(R·x) = R·f(x)  → 误差 ≈ 0")
    print("  不等变: f(R·x) ≠ R·f(x)  → 误差很大")
    print("=" * 70)

    # 构建网络
    print("\n[1/4] 构建网络...")
    std_conv = build_standard_conv(seed=42)
    e2_conv, e2_in_type = build_e2cnn_conv(seed=42)

    std_params = sum(p.numel() for p in std_conv.parameters())
    e2_params = sum(p.numel() for p in e2_conv.parameters())
    print(f"  标准 Conv:  {std_params} params")
    print(f"  E2CNN C4:   {e2_params} params ({e2_params/std_params*100:.0f}% of Conv)")

    # 创建测试图案
    print("\n[2/4] 创建几何测试图案...")
    test_images = {
        "圆环":       make_ring(64),
        "带缺口圆环": make_ring_with_notch(64),
        "L形":        make_L_shape(64),
        "箭头":       make_arrow(64),
    }
    for name, img in test_images.items():
        print(f"  {name}: {img.shape}, 非零像素={img.sum():.0f}")

    # 90° 等变性测试
    print("\n[3/4] 90° 旋转等变性测试...")
    print()
    print(f"{'图案':<12s} | {'Conv误差':>10s} | {'E2CNN误差':>10s} | {'Conv相对误差':>12s} | {'E2CNN相对误差':>13s} | 结论")
    print("-" * 90)

    results_list = []
    for name, img in test_images.items():
        r = test_equivariance(name, img, std_conv, e2_conv, e2_in_type)
        results_list.append(r)

        verdict = "✅ E2CNN 完美等变" if r['rel_err_e2'] < 1.0 else "⚠ 需检查"
        conv_verdict = "❌ 不等变" if r['rel_err_std'] > 5.0 else "~近似"

        print(f"{name:<12s} | {r['err_std_abs']:10.6f} | {r['err_e2_abs']:10.6f} | "
              f"{r['rel_err_std']:10.2f}% {conv_verdict} | {r['rel_err_e2']:10.4f}% {verdict}")

    # 全角度测试
    print("\n[4/4] 全角度旋转测试 (0°/90°/180°/270°)...")
    print()
    print(f"{'图案':<12s} | {'角度':>4s} | {'Conv误差':>12s} | {'E2CNN误差':>12s}")
    print("-" * 60)

    for name, img in test_images.items():
        rot_results = test_all_rotations(name, img, std_conv, e2_conv, e2_in_type)
        for angle, err_std, err_e2 in rot_results:
            tag_std = "≈0" if err_std < 1e-6 else f"{err_std:.6f}"
            tag_e2 = "≈0" if err_e2 < 1e-6 else f"{err_e2:.6f}"
            if angle == 0:
                # 0° 旋转两者都应该是 0
                tag_std = "0 (trivial)"
                tag_e2 = "0 (trivial)"
            print(f"{name:<12s} | {angle:>3d}° | {tag_std:>12s} | {tag_e2:>12s}")
        print("-" * 60)

    # 数值总结
    print("\n" + "=" * 70)
    print("📊 数值总结")
    print("=" * 70)
    avg_std = np.mean([r['rel_err_std'] for r in results_list])
    avg_e2 = np.mean([r['rel_err_e2'] for r in results_list])
    print(f"  普通 Conv 平均相对误差:  {avg_std:.2f}%")
    print(f"  E2CNN C4  平均相对误差:  {avg_e2:.4f}%")
    print(f"  等变性提升:              {avg_std / (avg_e2 + 1e-8):.0f}x")
    print()
    print("  结论:")
    print("    普通 Conv2d: 旋转输入后输出完全不同 → 不等变")
    print("    E2CNN R2Conv: 旋转输入后输出完美跟随旋转 → 等变!")
    print("    误差 < 0.01% 仅来自浮点精度，理论上精确为 0")
    print()

    # 物理意义
    print("=" * 70)
    print("🌀 气象雷达中的意义")
    print("=" * 70)
    print("  风暴可能从任何方向移动:")
    print("    - 普通 Conv: 必须通过数据增强学习每个方向")
    print("    - E2CNN:     任何方向天然等价，一个方向学会 = 所有方向学会")
    print()
    print("  等价于参数共享: C4群有4个元素 → 参数量可以更少")
    print(f"  实际对比: Conv={std_params} params vs E2CNN={e2_params} params")
    print("=" * 70)

    # 保存图片
    save_comparison_figure(results_list, test_images)


if __name__ == '__main__':
    main()
