"""
检查重庆雷达数据文件信息
========================

用于检查:
1. 数据文件数量
2. 实际图像尺寸
3. 数据范围和统计
4. 是否需要预处理

使用方法:
    python check_chongqing_data.py --data_dir /path/to/chongqing/data
"""

import os
import sys
import glob
import argparse
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))


def check_data_directory(data_dir: str):
    """检查数据目录"""
    print(f"\n{'='*60}")
    print(f"检查目录: {data_dir}")
    print(f"{'='*60}\n")

    # 查找数据文件
    pattern = os.path.join(data_dir, "day_simple_*.npy")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[错误] 未找到数据文件!")
        print(f"搜索路径: {pattern}")
        return

    print(f"[1] 文件数量: {len(files)}")

    # 检查第一个文件
    print(f"\n[2] 第一个文件: {os.path.basename(files[0])}")

    try:
        data = np.load(files[0], mmap_mode='r')
        print(f"    形状: {data.shape}")
        print(f"    类型: {data.dtype}")

        if data.ndim == 3:
            n_frames, height, width = data.shape
            print(f"    帧数: {n_frames}")
            print(f"    图像尺寸: {height} x {width}")

            # 计算时间覆盖
            interval_minutes = 6  # 重庆数据6分钟间隔
            total_hours = (n_frames - 1) * interval_minutes / 60
            print(f"    时间覆盖: 约 {total_hours:.1f} 小时")

            # 检查数据范围
            print(f"\n[3] 数据统计:")
            print(f"    最小值: {data.min():.4f}")
            print(f"    最大值: {data.max():.4f}")
            print(f"    平均值: {data.mean():.4f}")
            print(f"    标准差: {data.std():.4f}")

            # 检查有效帧比例
            valid_frames = np.sum(data > 0.01, axis=(1, 2))
            valid_ratio = np.mean(valid_frames > (height * width * 0.1))
            print(f"    有效帧比例: {valid_ratio:.1%}")

        else:
            print(f"    [警告] 数据维度不匹配: {data.shape}")

    except Exception as e:
        print(f"    [错误] 无法读取文件: {e}")
        return

    # 检查日期范围
    print(f"\n[4] 日期范围:")
    dates = []
    for f in files[:10]:  # 只检查前10个
        try:
            basename = os.path.basename(f)
            date_str = basename.replace('day_simple_', '').replace('.npy', '')
            dates.append(date_str)
        except:
            pass

    if dates:
        print(f"    最早: {dates[-1]}")
        print(f"    最晚: {dates[0]}")

    # 推荐配置
    print(f"\n[5] 推荐配置:")
    if data.ndim == 3:
        h, w = data.shape[1], data.shape[2]

        # 找最近的常用尺寸
        common_sizes = [256, 384, 512]
        nearest_h = min(common_sizes, key=lambda x: abs(x - h))
        nearest_w = min(common_sizes, key=lambda x: abs(x - w))

        if h == w:
            print(f"    图像尺寸: {h}x{w} (已经是正方形)")
        else:
            print(f"    图像尺寸: {h}x{w} (非正方形)")
            print(f"    建议: resize 到 {nearest_h}x{nearest_w}")

        # 建议帧配置
        max_seq_len = min(48, data.shape[0] // 2)
        print(f"    最大序列长度: {max_seq_len} 帧")
        print(f"    推荐 2h→2h: in_len=20, out_len=20 (6分钟/帧)")
        print(f"    或 使用24帧保持与SEVIR一致")

    print(f"\n{'='*60}\n")


def generate_config(height: int, width: int, data_dir: str):
    """生成配置"""
    print(f"\n[生成配置]")

    # 确定目标尺寸
    common_sizes = [256, 384, 512]
    nearest_h = min(common_sizes, key=lambda x: abs(x - height))

    print(f"原始尺寸: {height} x {width}")
    print(f"建议配置:")
    print(f"  img_height: {nearest_h}")
    print(f"  img_width: {nearest_h}")
    print(f"  data_dir: {data_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default='/data_8t/dengjiaxuan/vil_gpu_daily_240_simple',
                       help='重庆数据目录')
    parser.add_argument('--generate_config', action='store_true',
                       help='生成配置文件')
    args = parser.parse_args()

    check_data_directory(args.data_dir)

    if args.generate_config:
        # 读取第一个文件获取尺寸
        files = glob.glob(os.path.join(args.data_dir, "day_simple_*.npy"))
        if files:
            data = np.load(files[0], mmap_mode='r')
            if data.ndim == 3:
                generate_config(data.shape[1], data.shape[2], args.data_dir)
