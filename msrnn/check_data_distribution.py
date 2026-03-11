import numpy as np
import matplotlib.pyplot as plt
import os

# 加载一个样本
data_path = "/data_8t/zhouying/chongqing_radar_npy"
sample_file = os.listdir(data_path)[0]
data = np.load(os.path.join(data_path, sample_file))

print("="*60)
print("重庆雷达数据分布检查")
print("="*60)
print(f"文件: {sample_file}")
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"\n原始数据统计:")
print(f"  最小值: {data.min():.2f}")
print(f"  最大值: {data.max():.2f}")
print(f"  均值: {data.mean():.2f}")
print(f"  标准差: {data.std():.2f}")
print(f"  中位数: {np.median(data):.2f}")

# 检查数据分布
print(f"\n数据分布:")
print(f"  =0的比例: {(data == 0).sum() / data.size * 100:.2f}%")
print(f"  >0的比例: {(data > 0).sum() / data.size * 100:.2f}%")
print(f"  >10的比例: {(data > 10).sum() / data.size * 100:.2f}%")
print(f"  >20的比例: {(data > 20).sum() / data.size * 100:.2f}%")
print(f"  >40的比例: {(data > 40).sum() / data.size * 100:.2f}%")

# 归一化后的统计
normalized = data / 80.0
print(f"\n归一化后 (除以80):")
print(f"  范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
print(f"  均值: {normalized.mean():.4f}")
print(f"  非零像素的均值: {normalized[normalized > 0].mean():.4f}")

# 绘制直方图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(data.flatten(), bins=50, edgecolor='black')
plt.title('原始数据分布')
plt.xlabel('dBZ值')
plt.ylabel('频数')

plt.subplot(1, 2, 2)
plt.hist(normalized.flatten(), bins=50, edgecolor='black')
plt.title('归一化后分布')
plt.xlabel('归一化值')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('chongqing_data_distribution.png', dpi=150)
print(f"\n✅ 分布图已保存: chongqing_data_distribution.png")
