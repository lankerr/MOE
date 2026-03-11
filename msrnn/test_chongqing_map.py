import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from config import cfg
from model_new import Model
from net_params import nets
from load_chongqing import load_chongqing

# 标准雷达反射率色阶和颜色映射
RADAR_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
RADAR_COLORS = [
    '#0000FF', '#00C8FF', '#00FF00', '#32CD32', '#008000',
    '#FFFF00', '#FFD700', '#FFA500', '#FF6347', '#FF0000',
    '#C80000', '#FF69B4', '#800080', '#DDA0DD'
]

def get_radar_cmap():
    """创建雷达反射率颜色映射"""
    cols = RADAR_COLORS
    lev = RADAR_LEVELS
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(lev, len(cols))
    return cmap, norm

def remove_prefix(state_dict):
    """移除多GPU训练时的'module.'前缀"""
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    return new_dict

def reshape_patch_TBCHW(x, patch_size):
    """
    与Sevir测试代码一致的patch处理函数
    输入 x: [T, B, C, H, W]
    输出:  [T, B, patch_size²*C, H//patch_size, W//patch_size]
    """
    T, B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"尺寸{H}x{W}不能被{patch_size}整除"
    x = x.reshape(T, B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 4, 6, 2)  # T, B, h, w, p, p, C
    x = x.reshape(T, B, H // patch_size, W // patch_size, patch_size * patch_size * C)
    x = x.permute(0, 1, 4, 2, 3)  # T, B, C', H', W'
    return x

def reshape_patch_back(x, patch_size):
    """
    与Sevir测试代码一致的patch还原函数
    输入 x: [T, B, patch_size²*C, H', W']
    输出:  [T, B, C, H, W]
    """
    T, B, C_p, H_p, W_p = x.shape
    C = C_p // (patch_size * patch_size)
    x = x.permute(0, 1, 3, 4, 2)  # T, B, H', W', C'
    x = x.reshape(T, B, H_p, W_p, patch_size, patch_size, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)  # T, B, H', p, W', p, C
    x = x.reshape(T, B, H_p * patch_size, W_p * patch_size, C)
    x = x.permute(0, 1, 4, 2, 3)  # T, B, C, H, W
    return x

def dbz_to_sevir_scale(data):
    """
    将重庆的dBZ数据(0-80)转换为Sevir的0-255范围
    """
    # 简单线性映射
    data_sevir = data * (255.0 / 80.0)
    data_sevir = np.clip(data_sevir, 0, 255)
    return data_sevir

def sevir_to_dbz(data):
    """
    Sevir的0-255范围转换回dBZ
    """
    data_dbz = data * (80.0 / 255.0)
    return data_dbz

def compute_metrics_sevir(pred, target, thresholds):
    """
    使用Sevir测试代码相同的指标计算方法
    阈值: [16, 74, 132, 160, 181] (0-255范围)
    """
    # 确保在0-255范围
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    eps = 1e-6

    results = {}
    for th in thresholds:
        th = int(th)
        pred_bin = (pred >= th).astype(np.uint8)
        target_bin = (target >= th).astype(np.uint8)

        TP = np.logical_and(pred_bin == 1, target_bin == 1).sum()
        FP = np.logical_and(pred_bin == 1, target_bin == 0).sum()
        FN = np.logical_and(pred_bin == 0, target_bin == 1).sum()
        TN = np.logical_and(pred_bin == 0, target_bin == 0).sum()

        POD = TP / (TP + FN + eps)
        FAR = FP / (TP + FP + eps)
        CSI = TP / (TP + FP + FN + eps)
        HSS = 2 * (TP * TN - FN * FP) / ((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN) + eps)

        results[f'CSI_{th}'] = CSI
        results[f'POD_{th}'] = POD
        results[f'FAR_{th}'] = FAR
        results[f'HSS_{th}'] = HSS
    
    # 计算平均值
    for metric_type in ['CSI', 'POD', 'FAR', 'HSS']:
        avg = np.mean([results[f'{metric_type}_{th}'] for th in thresholds])
        results[f'avg_{metric_type}'] = avg
    
    return results

def normalize_to_dbz(data, inverse=False):
    """将0-1归一化数据转换为dBZ或反向转换"""
    if not inverse:
        # 0-1 -> 0-80 dBZ
        return data * 80.0
    else:
        # dBZ -> 0-1
        return data / 80.0

def visualize_comparison(input_seq, output_seq, target_seq, save_path, batch_idx, sample_idx):
    """使用雷达标准色阶可视化比较"""
    radar_cmap, radar_norm = get_radar_cmap()
    
    # 将数据转换为dBZ
    input_dbz = normalize_to_dbz(input_seq)
    output_dbz = normalize_to_dbz(output_seq)
    target_dbz = normalize_to_dbz(target_seq)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # 显示最后几帧输入
    for i in range(5):
        if i < input_dbz.shape[0]:
            ax = axes[0, i]
            # 移除 vmin 和 vmax 参数
            im = ax.imshow(input_dbz[-5+i], cmap=radar_cmap, norm=radar_norm)  # ✅ 修正
            ax.set_title(f'输入 t={-5+i}', fontsize=10)
            ax.set_facecolor('white')
            ax.axis('off')
    
    # 显示输出帧
    for i in range(5):
        if i < output_dbz.shape[0]:
            ax = axes[1, i]
            im = ax.imshow(output_dbz[i], cmap=radar_cmap, norm=radar_norm)  # ✅ 修正
            ax.set_title(f'预测 t={i}', fontsize=10)
            ax.set_facecolor('white')
            ax.axis('off')
    
    # 显示目标帧
    for i in range(5):
        if i < target_dbz.shape[0]:
            ax = axes[2, i]
            im = ax.imshow(target_dbz[i], cmap=radar_cmap, norm=radar_norm)  # ✅ 修正
            ax.set_title(f'目标 t={i}', fontsize=10)
            ax.set_facecolor('white')
            ax.axis('off')
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('反射率 (dBZ)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 设置颜色条刻度
    cbar.set_ticks(RADAR_LEVELS)
    cbar.set_ticklabels([str(l) for l in RADAR_LEVELS])
    
    plt.suptitle(f'批次{batch_idx}-样本{sample_idx} - 雷达反射率对比', 
                fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def visualize_single_frame(data, title, save_path):
    """使用雷达色阶可视化单个帧"""
    radar_cmap, radar_norm = get_radar_cmap()
    
    # 转换为dBZ
    data_dbz = normalize_to_dbz(data)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 移除 vmin 和 vmax 参数
    im = ax.imshow(data_dbz, cmap=radar_cmap, norm=radar_norm)  # ✅ 修正
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('反射率 (dBZ)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(RADAR_LEVELS)
    cbar.set_ticklabels([str(l) for l in RADAR_LEVELS])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False,
                       default="/data_8t/zhouying/msrnn/modelBak/01ConvLSTM_ps4_hs64_l6_test0608.pth",
                       help="预训练模型路径")
    parser.add_argument("--output", type=str, default="results/chongqing_radar_results.xlsx",
                       help="结果保存路径")
    parser.add_argument("--num", type=int, default=10,
                       help="测试序列数量")
    parser.add_argument("--batch", type=int, default=1,
                       help="batch大小")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="是否可视化结果")
    parser.add_argument("--visualize_dir", type=str, default="visualizations_radar",
                       help="可视化结果保存目录")
    args = parser.parse_args()
    
    print("=" * 60)
    print("重庆雷达数据测试 - 雷达标准色阶版")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"雷达色阶: {RADAR_LEVELS}")
    
    # 创建目录
    if args.visualize:
        os.makedirs(args.visualize_dir, exist_ok=True)
        debug_dir = os.path.join(args.visualize_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"可视化目录: {args.visualize_dir}")
    
    # 打印配置
    print(f"\n模型: {cfg.model_name}")
    print(f"输入长度: {cfg.in_len}, 输出长度: {cfg.out_len}")
    print(f"使用patch: {cfg.reshape_patch}, patch大小: {cfg.patch_size}")
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = Model(nets[0], nets[1], nets[2]).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = remove_prefix(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        state_dict = remove_prefix(checkpoint['state_dict'])
    else:
        state_dict = remove_prefix(checkpoint)
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("模型加载成功")
    
    # 加载数据 - 跳过数据分布分析
    print(f"\n加载重庆数据集...")
    
    # 直接创建数据加载器，跳过有问题的load_chongqing函数
    # 或者修复load_chongqing.py文件中的错误
    try:
        test_loader = load_chongqing(batch_size=args.batch, split='test', test_num=args.num)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("尝试跳过数据分布分析...")
        
        # 直接修改load_chongqing.py文件
        # 打开load_chongqing.py，注释掉第56行的数据分布分析
        # 或者在运行时跳过
        import importlib
        import load_chongqing
        
        # 重新导入模块
        importlib.reload(load_chongqing)
        test_loader = load_chongqing.load_chongqing(batch_size=args.batch, split='test', test_num=args.num)
    
    if len(test_loader) == 0:
        print("错误: 没有加载到数据")
        return
    
    print(f"测试批次数量: {len(test_loader)}")
    
    # 使用Sevir的阈值（转换为dBZ阈值）
    sevir_thresholds = [16, 74, 132, 160, 181]
    # 转换为dBZ阈值：sevir值 * (80/255)
    dbz_thresholds = [th * (80/255) for th in sevir_thresholds]
    print(f"Sevir阈值: {sevir_thresholds}")
    print(f"对应dBZ阈值: {[f'{t:.1f}' for t in dbz_thresholds]}")
    
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试进度")):
            inputs = batch[:, :cfg.in_len]    # [B, 13, 1, H, W]
            targets = batch[:, cfg.in_len:cfg.in_len+cfg.out_len]  # [B, 12, 1, H, W]
            
            print(f"\n批次 {batch_idx}:")
            print(f"  输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  目标范围: [{targets.min():.3f}, {targets.max():.3f}]")
            
            # 关键步骤：将重庆数据转换为Sevir数据范围
            # 重庆数据是0-80 dBZ归一化到0-1
            # 需要转换到0-255范围（Sevir数据范围）
            inputs_dbz = inputs.numpy() * 80.0  # 转换为dBZ
            inputs_sevir = dbz_to_sevir_scale(inputs_dbz)  # 转换为Sevir范围
            inputs_sevir = inputs_sevir / 255.0  # 归一化到0-1
            
            targets_dbz = targets.numpy() * 80.0
            targets_sevir = dbz_to_sevir_scale(targets_dbz)
            targets_sevir = targets_sevir / 255.0
            
            print(f"  Sevir范围输入: [{inputs_sevir.min():.3f}, {inputs_sevir.max():.3f}]")
            print(f"  Sevir范围目标: [{targets_sevir.min():.3f}, {targets_sevir.max():.3f}]")
            
            # 转换为tensor
            inputs_tensor = torch.FloatTensor(inputs_sevir).to(device)
            targets_tensor = torch.FloatTensor(targets_sevir).to(device)
            
            # 准备输入格式: [B, T, C, H, W] -> [T, B, C, H, W]
            inputs_t = inputs_tensor.permute(1, 0, 2, 3, 4)  # [13, B, 1, H, W]
            
            # 模型推理
            if cfg.reshape_patch:
                x_patch = reshape_patch_TBCHW(inputs_t, cfg.patch_size)
                print(f"  Patch输入形状: {x_patch.shape}")
                
                eta = 0.0
                epoch = 0
                y_patch, _ = model([x_patch, eta, epoch], mode='test')
                
                outputs = reshape_patch_back(y_patch, cfg.patch_size)
            else:
                eta = 0.0
                epoch = 0
                outputs, _ = model([inputs_t, eta, epoch], mode='test')
            
            print(f"  模型输出形状: {outputs.shape}")
            print(f"  模型输出范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
            
            # 确保输出是12帧
            if outputs.shape[0] > cfg.out_len:
                outputs = outputs[:cfg.out_len]
            
            # 准备目标数据
            targets_t = targets_tensor.permute(1, 0, 2, 3, 4)
            
            # 转换为numpy
            outputs_np = outputs.cpu().numpy()    # [12, B, 1, H, W]
            targets_np = targets_t.cpu().numpy()  # [12, B, 1, H, W]
            
            # 调整维度: [T, B, C, H, W] -> [B, T, H, W]
            outputs_np = np.transpose(outputs_np, (1, 0, 2, 3, 4))
            targets_np = np.transpose(targets_np, (1, 0, 2, 3, 4))
            
            outputs_np = outputs_np.squeeze(2)  # [B, 12, H, W]
            targets_np = targets_np.squeeze(2)  # [B, 12, H, W]
            
            # 计算每个样本的指标
            for b in range(outputs_np.shape[0]):
                pred = outputs_np[b]
                target = targets_np[b]
                
                # 转换为dBZ
                pred_dbz = normalize_to_dbz(pred)
                target_dbz = normalize_to_dbz(target)
                
                print(f"  样本{b}:")
                print(f"    预测dBZ范围: [{pred_dbz.min():.1f}, {pred_dbz.max():.1f}]")
                print(f"    目标dBZ范围: [{target_dbz.min():.1f}, {target_dbz.max():.1f}]")
                
                # 计算Sevir风格的指标（使用原始0-1数据）
                metrics = compute_metrics_sevir(pred, target, sevir_thresholds)
                metrics['batch'] = batch_idx
                metrics['sample'] = b
                metrics['pred_dbz_min'] = pred_dbz.min()
                metrics['pred_dbz_max'] = pred_dbz.max()
                metrics['pred_dbz_mean'] = pred_dbz.mean()
                metrics['target_dbz_min'] = target_dbz.min()
                metrics['target_dbz_max'] = target_dbz.max()
                metrics['target_dbz_mean'] = target_dbz.mean()
                
                all_metrics.append(metrics)
                
                # 可视化
                if args.visualize and batch_idx < 5 and b == 0:
                    input_vis = inputs[b].numpy().squeeze(1)
                    output_vis = outputs_np[b]
                    target_vis = targets_np[b]
                    
                    # 保存比较图
                    compare_path = os.path.join(args.visualize_dir, f"batch{batch_idx}_sample{b}_comparison.png")
                    visualize_comparison(input_vis, output_vis, target_vis, compare_path, batch_idx, b)
                    
                    # 保存单独的帧
                    for t in range(5):
                        if t < input_vis.shape[0]:
                            input_single_path = os.path.join(debug_dir, f"batch{batch_idx}_input_t{-5+t}.png")
                            visualize_single_frame(input_vis[-5+t], f'输入 t={-5+t}', input_single_path)
                        
                        if t < output_vis.shape[0]:
                            output_single_path = os.path.join(debug_dir, f"batch{batch_idx}_output_t{t}.png")
                            visualize_single_frame(output_vis[t], f'预测 t={t}', output_single_path)
                        
                        if t < target_vis.shape[0]:
                            target_single_path = os.path.join(debug_dir, f"batch{batch_idx}_target_t{t}.png")
                            visualize_single_frame(target_vis[t], f'目标 t={t}', target_single_path)
    
    # 保存结果
    if all_metrics:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        df = pd.DataFrame(all_metrics)
        df.to_excel(args.output, index=False)
        
        # 保存为CSV以便查看
        csv_path = args.output.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        
        # 计算平均指标
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"总测试样本数: {len(all_metrics)}")
        
        print("\n预测dBZ统计:")
        print(f"  平均范围: [{df['pred_dbz_min'].mean():.1f}, {df['pred_dbz_max'].mean():.1f}] dBZ")
        print(f"  平均均值: {df['pred_dbz_mean'].mean():.1f} dBZ")
        
        print("\n目标dBZ统计:")
        print(f"  平均范围: [{df['target_dbz_min'].mean():.1f}, {df['target_dbz_max'].mean():.1f}] dBZ")
        print(f"  平均均值: {df['target_dbz_mean'].mean():.1f} dBZ")
        
        print("\nSevir阈值指标:")
        for th, dbz_th in zip(sevir_thresholds, dbz_thresholds):
            csi_avg = df[f'CSI_{th}'].mean()
            pod_avg = df[f'POD_{th}'].mean()
            far_avg = df[f'FAR_{th}'].mean()
            print(f"阈值 {th:3d} (对应{dbz_th:.1f}dBZ): CSI={csi_avg:.3f}, POD={pod_avg:.3f}, FAR={far_avg:.3f}")
        
        print("\n平均指标:")
        print(f"平均CSI: {df['avg_CSI'].mean():.3f}")
        print(f"平均POD: {df['avg_POD'].mean():.3f}")
        print(f"平均FAR: {df['avg_FAR'].mean():.3f}")
        print(f"平均HSS: {df['avg_HSS'].mean():.3f}")
        
        print(f"\n✅ 详细结果已保存到:")
        print(f"  Excel文件: {args.output}")
        print(f"  CSV文件: {csv_path}")
        
        if args.visualize:
            print(f"  可视化文件: {args.visualize_dir}/")
    else:
        print("❌ 没有生成任何结果")

if __name__ == "__main__":
    main()