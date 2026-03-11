import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import sys
from datetime import datetime

# 添加路径
sys.path.append('/data_8t/zhouying/msrnn')

# 导入配置文件
from config import cfg

# ==================== 雷达色阶配置 ====================
RADAR_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
RADAR_COLORS = [
    '#0000FF', '#00C8FF', '#00FF00', '#32CD32', '#008000',
    '#FFFF00', '#FFD700', '#FFA500', '#FF6347', '#FF0000',
    '#C80000', '#FF69B4', '#800080', '#DDA0DD'
]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

print("=" * 60)
print("重庆雷达数据ConvLSTM外推测试")
print("=" * 60)
print(f"模型: ConvLSTM")
print(f"输入帧: {cfg.in_len}, 输出帧: {cfg.out_len}")
print(f"图像: {cfg.width}x{cfg.height}")
print("=" * 60)

# 配置
cfg.dataset = 'Chongqing'
cfg.batch = 1
cfg.reshape_patch = True

# 导入模型
from model_new import Model
from net_params import nets
from load_chongqing import load_chongqing_test


def remove_module_prefix(state_dict):
    """去除多GPU训练时产生的'module.'前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    return new_state_dict


def reshape_patch_TBCHW(x, patch_size):
    """将大图分割为patch"""
    T, B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    
    x = x.reshape(T, B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 4, 6, 2)
    x = x.reshape(T, B, H // patch_size, W // patch_size, patch_size * patch_size * C)
    x = x.permute(0, 1, 4, 2, 3)
    
    return x


def reshape_patch_back(x, patch_size):
    """将patch还原为原图"""
    T, B, C_p, H_p, W_p = x.shape
    C = C_p // (patch_size * patch_size)
    
    x = x.permute(0, 1, 3, 4, 2)
    x = x.reshape(T, B, H_p, W_p, patch_size, patch_size, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(T, B, H_p * patch_size, W_p * patch_size, C)
    x = x.permute(0, 1, 4, 2, 3)
    
    return x


def postprocess_prediction(pred, method='clip'):
    """
    后处理预测结果 - 修复版：确保在[0,1]范围内
    问题3修复：增加clip截断，确保没有负值
    """
    if method == 'clip':
        # 关键修复：确保完全在[0,1]范围内
        pred = np.clip(pred, 0.0, 1.0)
    elif method == 'shift':
        if pred.min() < 0:
            pred = pred - pred.min()
            # 重新归一化到[0,1]
            if pred.max() > 0:
                pred = pred / pred.max()
    elif method == 'none':
        pass
    else:
        raise ValueError(f"未知的后处理方法: {method}")
    
    return pred


def compute_metrics(pred, target, thresholds=[10, 15, 20, 25, 30, 35, 40]):
    """
    计算CSI, POD, FAR指标
    """
    # 转换为dBZ
    pred_dbz = pred * 80.0
    target_dbz = target * 80.0
    
    # 数据清洗
    pred_dbz = np.nan_to_num(pred_dbz, nan=0.0)
    target_dbz = np.nan_to_num(target_dbz, nan=0.0)
    pred_dbz = np.clip(pred_dbz, 0, 80)
    target_dbz = np.clip(target_dbz, 0, 80)
    
    eps = 1e-10
    results = {}
    
    for th in thresholds:
        # 二值化
        pred_bin = (pred_dbz >= th).astype(np.float32)
        target_bin = (target_dbz >= th).astype(np.float32)
        
        # 统计
        TP = np.logical_and(pred_bin == 1, target_bin == 1).sum()
        FP = np.logical_and(pred_bin == 1, target_bin == 0).sum()
        FN = np.logical_and(pred_bin == 0, target_bin == 1).sum()
        
        # 计算指标
        if TP + FN > 0:
            POD = TP / (TP + FN + eps)
        else:
            POD = 0.0
            
        if TP + FP > 0:
            FAR = FP / (TP + FP + eps)
        else:
            FAR = 0.0
            
        if TP + FP + FN > 0:
            CSI = TP / (TP + FP + FN + eps)
        else:
            CSI = 0.0
        
        # BIAS
        if TP + FN > 0:
            BIAS = (TP + FP) / (TP + FN + eps)
        else:
            BIAS = 0.0
        
        results[f'CSI_{th}'] = float(CSI)
        results[f'POD_{th}'] = float(POD)
        results[f'FAR_{th}'] = float(FAR)
        results[f'BIAS_{th}'] = float(BIAS)
    
    return results


def plot_sequence_comparison(input_seq, target_seq, pred_seq, sample_idx, output_dir):
    """
    绘制序列对比图 - 修复版
    问题1修复：显示所有帧数的图
    第一行：13帧历史输入
    第二行：12帧目标图  
    第三行：12帧预测图
    """
    try:
        # 转换为dBZ
        input_dbz = input_seq * 80.0
        target_dbz = target_seq * 80.0
        pred_dbz = pred_seq * 80.0
        
        # 获取所有帧数
        input_frames = input_seq.shape[0]  # 13
        target_frames = target_seq.shape[0]  # 12
        pred_frames = pred_seq.shape[0]  # 12
        
        # 最大列数（13列，因为输入有13帧）
        max_cols = input_frames  # 13
        
        # 创建子图 - 3行，13列
        fig, axes = plt.subplots(3, max_cols, figsize=(max_cols * 1.5, 6))
        
        # 处理单列情况
        if max_cols == 1:
            axes = axes.reshape(3, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(3, -1)
        
        fig.patch.set_facecolor('white')
        
        # 绘制雷达图的函数 - 恢复原来的contourf方式
        def plot_radar(data, ax, title):
            try:
                # 使用contourf配合RADAR_LEVELS来定义颜色区间
                # 确保数据在合适的范围内
                data_clipped = np.clip(data, 0, 80)
                
                # 创建掩码，将0-5dBZ的区域设为白色
                mask_0_5 = (data_clipped >= 0) & (data_clipped < 5)
                
                # 使用contourf绘制
                contour = ax.contourf(data_clipped, levels=RADAR_LEVELS, cmap=RADAR_CMAP, extend='max')
                
                # 将0-5dBZ的区域设为白色
                if mask_0_5.any():
                    # 创建一个白色掩码
                    white_mask = np.zeros_like(data_clipped, dtype=bool)
                    white_mask[mask_0_5] = True
                    # 应用白色掩码
                    ax.contourf(white_mask, levels=[0.5, 1], colors=['white'], alpha=1.0)
                
                ax.set_title(title, fontsize=8, pad=2)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_facecolor('white')
                return contour
            except Exception as e:
                print(f"绘图错误: {e}")
                # 备用方案：使用imshow
                try:
                    data_clipped = np.clip(data, 0, 80)
                    im = ax.imshow(data_clipped, vmin=0, vmax=80, cmap=RADAR_CMAP, origin='lower')
                    ax.set_title(title, fontsize=8, pad=2)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_facecolor('white')
                    return im
                except:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)
                    ax.axis('off')
                    return None
        
        # 第一行：输入帧（13帧）
        for i in range(input_frames):
            plot_radar(input_dbz[i], axes[0, i], f'In {i}')
        
        # 第二行：真实目标（12帧），从第1列开始
        for i in range(target_frames):
            plot_radar(target_dbz[i], axes[1, i+1], f'Tar {i}')
        
        # 第三行：预测结果（12帧），从第1列开始
        for i in range(pred_frames):
            plot_radar(pred_dbz[i], axes[2, i+1], f'Pred {i}')
        
        # 隐藏多余的列
        for i in range(max_cols):
            # 第1行的所有列都有效（13个输入）
            # 第2行和第3行的第0列空着
            if i == 0:
                axes[1, i].axis('off')
                axes[2, i].axis('off')
            # 第2行和第3行，从第13列开始空着
            elif i > target_frames:
                axes[1, i].axis('off')
                axes[2, i].axis('off')
        
        # 添加标题
        plt.suptitle(f'ConvLSTM Prediction - Sample {sample_idx+1}', fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图像
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, f'comparison_{sample_idx+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"绘图错误: {e}")
        return None


def save_results(all_metrics, output_dir, thresholds=[10, 15, 20, 25, 30, 35, 40]):
    """保存结果"""
    if not all_metrics:
        return
    
    # 保存CSV
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, 'metrics_results.csv')
    df.to_csv(csv_path, index=False)
    
    # 计算平均指标
    avg_metrics = {}
    for th in thresholds:
        csi_key = f'CSI_{th}'
        pod_key = f'POD_{th}'
        far_key = f'FAR_{th}'
        bias_key = f'BIAS_{th}'
        
        if csi_key in df.columns:
            avg_metrics[csi_key] = df[csi_key].mean()
            avg_metrics[pod_key] = df[pod_key].mean()
            avg_metrics[far_key] = df[far_key].mean()
            avg_metrics[bias_key] = df[bias_key].mean()
    
    # 保存平均指标
    avg_df = pd.DataFrame([avg_metrics])
    avg_csv_path = os.path.join(output_dir, 'average_metrics.csv')
    avg_df.to_csv(avg_csv_path, index=False)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    print(f"测试样本数: {len(df)}")
    print(f"\n平均指标:")
    print(f"{'阈值(dBZ)':<10} {'CSI':<10} {'POD':<10} {'FAR':<10} {'BIAS':<10}")
    print(f"{'-'*50}")
    
    for th in thresholds:
        csi = avg_metrics.get(f'CSI_{th}', 0)
        pod = avg_metrics.get(f'POD_{th}', 0)
        far = avg_metrics.get(f'FAR_{th}', 0)
        bias = avg_metrics.get(f'BIAS_{th}', 0)
        
        print(f"{th:<10} {csi:<10.3f} {pod:<10.3f} {far:<10.3f} {bias:<10.3f}")
    
    print(f"\n结果保存到:")
    print(f"  详细指标: {csv_path}")
    print(f"  平均指标: {avg_csv_path}")
    
    return df, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='重庆雷达数据ConvLSTM测试')
    parser.add_argument('--resume', type=str, 
                       default='/data_8t/zhouying/msrnn/modelBak/01ConvLSTM_ps4_hs64_l6_test0608.pth',
                       help='预训练模型路径')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='测试样本数量')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图像')
    parser.add_argument('--postprocess', type=str, default='clip',
                       choices=['clip', 'shift', 'none'],
                       help='后处理方法')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.save_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = Model(nets[0], nets[1], nets[2]).to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    else:
        state_dict = remove_module_prefix(checkpoint)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"模型加载成功: {args.resume}")
    
    # 加载测试数据
    test_loader = load_chongqing_test(batch_size=1)
    if test_loader is None:
        print("数据加载失败")
        return
    
    # 开始测试
    all_metrics = []
    sample_count = 0
    
    print(f"\n开始测试 {args.num_samples} 个样本...")
    # 在 test_chongqing.py 的 main() 函数中，找到推理部分：

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试进度")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if sample_count >= args.num_samples:
                break
            try:
                # ========== 添加详细调试信息的开始 ==========
                print(f"\n{'='*60}")
                print(f"处理批次 {batch_idx+1}, 样本 {sample_count+1}")
                print(f"{'='*60}")
                
                # 打印原始数据形状
                print(f"原始数据形状:")
                print(f"  inputs: {inputs.shape} (应该是 [1, 13, 1, H, W])")
                print(f"  targets: {targets.shape} (应该是 [1, 12, 1, H, W])")
                
                # 调整维度 - inputs只需要13帧
                inputs = inputs.permute(1, 0, 2, 3, 4).to(device)  # [13, 1, 1, H, W]
                targets = targets.permute(1, 0, 2, 3, 4).to(device)  # [12, 1, 1, H, W]
                
                print(f"\n调整维度后:")
                print(f"  inputs: {inputs.shape} (应该是 [13, 1, 1, H, W])")
                print(f"  targets: {targets.shape} (应该是 [12, 1, 1, H, W])")
                
                # ========== 关键修复：构建正确的输入序列 ==========
                # 模型现在只需要13帧输入
                # 如果模型需要25帧，我们需要创建dummy帧
                
                # 方案A：如果修改后的模型只需要13帧输入
                if cfg.reshape_patch:
                    inputs_patch = reshape_patch_TBCHW(inputs, cfg.patch_size)
                else:
                    inputs_patch = inputs
                
                print(f"\n模型输入形状: {inputs_patch.shape}")
                print(f"期望模型输出: {cfg.out_len}帧预测")
                
                # 推理 - 现在模型应该只接收13帧输入
                predictions, _ = model([inputs_patch, 0, cfg.epoch], mode='test')
                
                print(f"模型输出 predictions形状: {predictions.shape}")
                
                # 检查预测长度
                if predictions.shape[0] < cfg.out_len:
                    print(f"警告：模型只返回了{predictions.shape[0]}帧，但需要{cfg.out_len}帧")
                    # 可能需要截取或重复
                    if predictions.shape[0] > 0:
                        predictions = predictions[:cfg.out_len]
                    else:
                        # 如果模型没有返回预测，创建零预测
                        predictions = torch.zeros(cfg.out_len, *predictions.shape[1:]).to(device)
                
                # ========== 方案B：如果模型仍然需要25帧输入 ==========
                # 取消注释以下代码如果方案A不工作
                
                # # 创建包含13帧输入和12帧零目标的完整序列
                # dummy_targets = torch.zeros_like(targets)
                # full_sequence = torch.cat([inputs, dummy_targets], dim=0)  # [25, 1, 1, H, W]
                
                # if cfg.reshape_patch:
                #     full_sequence_patch = reshape_patch_TBCHW(full_sequence, cfg.patch_size)
                # else:
                #     full_sequence_patch = full_sequence
                
                # print(f"\n完整序列形状: {full_sequence.shape}")
                # print(f"Patch处理后: {full_sequence_patch.shape}")
                
                # # 推理
                # predictions, _ = model([full_sequence_patch, 0, cfg.epoch], mode='test')
                # print(f"模型输出 predictions形状: {predictions.shape}")
                
                # # 取后12帧作为预测
                # if predictions.shape[0] >= cfg.out_len:
                #     predictions = predictions[-cfg.out_len:]
                # else:
                #     print(f"警告：模型输出帧数不足")
                #     predictions = predictions[:cfg.out_len]
                
                # ========== 后续处理 ==========
                if cfg.reshape_patch:
                    predictions = reshape_patch_back(predictions, cfg.patch_size)
                    print(f"还原patch后 predictions形状: {predictions.shape}")
                
                # 转换为numpy
                inputs_np = inputs.permute(1, 0, 2, 3, 4).cpu().numpy()  # [1, 13, 1, H, W]
                targets_np = targets.permute(1, 0, 2, 3, 4).cpu().numpy()  # [1, 12, 1, H, W]
                preds_np = predictions.permute(1, 0, 2, 3, 4).cpu().numpy()  # [1, 12, 1, H, W]
                
                print(f"\n转换为numpy后:")
                print(f"  inputs_np: {inputs_np.shape}")
                print(f"  targets_np: {targets_np.shape}")
                print(f"  preds_np: {preds_np.shape}")
                
                # 移除通道维度
                inputs_np = inputs_np.squeeze(2)
                targets_np = targets_np.squeeze(2)
                preds_np = preds_np.squeeze(2)
            
            # ... 后续代码保持不变 ...
    # with torch.no_grad():
    #     pbar = tqdm(test_loader, desc="测试进度")
    #     for batch_idx, (inputs, targets) in enumerate(pbar):
    #         if sample_count >= args.num_samples:
    #             break
            
    #         try:
    #             # ========== 添加详细调试信息 ==========
    #             print(f"\n{'='*60}")
    #             print(f"处理批次 {batch_idx+1}, 样本 {sample_count+1}")
    #             print(f"{'='*60}")
                
    #             # 打印原始数据形状
    #             print(f"原始数据形状:")
    #             print(f"  inputs: {inputs.shape} (应该是 [1, 13, 1, H, W])")
    #             print(f"  targets: {targets.shape} (应该是 [1, 12, 1, H, W])")
                
    #             # 打印数据统计
    #             print(f"\n原始数据统计:")
    #             print(f"  inputs范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
    #             print(f"  targets范围: [{targets.min():.3f}, {targets.max():.3f}]")
                
    #             # 调整维度
    #             inputs = inputs.permute(1, 0, 2, 3, 4).to(device)
    #             targets = targets.permute(1, 0, 2, 3, 4).to(device)
                
    #             print(f"\n调整维度后:")
    #             print(f"  inputs: {inputs.shape} (应该是 [13, 1, 1, H, W])")
    #             print(f"  targets: {targets.shape} (应该是 [12, 1, 1, H, W])")
                
    #             # Patch处理
    #             if cfg.reshape_patch:
    #                 inputs_patch = reshape_patch_TBCHW(inputs, cfg.patch_size)
    #                 print(f"\nPatch处理后 inputs_patch形状: {inputs_patch.shape}")
    #             else:
    #                 inputs_patch = inputs
                
    #             # 推理
    #             print(f"\n开始模型推理...")
    #             predictions, _ = model([inputs_patch, 0, cfg.epoch], mode='test')
    #             predictions = predictions[:cfg.out_len]
    #             print(f"模型输出 predictions形状: {predictions.shape}")
                
    #             if cfg.reshape_patch:
    #                 predictions = reshape_patch_back(predictions, cfg.patch_size)
    #                 print(f"还原patch后 predictions形状: {predictions.shape}")
                
    #             # 转换为numpy
    #             inputs_np = inputs.permute(1, 0, 2, 3, 4).cpu().numpy()
    #             targets_np = targets.permute(1, 0, 2, 3, 4).cpu().numpy()
    #             preds_np = predictions.permute(1, 0, 2, 3, 4).cpu().numpy()
                
    #             print(f"\n转换为numpy后:")
    #             print(f"  inputs_np: {inputs_np.shape} (应该是 [1, 13, 1, H, W])")
    #             print(f"  targets_np: {targets_np.shape} (应该是 [1, 12, 1, H, W])")
    #             print(f"  preds_np: {preds_np.shape} (应该是 [1, 12, 1, H, W])")
                
    #             # 移除通道维度
    #             inputs_np = inputs_np.squeeze(2)
    #             targets_np = targets_np.squeeze(2)
    #             preds_np = preds_np.squeeze(2)
                
                print(f"\n移除通道维度后:")
                print(f"  inputs_np: {inputs_np.shape} (应该是 [1, 13, H, W])")
                print(f"  targets_np: {targets_np.shape} (应该是 [1, 12, H, W])")
                print(f"  preds_np: {preds_np.shape} (应该是 [1, 12, H, W])")
                
                # 处理批次中的每个序列
                for seq_idx in range(inputs_np.shape[0]):
                    if sample_count >= args.num_samples:
                        break
                    
                    print(f"\n处理序列 {seq_idx+1}...")
                    
                    input_sample = inputs_np[seq_idx]
                    target_sample = targets_np[seq_idx]
                    pred_sample = preds_np[seq_idx]
                    
                    # 打印详细统计信息
                    print(f"数据统计:")
                    print(f"  输入范围: [{input_sample.min():.3f}, {input_sample.max():.3f}], 均值: {input_sample.mean():.3f}")
                    print(f"  目标范围: [{target_sample.min():.3f}, {target_sample.max():.3f}], 均值: {target_sample.mean():.3f}")
                    print(f"  预测范围（处理前）: [{pred_sample.min():.3f}, {pred_sample.max():.3f}], 均值: {pred_sample.mean():.3f}")
                    
                    # 计算输入最后一帧和目标第一帧的相似度
                    last_input = input_sample[-1]  # 输入的第13帧
                    first_target = target_sample[0]  # 目标的第1帧
                    first_pred = pred_sample[0]  # 预测的第1帧
                    
                    # 计算MSE相似度
                    mse_input_target = np.mean((last_input - first_target) ** 2)
                    mse_input_pred = np.mean((last_input - first_pred) ** 2)
                    mse_target_pred = np.mean((first_target - first_pred) ** 2)
                    
                    print(f"\n相似度分析:")
                    print(f"  输入最后一帧 vs 目标第一帧 MSE: {mse_input_target:.6f}")
                    print(f"  输入最后一帧 vs 预测第一帧 MSE: {mse_input_pred:.6f}")
                    print(f"  目标第一帧 vs 预测第一帧 MSE: {mse_target_pred:.6f}")
                    
                    if mse_input_pred < mse_target_pred:
                        print(f"  ⚠️ 预测更像输入！差异: {mse_target_pred - mse_input_pred:.6f}")
                    
                    # 后处理
                    pred_processed = postprocess_prediction(pred_sample, args.postprocess)
                    print(f"  预测范围（处理后）: [{pred_processed.min():.3f}, {pred_processed.max():.3f}], 均值: {pred_processed.mean():.3f}")
                    
                    # 计算指标
                    thresholds = [10, 15, 20, 25, 30, 35, 40]
                    metrics = compute_metrics(pred_processed, target_sample, thresholds)
                    
                    all_metrics.append(metrics)
                    
                    # 打印关键指标
                    print(f"关键指标:")
                    for th in [10, 20, 30, 40]:
                        csi = metrics.get(f'CSI_{th}', 0)
                        pod = metrics.get(f'POD_{th}', 0)
                        far = metrics.get(f'FAR_{th}', 0)
                        bias = metrics.get(f'BIAS_{th}', 0)
                        print(f"  {th:2d}dBZ: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}, BIAS={bias:.3f}")
                    
                    # 可视化
                    if args.visualize:
                        save_path = plot_sequence_comparison(
                            input_sample, target_sample, pred_processed,
                            sample_count, output_dir
                        )
                        if save_path:
                            print(f"  图像保存: {save_path}")
                    
                    sample_count += 1
                    pbar.set_description(f"测试进度: {sample_count}/{args.num_samples}")
                    
            except Exception as e:
                print(f"推理错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    if all_metrics:
        df, avg_metrics = save_results(all_metrics, output_dir)
        print(f"\n测试完成! 输出目录: {output_dir}")
    else:
        print("没有处理任何样本!")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    main()