import os
import torch
import torch.nn as nn
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
print("重庆雷达数据MIM外推测试")
print("=" * 60)
print(f"模型: MIM (Memory In Memory)")
print(f"输入帧: {cfg.in_len}, 输出帧: {cfg.out_len}")
print(f"图像: {cfg.width}x{cfg.height}")
print("=" * 60)

# 配置
cfg.dataset = 'Chongqing'
cfg.batch = 1
cfg.reshape_patch = True

# 导入MIM模型和相关模块
sys.path.append(os.path.join('/data_8t/zhouying/msrnn', 'models'))
from mim import MIM
from load_chongqing import load_chongqing_test

# 导入model_new中的make_layers
try:
    from util.utils import make_layers
    print("成功导入make_layers")
except ImportError:
    print("警告：无法导入make_layers，使用简化版本")
    def make_layers(param):
        """简化版的make_layers函数"""
        if isinstance(param, dict):
            for key, value in param.items():
                if key == 'conv_embed' or key == 'conv_fc':
                    in_channels, out_channels, kernel_size, stride, padding, groups = value
                    return nn.Conv2d(in_channels, out_channels, kernel_size, 
                                    stride=stride, padding=padding, groups=groups)
        return None


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
    # 输入可能是 [T, C, H, W] 或 [T, B, C, H, W]
    if x.dim() == 4:
        # 添加batch维度 [T, C, H, W] -> [T, 1, C, H, W]
        x = x.unsqueeze(1)
    
    T, B, C_p, H_p, W_p = x.shape
    C = C_p // (patch_size * patch_size)
    
    x = x.permute(0, 1, 3, 4, 2)
    x = x.reshape(T, B, H_p, W_p, patch_size, patch_size, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(T, B, H_p * patch_size, W_p * patch_size, C)
    x = x.permute(0, 1, 4, 2, 3)
    
    # 移除batch维度
    if B == 1:
        x = x.squeeze(1)
    
    return x


def postprocess_prediction(pred, method='clip'):
    """
    后处理预测结果
    """
    if method == 'clip':
        pred = np.clip(pred, 0.0, 1.0)
    elif method == 'shift':
        if pred.min() < 0:
            pred = pred - pred.min()
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
    pred_dbz = pred * 80.0
    target_dbz = target * 80.0
    
    pred_dbz = np.nan_to_num(pred_dbz, nan=0.0)
    target_dbz = np.nan_to_num(target_dbz, nan=0.0)
    pred_dbz = np.clip(pred_dbz, 0, 80)
    target_dbz = np.clip(target_dbz, 0, 80)
    
    eps = 1e-10
    results = {}
    
    for th in thresholds:
        pred_bin = (pred_dbz >= th).astype(np.float32)
        target_bin = (target_dbz >= th).astype(np.float32)
        
        TP = np.logical_and(pred_bin == 1, target_bin == 1).sum()
        FP = np.logical_and(pred_bin == 1, target_bin == 0).sum()
        FN = np.logical_and(pred_bin == 0, target_bin == 1).sum()
        
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
    绘制序列对比图
    """
    try:
        input_dbz = input_seq * 80.0
        target_dbz = target_seq * 80.0
        pred_dbz = pred_seq * 80.0
        
        input_frames = input_seq.shape[0]
        target_frames = target_seq.shape[0]
        pred_frames = pred_seq.shape[0]
        
        max_cols = input_frames
        
        fig, axes = plt.subplots(3, max_cols, figsize=(max_cols * 1.5, 6))
        
        if max_cols == 1:
            axes = axes.reshape(3, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(3, -1)
        
        fig.patch.set_facecolor('white')
        
        def plot_radar(data, ax, title):
            try:
                data_clipped = np.clip(data, 0, 80)
                mask_0_5 = (data_clipped >= 0) & (data_clipped < 5)
                
                contour = ax.contourf(data_clipped, levels=RADAR_LEVELS, cmap=RADAR_CMAP, extend='max')
                
                if mask_0_5.any():
                    white_mask = np.zeros_like(data_clipped, dtype=bool)
                    white_mask[mask_0_5] = True
                    ax.contourf(white_mask, levels=[0.5, 1], colors=['white'], alpha=1.0)
                
                ax.set_title(title, fontsize=8, pad=2)
                ax.set_aspect('equal')
                ax.axis('off')
                ax.set_facecolor('white')
                return contour
            except Exception as e:
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
        
        for i in range(input_frames):
            plot_radar(input_dbz[i], axes[0, i], f'In {i}')
        
        for i in range(target_frames):
            plot_radar(target_dbz[i], axes[1, i+1], f'Tar {i}')
        
        for i in range(pred_frames):
            plot_radar(pred_dbz[i], axes[2, i+1], f'Pred {i}')
        
        for i in range(max_cols):
            if i == 0:
                axes[1, i].axis('off')
                axes[2, i].axis('off')
            elif i > target_frames:
                axes[1, i].axis('off')
                axes[2, i].axis('off')
        
        plt.suptitle(f'MIM Prediction - Sample {sample_idx+1}', fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
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
    
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, 'metrics_results.csv')
    df.to_csv(csv_path, index=False)
    
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
    
    avg_df = pd.DataFrame([avg_metrics])
    avg_csv_path = os.path.join(output_dir, 'average_metrics.csv')
    avg_df.to_csv(avg_csv_path, index=False)
    
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


class MIM_Model(nn.Module):
    """
    MIM模型的完整封装 - 按照model_new.py的结构
    """
    def __init__(self, embed, rnn, fc):
        super().__init__()
        self.embed = make_layers(embed)
        self.rnns = rnn  # MIM模型
        self.fc = make_layers(fc)
        
    def forward(self, x, m, layer_hiddens, mode='test'):
        """
        前向传播 - 按照model_new.py的测试模式逻辑
        x: 输入张量 [B, C, H, W]
        m: memory状态
        layer_hiddens: 各层隐藏状态
        """
        # 确保输入是4D: [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # 调用MIM模型
        output, m, layer_hiddens, decouple_loss = self.rnns(x, m, layer_hiddens, self.embed, self.fc)
        
        return output, m, layer_hiddens


def main():
    parser = argparse.ArgumentParser(description='重庆雷达数据MIM测试')
    parser.add_argument('--resume', type=str, 
                       default='/data_8t/zhouying/msrnn/modelBak/01MIM_ps4_hs_64_l6_test0609.pth',
                       help='预训练模型路径')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='测试样本数量')
    parser.add_argument('--save_dir', type=str, default='./test_results_mim',
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
    
    print(f"加载MIM模型...")
    
    # 导入net_params来获取nets参数
    from net_params import nets
    
    # 检查nets参数
    print(f"nets结构:")
    print(f"  nets[0] (embed): {nets[0]}")
    print(f"  nets[2] (fc): {nets[2]}")
    
    # 根据patch处理调整b_h_w参数
    if cfg.reshape_patch:
        patch_size = cfg.patch_size
        H = cfg.height // patch_size
        W = cfg.width // patch_size
        input_channel = patch_size * patch_size  # 16
        output_channel = patch_size * patch_size  # 16
    else:
        H = cfg.height
        W = cfg.width
        input_channel = 1
        output_channel = 1
    
    # 模型参数
    b_h_w = (cfg.batch, H, W)
    
    print(f"MIM模型参数:")
    print(f"  input_channel: {input_channel}")
    print(f"  output_channel: {output_channel}")
    print(f"  b_h_w: {b_h_w}")
    print(f"  kernel_size: {cfg.kernel_size}")
    
    # 创建MIM模型（作为rnn部分）
    mim_rnn = MIM(input_channel=cfg.lstm_hidden_state,  # 64，由embed层输出
                  output_channel=cfg.lstm_hidden_state,  # 64，输入到fc层
                  b_h_w=b_h_w,
                  kernel_size=cfg.kernel_size,
                  stride=1,
                  padding=cfg.kernel_size//2)
    
    # 创建完整的MIM_Model（按照model_new.py的结构）
    model = MIM_Model(embed=nets[0],  # OrderedDict([('conv_embed', [16, 64, 1, 1, 0, 1])])
                      rnn=mim_rnn,    # MIM模型
                      fc=nets[2])     # OrderedDict([('conv_fc', [64, 16, 1, 1, 0, 1])])
    
    model = model.to(device)
    
    # 加载预训练权重
    print(f"加载权重: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    else:
        state_dict = remove_module_prefix(checkpoint)
    
    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        print("模型权重加载成功")
    except Exception as e:
        print(f"权重加载错误: {e}")
        # print("尝试调整权重加载...")
        
        # # 手动调整权重键名
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     # 将module.rnns.lstm改为module.rnns.lstm（保持一致）
        #     # 将module.embed.conv_embed改为embed.conv_embed
        #     # 将module.fc.conv_fc改为fc.conv_fc
        #     new_key = k
        #     if new_key.startswith('module.embed.'):
        #         new_key = new_key.replace('module.embed.', 'embed.')
        #     elif new_key.startswith('module.rnns.'):
        #         new_key = new_key.replace('module.rnns.', 'rnns.')
        #     elif new_key.startswith('module.fc.'):
        #         new_key = new_key.replace('module.fc.', 'fc.')
        #     new_state_dict[new_key] = v
        
        # 再次尝试加载
        # try:
        #     model.load_state_dict(new_state_dict, strict=False)
        #     print("权重加载成功（非严格模式）")
        # except Exception as e2:
        #     print(f"第二次加载错误: {e2}")
        #     print("尝试只加载匹配的参数...")
            
        #     # 只加载形状匹配的参数
        #     model_dict = model.state_dict()
        #     matched_keys = []
        #     for k, v in new_state_dict.items():
        #         if k in model_dict and v.shape == model_dict[k].shape:
        #             model_dict[k] = v
        #             matched_keys.append(k)
            
        #     model.load_state_dict(model_dict, strict=False)
        #     print(f"加载了 {len(matched_keys)}/{len(model_dict)} 个参数")
    
    model.eval()
    print("模型加载完成")
    
    # 加载测试数据
    print("加载测试数据...")
    test_loader = load_chongqing_test(batch_size=1)
    
    if test_loader is None:
        print("数据加载失败")
        return
    
    # 开始测试
    all_metrics = []
    sample_count = 0
    
    print(f"\n开始测试 {args.num_samples} 个样本...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试进度")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if sample_count >= args.num_samples:
                break
            
            try:
                # 调整维度 [B, T, C, H, W] -> [T, B, C, H, W]
                inputs = inputs.permute(1, 0, 2, 3, 4).to(device)
                targets = targets.permute(1, 0, 2, 3, 4).to(device)
                
                # Patch处理
                if cfg.reshape_patch:
                    inputs_patch = reshape_patch_TBCHW(inputs, cfg.patch_size)
                else:
                    inputs_patch = inputs
                
                print(f"\n样本 {sample_count+1}:")
                print(f"输入patch形状: {inputs_patch.shape}")
                print(f"目标形状: {targets.shape}")
                
                # 初始化MIM状态
                m = None
                layer_hiddens = None
                
                # 第一步：处理所有输入帧（建立隐藏状态）
                print(f"处理输入帧建立隐藏状态...")
                for t in range(cfg.in_len):
                    input_t = inputs_patch[t]  # [B, C, H, W]
                    _, m, layer_hiddens = model(input_t, m, layer_hiddens, mode='test')
                
                # 第二步：进行out_len步自回归预测
                print(f"进行自回归预测...")
                predictions = []
                current_input = inputs_patch[cfg.in_len - 1]  # 使用最后一帧输入作为起始
                
                for t in range(cfg.out_len):
                    # 单步预测
                    output, m, layer_hiddens = model(current_input, m, layer_hiddens, mode='test')
                    predictions.append(output)
                    
                    # 使用预测结果作为下一步的输入（自回归）
                    current_input = output
                
                # 堆叠预测结果
                predictions = torch.stack(predictions, dim=0)  # [T, B, C, H, W]
                print(f"预测形状: {predictions.shape}")
                
                # 还原patch
                if cfg.reshape_patch:
                    predictions = reshape_patch_back(predictions, cfg.patch_size)
                    print(f"patch还原后形状: {predictions.shape}")
                
                # 确保predictions有正确的维度 [T, B, C, H, W]
                if predictions.dim() == 4:
                    predictions = predictions.unsqueeze(1)  # [T, C, H, W] -> [T, 1, C, H, W]
                
                # 转换为numpy
                inputs_np = inputs.permute(1, 0, 2, 3, 4).cpu().numpy()  # [B, T, C, H, W]
                targets_np = targets.permute(1, 0, 2, 3, 4).cpu().numpy()
                preds_np = predictions.permute(1, 0, 2, 3, 4).cpu().numpy()
                
                # 移除通道维度
                inputs_np = inputs_np.squeeze(2)  # [B, T, H, W]
                targets_np = targets_np.squeeze(2)
                preds_np = preds_np.squeeze(2)
                
                # 处理批次中的每个序列
                for seq_idx in range(inputs_np.shape[0]):
                    if sample_count >= args.num_samples:
                        break
                    
                    print(f"处理样本 {sample_count+1}...")
                    
                    input_sample = inputs_np[seq_idx]
                    target_sample = targets_np[seq_idx]
                    pred_sample = preds_np[seq_idx]
                    
                    # 打印数据范围
                    print(f"输入范围: [{input_sample.min():.3f}, {input_sample.max():.3f}]")
                    print(f"目标范围: [{target_sample.min():.3f}, {target_sample.max():.3f}]")
                    print(f"预测范围（处理前）: [{pred_sample.min():.3f}, {pred_sample.max():.3f}]")
                    
                    # 后处理
                    pred_processed = postprocess_prediction(pred_sample, args.postprocess)
                    print(f"预测范围（处理后）: [{pred_processed.min():.3f}, {pred_processed.max():.3f}]")
                    
                    # 计算指标
                    thresholds = [10, 15, 20, 25, 30, 35, 40]
                    metrics = compute_metrics(pred_processed, target_sample, thresholds)
                    
                    all_metrics.append(metrics)
                    
                    # 打印关键指标
                    print(f"样本 {sample_count+1} 关键指标:")
                    for th in [10, 20, 30]:
                        csi = metrics.get(f'CSI_{th}', 0)
                        pod = metrics.get(f'POD_{th}', 0)
                        far = metrics.get(f'FAR_{th}', 0)
                        bias = metrics.get(f'BIAS_{th}', 0)
                        print(f"  {th}dBZ: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}, BIAS={bias:.3f}")
                    
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
                import traceback
                print(f"\n推理错误: {e}")
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