# /data_8t/zhouying/msrnn/config.py
from util.ordered_easydict import OrderedEasyDict as edict
import os
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np

# 初始化配置
cfg = edict()

# ========== 模型配置 ==========
cfg.model_name = 'ConvLSTM'  # 使用ConvLSTM模型
cfg.gpu = '0'  # 单卡测试
cfg.gpu_nums = 1

# ========== 路径配置 ==========
cfg.work_path = 'experiments'
cfg.dataset = 'Chongqing'  # 重庆数据集
cfg.data_path = 'Spatiotemporal'  # 根据数据集类型

# ========== 模型参数 ==========
cfg.lstm_hidden_state = 64  # 隐藏状态维度
cfg.kernel_size = 3  # 卷积核大小
cfg.batch = 2  # 批次大小
cfg.LSTM_conv = Conv2d
cfg.LSTM_deconv = ConvTranspose2d
cfg.CONV_conv = Conv2d

# ========== 数据集配置 ==========
# 重庆雷达数据配置
cfg.width = 384      # 图像宽度
cfg.height = 384     # 图像高度  
cfg.in_len = 13      # 输入序列长度
cfg.out_len = 12     # 输出序列长度（预测长度）
cfg.epoch = 1       # 训练轮数

# ========== 训练参数 ==========
cfg.early_stopping = True
cfg.early_stopping_patience = 20
cfg.valid_num = int(cfg.epoch * 1)
cfg.valid_epoch = cfg.epoch // cfg.valid_num
cfg.LR = 0.0001
cfg.optimizer = 'AdamW'
cfg.dataloader_thread = 4
cfg.data_type = np.float32

# ========== 采样策略 ==========
cfg.scheduled_sampling = False
cfg.reverse_scheduled_sampling = False

# ========== 其他模型参数 ==========
cfg.TrajGRU_link_num = 10
cfg.ce_iters = 5
cfg.decouple_loss_weight = 0.01
cfg.la_num = 30
cfg.LSTM_layers = 6  # LSTM层数
cfg.metrics_decimals = 3

# ========== 路径配置 ==========
cfg.root_path = 'MS-RNN-main'

cfg.GLOBAL = edict()
cfg.GLOBAL.MODEL_LOG_SAVE_PATH = os.path.join(cfg.work_path, 'save', cfg.dataset, cfg.model_name)
cfg.GLOBAL.DATASET_PATH = os.path.join(cfg.root_path, cfg.data_path, 'dataset', cfg.dataset)

# ========== 重庆数据特定配置 ==========
cfg.Chongqing = edict()
# 雷达反射率阈值（用于指标计算）
cfg.Chongqing.THRESHOLDS = [0, 10, 20, 30, 40]  # dBZ单位
# 雷达颜色映射
cfg.Chongqing.RADAR_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
cfg.Chongqing.RADAR_COLORS = [
    '#0000FF', '#00C8FF', '#00FF00', '#32CD32', '#008000',
    '#FFFF00', '#FFD700', '#FFA500', '#FF6347', '#FF0000',
    '#C80000', '#FF69B4', '#800080', '#DDA0DD'
]
# 高度层配置
cfg.Chongqing.HEIGHT_LAYERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 
                              8.0, 9.0, 10.0, 12.0, 14.0, 15.5, 17.0, 19.0]
cfg.Chongqing.TARGET_HEIGHT_IDX = 6  # 3.5km高度层

# ========== 解码器配置 ==========
cfg.decoder_seed = 'zero'

# ========== Patch配置 ==========
cfg.reshape_patch = True  # 启用patch化
cfg.patch_size = 4        # patch大小

# ========== 其他配置 ==========
cfg.kth_only_run = False
cfg.eval_len = 10

# print(f"配置加载完成: 数据集={cfg.dataset}, 模型={cfg.model_name}")
# print(f"输入长度={cfg.in_len}, 输出长度={cfg.out_len}")
# print(f"图像尺寸={cfg.width}x{cfg.height}, patch大小={cfg.patch_size}")