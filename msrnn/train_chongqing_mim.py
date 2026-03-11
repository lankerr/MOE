import os
import torch
import numpy as np
import bz2
import cinrad.io
import cv2
import glob
import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from config import cfg


class ChongqingRadarDataset(Dataset):
    """重庆天气雷达数据集加载器"""
    
    def __init__(self, data_dirs, mode='train', seq_len=25, target_height_idx=6, 
                 transform=None, debug=False):
        """
        参数:
            data_dirs: 数据目录列表
            mode: 'train', 'valid', 'test'
            seq_len: 序列总长度 (13输入 + 12输出 = 25)
            target_height_idx: 目标高度层索引，默认6对应3.5km
            transform: 数据增强变换
            debug: 调试模式，打印更多信息
        """
        self.mode = mode
        self.seq_len = seq_len
        self.input_len = cfg.in_len  # 13
        self.output_len = cfg.out_len  # 12
        self.img_size = (cfg.width, cfg.height)
        self.target_height_idx = target_height_idx
        self.transform = transform
        self.debug = debug
        
        if debug:
            print(f"\n初始化数据集: 模式={mode}")
            print(f"  序列总长度={seq_len} (输入{self.input_len}+输出{self.output_len})")
            print(f"  图像尺寸={self.img_size}, 高度层索引={target_height_idx}")
            print(f"  数据目录: {data_dirs}")
        
        # 收集所有雷达数据文件
        self.data_files = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                if debug:
                    print(f"警告: 目录不存在: {data_dir}")
                continue
                
            pattern = os.path.join(data_dir, "**", "*.bin.DATA_DBZ.bz2")
            files = glob.glob(pattern, recursive=True)
            
            if debug and files:
                print(f"  目录 {data_dir}: 找到 {len(files)} 个文件")
                
            files.sort(key=lambda x: self._extract_time_from_filename(x))
            self.data_files.extend(files)
        
        if debug:
            print(f"总共找到 {len(self.data_files)} 个雷达数据文件")
        
        if not self.data_files:
            print("错误: 没有找到任何数据文件！")
            print("请检查数据路径:", data_dirs)
            return
        
        # 创建序列索引
        self.sequences = self._create_sequences()
        
        if debug:
            print(f"创建了 {len(self.sequences)} 个序列样本")
            if self.sequences:
                # 显示第一个序列的时间信息
                first_seq = self.sequences[0]
                start_time = self._extract_time_from_filename(first_seq[0])
                end_time = self._extract_time_from_filename(first_seq[-1])
                print(f"第一个序列时间范围: {start_time} 到 {end_time}")
                print(f"持续时间: {(end_time - start_time).total_seconds()/60:.1f}分钟")
        
        # 统计数据信息（用于归一化）
        self._compute_statistics()
    
    def _extract_time_from_filename(self, filename):
        """从文件名中提取时间信息"""
        try:
            basename = os.path.basename(filename)
            
            if 'RADAMOSAIC_' in basename and '.bin' in basename:
                start_idx = basename.find('RADAMOSAIC_') + len('RADAMOSAIC_')
                end_idx = basename.find('.bin')
                time_str = basename[start_idx:end_idx]
                
                if len(time_str) == 14 and time_str.isdigit():
                    return datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S")
            
            # 尝试其他可能的格式
            if 'SWAN' in basename:
                # 尝试提取YYYYMMDD_HHMMSS格式
                import re
                match = re.search(r'(\d{8}_\d{6})', basename)
                if match:
                    time_str = match.group(1).replace('_', '')
                    if len(time_str) == 14:
                        return datetime.datetime.strptime(time_str, "%Y%m%d%H%M%S")
            
            return datetime.datetime(2000, 1, 1)
        except Exception as e:
            if self.debug:
                print(f"解析文件名 {basename} 时出错: {e}")
            return datetime.datetime(2000, 1, 1)
    
    def _create_sequences(self):
        """创建连续的序列索引"""
        sequences = []
        
        if not self.data_files:
            return sequences
        
        # 按时间排序
        sorted_files = sorted(self.data_files, key=lambda x: self._extract_time_from_filename(x))
        
        if self.debug:
            print(f"\n排序后文件示例 (前5个):")
            for i, f in enumerate(sorted_files[:5]):
                time = self._extract_time_from_filename(f)
                print(f"  {i+1}. {os.path.basename(f)} - {time}")
        
        # 按6分钟间隔分组序列
        current_sequence = []
        sequence_groups = []
        
        for i, file_path in enumerate(sorted_files):
            if not current_sequence:
                current_sequence.append(file_path)
                continue
            
            # 检查时间连续性（6分钟间隔）
            prev_time = self._extract_time_from_filename(current_sequence[-1])
            curr_time = self._extract_time_from_filename(file_path)
            time_diff = (curr_time - prev_time).total_seconds() / 60  # 分钟
            
            # 允许±1分钟的误差
            if abs(time_diff - 6) < 2:
                current_sequence.append(file_path)
            else:
                if len(current_sequence) >= self.seq_len:
                    sequence_groups.append(current_sequence)
                elif self.debug and len(current_sequence) > 0:
                    print(f"  丢弃短序列: {len(current_sequence)}帧 (< {self.seq_len})")
                current_sequence = [file_path]
        
        # 处理最后一个序列
        if len(current_sequence) >= self.seq_len:
            sequence_groups.append(current_sequence)
        
        if self.debug:
            print(f"找到 {len(sequence_groups)} 个连续序列组")
        
        # 从每个序列组中生成固定长度的子序列
        all_sequences = []
        for seq_group in sequence_groups:
            # 使用滑动窗口生成序列
            if len(seq_group) >= self.seq_len:
                if self.mode == 'train':
                    # 训练模式：使用重叠窗口增加数据量
                    step = max(1, self.seq_len // 3)  # 75%重叠
                else:
                    # 验证/测试模式：使用非重叠或小重叠窗口
                    step = max(1, self.seq_len)
                
                for i in range(0, len(seq_group) - self.seq_len + 1, step):
                    all_sequences.append(seq_group[i:i + self.seq_len])
        
        if self.debug:
            print(f"生成 {len(all_sequences)} 个训练样本")
            if all_sequences:
                # 检查序列的时间连续性
                sample_seq = all_sequences[0]
                times = [self._extract_time_from_filename(f) for f in sample_seq]
                time_diffs = [(times[i+1] - times[i]).total_seconds()/60 for i in range(len(times)-1)]
                avg_diff = np.mean(time_diffs)
                print(f"样本序列平均时间间隔: {avg_diff:.1f}分钟")
        
        return all_sequences
    
    def _compute_statistics(self):
        """计算数据统计信息（用于分析）"""
        if not self.sequences:
            return
        
        # 抽样计算统计信息
        sample_size = min(100, len(self.sequences))
        sampled_sequences = np.random.choice(len(self.sequences), sample_size, replace=False)
        
        all_values = []
        for idx in sampled_sequences:
            file_sequence = self.sequences[idx]
            # 读取中间一帧
            mid_idx = len(file_sequence) // 2
            file_path = file_sequence[mid_idx]
            try:
                data = self._read_radar_data_simple(file_path)
                all_values.append(data.flatten())
            except:
                continue
        
        if all_values:
            all_values = np.concatenate(all_values)
            valid_values = all_values[all_values > 0]  # 只考虑有效值
            
            if len(valid_values) > 0:
                self.mean_intensity = np.mean(valid_values)
                self.std_intensity = np.std(valid_values)
                self.percentiles = np.percentile(valid_values, [10, 25, 50, 75, 90, 95])
                
                if self.debug:
                    print(f"\n数据统计信息 (基于{sample_size}个样本):")
                    print(f"  平均值: {self.mean_intensity:.3f}")
                    print(f"  标准差: {self.std_intensity:.3f}")
                    print(f"  百分位数: 10%={self.percentiles[0]:.3f}, "
                          f"50%={self.percentiles[2]:.3f}, "
                          f"90%={self.percentiles[4]:.3f}")
                    print(f"  有效值比例: {len(valid_values)/len(all_values)*100:.1f}%")
            else:
                self.mean_intensity = 15.0
                self.std_intensity = 10.0
                self.percentiles = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        else:
            self.mean_intensity = 15.0
            self.std_intensity = 10.0
            self.percentiles = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    
    def _read_radar_data_simple(self, file_path):
        """快速读取雷达数据（用于统计计算）"""
        try:
            f = cinrad.io.SWAN(file_path, product="CR")
            if f.data is None or len(f.data) == 0:
                return np.zeros(self.img_size, dtype=np.float32)
            
            if len(f.data) <= self.target_height_idx:
                return np.zeros(self.img_size, dtype=np.float32)
            
            height_data = f.data[self.target_height_idx]
            height_data = np.nan_to_num(height_data, nan=-999.0)
            mask_valid = height_data >= 0
            height_data[~mask_valid] = 0.0
            height_data = np.clip(height_data, 0, 80.0)
            
            # 简单调整尺寸
            if height_data.shape != self.img_size:
                height_data = cv2.resize(height_data, self.img_size, 
                                        interpolation=cv2.INTER_NEAREST)
            
            return height_data
        except:
            return np.zeros(self.img_size, dtype=np.float32)
    
    def _create_empty_data(self):
        """创建空数据"""
        return np.zeros(self.img_size, dtype=np.float32)
    
    def _read_radar_data(self, file_path):
        """读取单个雷达数据文件并预处理"""
        try:
            # 使用cinrad读取SWAN数据
            f = cinrad.io.SWAN(file_path, product="CR")
            
            # 检查数据有效性
            if f.data is None or len(f.data) == 0:
                if self.debug and np.random.random() < 0.01:  # 随机采样打印
                    print(f"  空数据文件: {os.path.basename(file_path)}")
                return self._create_empty_data()
            
            # 确保有足够的高度层
            if len(f.data) <= self.target_height_idx:
                if self.debug:
                    print(f"  高度层不足: {len(f.data)} < {self.target_height_idx}")
                return self._create_empty_data()
            
            # 提取指定高度层数据
            height_data = f.data[self.target_height_idx]
            
            # 处理NaN值和无效数据
            height_data = np.nan_to_num(height_data, nan=-999.0)
            
            # 移除填充值（负值通常表示无数据）
            mask_valid = height_data >= 0
            height_data[~mask_valid] = 0.0
            
            # 限制在合理范围
            height_data = np.clip(height_data, 0, 80.0)
            
            # 调整尺寸
            height_data = cv2.resize(height_data, self.img_size, 
                                    interpolation=cv2.INTER_LINEAR)
            
            # 归一化到[0,1]
            height_data = height_data / 80.0
            
            # 确保在[0,1]范围内
            height_data = np.clip(height_data, 0, 1)
            
            return height_data.astype(np.float32)
            
        except Exception as e:
            if self.debug and np.random.random() < 0.01:
                print(f"读取文件 {os.path.basename(file_path)} 时出错: {e}")
            return self._create_empty_data()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取一个序列样本"""
        file_sequence = self.sequences[idx]
        
        # 读取并预处理所有帧
        frames = []
        for i, file_path in enumerate(file_sequence):
            frame_data = self._read_radar_data(file_path)
            frames.append(frame_data)
        
        # 转换为numpy数组
        sequence_array = np.stack(frames)  # [seq_len, H, W]
        
        # 数据增强（仅在训练模式下）
        if self.transform and self.mode == 'train':
            sequence_array = self.transform(sequence_array)
        
        # 转换为tensor
        sequence_tensor = torch.from_numpy(sequence_array).float()
        
        # 分割输入和目标
        input_seq = sequence_tensor[:self.input_len]  # 前13帧
        target_seq = sequence_tensor[self.input_len:self.input_len+self.output_len]  # 后12帧
        
        # 添加通道维度 [seq_len, 1, H, W]
        input_seq = input_seq.unsqueeze(1)
        target_seq = target_seq.unsqueeze(1)
        
        return input_seq, target_seq


# ==================== 数据增强类 ====================
class ChongqingAugmentation:
    """重庆雷达数据增强类"""
    def __init__(self, p=0.5):
        self.p = p  # 应用增强的概率
    
    def __call__(self, sequence):
        """
        sequence: [T, H, W]
        返回增强后的序列
        """
        if np.random.random() > self.p:
            return sequence
        
        augmented = sequence.copy()
        
        # 1. 随机水平翻转
        if np.random.random() < 0.5:
            augmented = np.flip(augmented, axis=2)  # 水平翻转
        
        # 2. 随机垂直翻转
        if np.random.random() < 0.5:
            augmented = np.flip(augmented, axis=1)  # 垂直翻转
        
        # 3. 随机旋转90度
        k = np.random.randint(0, 4)
        if k > 0:
            augmented = np.rot90(augmented, k=k, axes=(1, 2))
        
        # 4. 亮度/对比度调整
        if np.random.random() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)  # 对比度
            beta = np.random.uniform(-0.05, 0.05)  # 亮度
            augmented = alpha * augmented + beta
            augmented = np.clip(augmented, 0, 1)
        
        # 5. 随机裁剪（重庆特定：保留中心区域）
        if np.random.random() < 0.3 and sequence.shape[1] > 256 and sequence.shape[2] > 256:
            h, w = sequence.shape[1], sequence.shape[2]
            crop_h, crop_w = 256, 256
            start_h = np.random.randint(0, h - crop_h)
            start_w = np.random.randint(0, w - crop_w)
            augmented = augmented[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # 调整回原始尺寸
            if (crop_h, crop_w) != sequence.shape[1:]:
                augmented_resized = np.zeros_like(sequence)
                for t in range(augmented.shape[0]):
                    augmented_resized[t] = cv2.resize(
                        augmented[t], (w, h), interpolation=cv2.INTER_LINEAR
                    )
                augmented = augmented_resized
        
        return augmented


# ==================== 数据加载函数 ====================
def load_chongqing_train(batch_size=None, num_workers=None, debug=False):
    """加载训练集"""
    if batch_size is None:
        batch_size = cfg.batch
    if num_workers is None:
        num_workers = cfg.dataloader_thread
    
    # 数据根目录
    data_root = "/data_8t/WSG/data/chongqing/TDMOSAIC"
    
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        return None
    
    print(f"\n加载重庆雷达训练数据...")
    print(f"数据路径: {data_root}")
    
    # 收集所有日期目录
    date_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            date_dirs.append(item_path)
    
    if not date_dirs:
        print(f"警告: 未找到日期目录，尝试直接使用根目录")
        date_dirs = [data_root]
    
    date_dirs.sort()
    print(f"找到 {len(date_dirs)} 个日期目录")
    
    # 数据划分：前70%训练，中间15%验证，后15%测试
    n_total = len(date_dirs)
    n_train = int(n_total * 0.70)
    train_dirs = date_dirs[:n_train]
    
    print(f"训练数据: {len(train_dirs)} 天 ({n_train/n_total*100:.1f}%)")
    for i, dir_path in enumerate(train_dirs[:3]):
        print(f"  {i+1}. {os.path.basename(dir_path)}")
    if len(train_dirs) > 3:
        print(f"  ... 和 {len(train_dirs)-3} 个更多")
    
    # 创建数据增强
    transform = ChongqingAugmentation(p=0.3) if cfg.scheduled_sampling else None
    
    # 创建训练数据集
    try:
        train_dataset = ChongqingRadarDataset(
            train_dirs, mode='train', transform=transform, debug=debug
        )
    except Exception as e:
        print(f"创建训练数据集时出错: {e}")
        print("尝试使用前50%的数据...")
        train_dirs = date_dirs[:len(date_dirs)//2]
        train_dataset = ChongqingRadarDataset(train_dirs, mode='train', debug=debug)
    
    if len(train_dataset) == 0:
        print("错误: 训练数据集为空！")
        return None
    
    print(f"训练集包含 {len(train_dataset)} 个样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    
    return train_loader


def load_chongqing_valid(batch_size=None, num_workers=None, debug=False):
    """加载验证集"""
    if batch_size is None:
        batch_size = cfg.batch
    if num_workers is None:
        num_workers = cfg.dataloader_thread
    
    data_root = "/data_8t/WSG/data/chongqing/TDMOSAIC"
    
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        return None
    
    print(f"\n加载重庆雷达验证数据...")
    
    # 收集所有日期目录
    date_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            date_dirs.append(item_path)
    
    if not date_dirs:
        print(f"警告: 未找到日期目录")
        return None
    
    date_dirs.sort()
    
    # 数据划分：前70%训练，中间15%验证，后15%测试
    n_total = len(date_dirs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    val_dirs = date_dirs[n_train:n_train+n_val]
    
    print(f"验证数据: {len(val_dirs)} 天 ({n_val/n_total*100:.1f}%)")
    for i, dir_path in enumerate(val_dirs[:3]):
        print(f"  {i+1}. {os.path.basename(dir_path)}")
    if len(val_dirs) > 3:
        print(f"  ... 和 {len(val_dirs)-3} 个更多")
    
    # 创建验证数据集
    try:
        val_dataset = ChongqingRadarDataset(
            val_dirs, mode='valid', transform=None, debug=debug
        )
    except Exception as e:
        print(f"创建验证数据集时出错: {e}")
        val_dirs = date_dirs[n_train:n_train+min(n_val, 10)]
        val_dataset = ChongqingRadarDataset(val_dirs, mode='valid', debug=debug)
    
    if len(val_dataset) == 0:
        print("错误: 验证数据集为空！")
        return None
    
    print(f"验证集包含 {len(val_dataset)} 个样本")
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False
    )
    
    return val_loader


def load_chongqing_test(batch_size=None, num_workers=None, debug=False):
    """加载测试集"""
    if batch_size is None:
        batch_size = cfg.batch
    if num_workers is None:
        num_workers = cfg.dataloader_thread
    
    data_root = "/data_8t/WSG/data/chongqing/TDMOSAIC"
    
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        return None
    
    print(f"\n加载重庆雷达测试数据...")
    
    # 收集所有日期目录
    date_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            date_dirs.append(item_path)
    
    if not date_dirs:
        print(f"警告: 未找到日期目录")
        return None
    
    date_dirs.sort()
    
    # 数据划分：前70%训练，中间15%验证，后15%测试
    n_total = len(date_dirs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    test_dirs = date_dirs[n_train+n_val:]
    
    print(f"测试数据: {len(test_dirs)} 天 ({len(test_dirs)/n_total*100:.1f}%)")
    for i, dir_path in enumerate(test_dirs[:3]):
        print(f"  {i+1}. {os.path.basename(dir_path)}")
    if len(test_dirs) > 3:
        print(f"  ... 和 {len(test_dirs)-3} 个更多")
    
    # 创建测试数据集
    try:
        test_dataset = ChongqingRadarDataset(
            test_dirs, mode='test', transform=None, debug=debug
        )
    except Exception as e:
        print(f"创建测试数据集时出错: {e}")
        test_dirs = date_dirs[-min(30, len(date_dirs)//5):]  # 最后的部分
        test_dataset = ChongqingRadarDataset(test_dirs, mode='test', debug=debug)
    
    if len(test_dataset) == 0:
        print("错误: 测试数据集为空！")
        return None
    
    print(f"测试集包含 {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False
    )
    
    return test_loader


def load_chongqing_all(batch_size=None, num_workers=None, debug=False):
    """一次性加载训练、验证、测试集"""
    train_loader = load_chongqing_train(batch_size, num_workers, debug)
    val_loader = load_chongqing_valid(batch_size, num_workers, debug)
    test_loader = load_chongqing_test(batch_size, num_workers, debug)
    
    return train_loader, val_loader, test_loader


# ==================== 数据统计函数 ====================
def analyze_dataset(data_dirs, mode='train'):
    """分析数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"分析数据集: {mode}")
    print(f"{'='*60}")
    
    dataset = ChongqingRadarDataset(data_dirs, mode=mode, debug=True)
    
    if len(dataset) == 0:
        print("数据集为空！")
        return
    
    # 采样分析
    sample_size = min(100, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    all_inputs = []
    all_targets = []
    
    for idx in indices:
        inputs, targets = dataset[idx]
        all_inputs.append(inputs.numpy())
        all_targets.append(targets.numpy())
    
    all_inputs = np.concatenate(all_inputs, axis=0)  # [N*T_in, 1, H, W]
    all_targets = np.concatenate(all_targets, axis=0)  # [N*T_out, 1, H, W]
    
    # 转换为dBZ
    inputs_dbz = all_inputs * 80.0
    targets_dbz = all_targets * 80.0
    
    print(f"\n统计信息 (基于{sample_size}个样本):")
    print(f"输入帧数: {len(all_inputs)}")
    print(f"目标帧数: {len(all_targets)}")
    
    print(f"\n输入数据 (dBZ):")
    print(f"  最小值: {inputs_dbz.min():.2f}")
    print(f"  最大值: {inputs_dbz.max():.2f}")
    print(f"  平均值: {inputs_dbz.mean():.2f}")
    print(f"  标准差: {inputs_dbz.std():.2f}")
    
    print(f"\n目标数据 (dBZ):")
    print(f"  最小值: {targets_dbz.min():.2f}")
    print(f"  最大值: {targets_dbz.max():.2f}")
    print(f"  平均值: {targets_dbz.mean():.2f}")
    print(f"  标准差: {targets_dbz.std():.2f}")
    
    # 计算不同阈值的数据比例
    thresholds = [10, 20, 30, 40]
    print(f"\n不同反射率阈值的数据比例:")
    for th in thresholds:
        input_ratio = (inputs_dbz >= th).mean() * 100
        target_ratio = (targets_dbz >= th).mean() * 100
        print(f"  {th:2d} dBZ: 输入={input_ratio:5.2f}%, 目标={target_ratio:5.2f}%")
    
    return dataset


# ==================== 测试函数 ====================
def test_dataset():
    """测试数据集加载"""
    print("测试数据集加载...")
    
    # 测试训练集
    train_loader = load_chongqing_train(batch_size=2, debug=True)
    
    if train_loader is None:
        print("训练集加载失败")
        return
    
    # 获取一个批次
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"\n批次 {i+1}:")
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        print(f"  输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  目标范围: [{targets.min():.3f}, {targets.max():.3f}]")
        
        if i >= 1:  # 只测试前2个批次
            break
    
    # 测试验证集
    val_loader = load_chongqing_valid(batch_size=2, debug=False)
    if val_loader:
        print(f"\n验证集批次数量: {len(val_loader)}")
    
    # 测试测试集
    test_loader = load_chongqing_test(batch_size=2, debug=False)
    if test_loader:
        print(f"测试集批次数量: {len(test_loader)}")


if __name__ == "__main__":
    # 测试所有功能
    test_dataset()
    
    # 分析数据集统计信息
    data_root = "/data_8t/WSG/data/chongqing/TDMOSAIC"
    if os.path.exists(data_root):
        # 获取前几个目录进行分析
        date_dirs = []
        for item in os.listdir(data_root)[:5]:  # 只分析前5天
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
                date_dirs.append(item_path)
        
        if date_dirs:
            analyze_dataset(date_dirs, mode='train')