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
    
    def __init__(self, data_dirs, mode='test', seq_len=25, target_height_idx=6):
        """
        参数:
            data_dirs: 数据目录列表
            mode: 'train', 'valid', 'test'
            seq_len: 序列总长度 (13输入 + 12输出 = 25)
            target_height_idx: 目标高度层索引，默认6对应3.5km
        """
        self.mode = mode
        self.seq_len = seq_len
        self.input_len = cfg.in_len  # 13
        self.output_len = cfg.out_len  # 12
        self.img_size = (cfg.width, cfg.height)
        self.target_height_idx = target_height_idx
        
        print(f"初始化数据集: 模式={mode}, 序列长度={seq_len}")
        print(f"输入长度={self.input_len}, 输出长度={self.output_len}")
        print(f"图像尺寸={self.img_size}, 高度层索引={target_height_idx}")
        
        # 收集所有雷达数据文件
        self.data_files = []
        for data_dir in data_dirs:
            pattern = os.path.join(data_dir, "**", "*.bin.DATA_DBZ.bz2")
            files = glob.glob(pattern, recursive=True)
            files.sort(key=lambda x: self._extract_time_from_filename(x))
            self.data_files.extend(files)
        
        print(f"找到 {len(self.data_files)} 个雷达数据文件")
        
        if not self.data_files:
            print("警告: 没有找到任何数据文件！")
            print("请检查数据路径:", data_dirs)
            print("文件模式:", os.path.join(data_dirs[0] if data_dirs else "", "**", "*.bin.DATA_DBZ.bz2"))
        
        # 创建序列索引
        self.sequences = self._create_sequences()
        print(f"创建了 {len(self.sequences)} 个序列样本")
        
        # 显示使用的数据集信息
        if self.sequences:
            sample_file = self.sequences[0][0]
            print(f"样本文件示例: {os.path.basename(sample_file)}")
            try:
                height_layers = cfg.Chongqing.HEIGHT_LAYERS
                print(f"使用高度层: {height_layers[self.target_height_idx]}km")
            except:
                default_height_layers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 
                                        8.0, 9.0, 10.0, 12.0, 14.0, 15.5, 17.0, 19.0]
                print(f"使用高度层: {default_height_layers[self.target_height_idx]}km")
    
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
            
            return datetime.datetime(2000, 1, 1)
        except Exception as e:
            print(f"解析文件名 {basename} 时出错: {e}")
            return datetime.datetime(2000, 1, 1)
    
    def _create_sequences(self):
        """创建连续的序列索引"""
        sequences = []
        
        if not self.data_files:
            return sequences
        
        # 按时间排序
        sorted_files = sorted(self.data_files, key=lambda x: self._extract_time_from_filename(x))
        
        current_sequence = []
        for i, file_path in enumerate(sorted_files):
            if not current_sequence:
                current_sequence.append(file_path)
                continue
            
            # 检查时间连续性（6分钟间隔）
            prev_time = self._extract_time_from_filename(current_sequence[-1])
            curr_time = self._extract_time_from_filename(file_path)
            time_diff = (curr_time - prev_time).total_seconds() / 60  # 分钟
            
            if abs(time_diff - 6) < 1:
                current_sequence.append(file_path)
            else:
                if len(current_sequence) >= self.seq_len:
                    sequences.extend(self._split_sequence(current_sequence))
                current_sequence = [file_path]
        
        # 处理最后一个序列
        if len(current_sequence) >= self.seq_len:
            sequences.extend(self._split_sequence(current_sequence))
        
        if sequences:
            print(f"第一个序列的时间范围:")
            first_seq = sequences[0]
            start_time = self._extract_time_from_filename(first_seq[0])
            end_time = self._extract_time_from_filename(first_seq[-1])
            print(f"  开始: {start_time}")
            print(f"  结束: {end_time}")
            print(f"  持续时间: {(end_time - start_time).total_seconds()/60:.1f}分钟")
        
        return sequences
    
    def _split_sequence(self, file_list):
        """将长序列分割为固定长度的子序列"""
        sequences = []
        for i in range(0, len(file_list) - self.seq_len + 1, self.seq_len):
            sequences.append(file_list[i:i + self.seq_len])
        
        if len(sequences) < 10 and len(file_list) >= self.seq_len:
            print("数据量较少，使用滑动窗口生成序列...")
            sequences = []
            step = max(1, self.seq_len // 2)
            for i in range(0, len(file_list) - self.seq_len + 1, step):
                sequences.append(file_list[i:i + self.seq_len])
        
        return sequences
    
    def _create_empty_data(self):
        """创建空数据"""
        return np.zeros(self.img_size, dtype=np.float32)
    
    def _read_radar_data(self, file_path):
        """读取单个雷达数据文件并预处理 - 修复版"""
        try:
            # 使用cinrad读取SWAN数据
            f = cinrad.io.SWAN(file_path, product="CR")
            
            # 检查数据有效性
            if f.data is None or len(f.data) == 0:
                return self._create_empty_data()
            
            # 确保有足够的高度层
            if len(f.data) <= self.target_height_idx:
                return self._create_empty_data()
            
            # 提取指定高度层数据
            height_data = f.data[self.target_height_idx]
            
            # 处理NaN值和无效数据 - 关键修复！
            height_data = np.nan_to_num(height_data, nan=-999.0)
            
            # 移除填充值（负值通常表示无数据）
            # 雷达反射率有效范围通常是 0-80 dBZ
            mask_valid = height_data >= 0
            height_data[~mask_valid] = 0.0
            
            # 限制在合理范围
            height_data = np.clip(height_data, 0, 80.0)
            
            # 调整尺寸
            height_data = cv2.resize(height_data, self.img_size, 
                                    interpolation=cv2.INTER_LINEAR)
            
            # 归一化到[0,1] - 保持和之前一致
            height_data = height_data / 80.0
            
            # 确保在[0,1]范围内
            height_data = np.clip(height_data, 0, 1)
            
            return height_data.astype(np.float32)
            
        except Exception as e:
            print(f"读取文件 {os.path.basename(file_path)} 时出错: {e}")
            return self._create_empty_data()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取一个序列样本"""
        file_sequence = self.sequences[idx]
        
        # 读取并预处理所有帧
        frames = []
        for file_path in file_sequence:
            frame_data = self._read_radar_data(file_path)
            frames.append(frame_data)
        
        # 转换为numpy数组
        sequence_array = np.stack(frames)  # [seq_len, H, W]
        
        # 转换为tensor
        sequence_tensor = torch.from_numpy(sequence_array).float()
        
        # 分割输入和目标
        input_seq = sequence_tensor[:self.input_len]  # 前13帧
        target_seq = sequence_tensor[self.input_len:self.input_len+self.output_len]  # 后12帧
        
        # 添加通道维度 [seq_len, 1, H, W]
        input_seq = input_seq.unsqueeze(1)
        target_seq = target_seq.unsqueeze(1)
        
        return input_seq, target_seq


def load_chongqing_test(batch_size=None, num_workers=None):
    """仅加载测试集，用于测试脚本"""
    
    if batch_size is None:
        batch_size = cfg.batch
    if num_workers is None:
        num_workers = cfg.dataloader_thread
    
    # 数据根目录
    data_root = "/data_8t/WSG/data/chongqing/TDMOSAIC"
    
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        return None
    
    print(f"加载重庆雷达测试数据...")
    print(f"数据路径: {data_root}")
    
    # 收集测试数据
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
    
    # 取后20%作为测试
    total_days = len(date_dirs)
    test_start = max(0, total_days - 30)
    test_dirs = date_dirs[test_start:]
    
    print(f"使用 {len(test_dirs)} 天的测试数据:")
    for i, dir_path in enumerate(test_dirs[:5]):
        print(f"  {i+1}. {os.path.basename(dir_path)}")
    if len(test_dirs) > 5:
        print(f"  ... 和 {len(test_dirs)-5} 个更多")
    
    # 创建测试数据集
    try:
        test_dataset = ChongqingRadarDataset(test_dirs, mode='test')
    except Exception as e:
        print(f"创建测试数据集时出错: {e}")
        print("尝试使用所有数据...")
        test_dataset = ChongqingRadarDataset(date_dirs, mode='test')
    
    if len(test_dataset) == 0:
        print("错误: 测试数据集为空！")
        return None
    
    print(f"测试集包含 {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers,
                           pin_memory=True, drop_last=False)    
    return test_loader


# 测试函数
def test_dataset():
    """测试数据集加载"""
    print("测试数据集加载...")
    test_loader = load_chongqing_test(batch_size=2)
    
    if test_loader is None:
        print("数据集加载失败")
        return
    
    # 获取一个批次
    for i, (inputs, targets) in enumerate(test_loader):
        print(f"批次 {i+1}:")
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        print(f"  输入范围: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  目标范围: [{targets.min():.3f}, {targets.max():.3f}]")
        
        if i >= 0:
            break

if __name__ == "__main__":
    test_dataset()