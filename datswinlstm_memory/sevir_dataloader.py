"""
SEVIR DataLoader - 核心数据加载器
用于从 SEVIR 数据集加载 VIL (垂直积分液态水含量) 数据
"""

import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple, Union


class SEVIRDataLoader:
    """
    SEVIR 数据加载器
    
    支持:
    - 从 CATALOG.csv 读取元数据
    - 按日期范围筛选事件
    - 从 HDF5 文件加载图像序列
    - 数据预处理和归一化
    """
    
    # VIL 数据的归一化参数
    VIL_SCALE = 47.0  # dBZ to VIL scale factor
    VIL_OFFSET = 0.0
    
    def __init__(
        self,
        catalog_path: str,
        data_dir: str,
        img_type: str = 'vil',
        seq_len: int = 49,
        raw_seq_len: int = 49,
        sample_mode: str = 'sequent',
        stride: int = 12,
        batch_size: int = 1,
        shuffle: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        preprocess: bool = True,
        rescale_method: str = '01',
        verbose: bool = False
    ):
        """
        初始化 SEVIR 数据加载器
        
        Args:
            catalog_path: CATALOG.csv 文件路径
            data_dir: 数据目录路径
            img_type: 图像类型 ('vil', 'vis', 'ir069', 'ir107', 'lght')
            seq_len: 输出序列长度
            raw_seq_len: 原始序列长度 (SEVIR 默认 49)
            sample_mode: 采样模式 ('sequent' 或 'random')
            stride: 采样步长
            batch_size: 批次大小
            shuffle: 是否打乱数据
            start_date: 开始日期
            end_date: 结束日期
            preprocess: 是否进行预处理
            rescale_method: 归一化方法 ('01' 或 'minmax')
            verbose: 是否打印详细信息
        """
        self.catalog_path = catalog_path
        self.data_dir = data_dir
        self.img_type = img_type.lower()
        self.seq_len = seq_len
        self.raw_seq_len = raw_seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_date = start_date
        self.end_date = end_date
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        
        # 加载目录
        self.catalog = self._load_catalog()
        self.file_dict = self._build_file_dict()
        
        if self.verbose:
            print(f"Loaded {len(self.catalog)} events from catalog")
            print(f"Found {len(self.file_dict)} unique files")
    
    def _load_catalog(self) -> pd.DataFrame:
        """加载并过滤 CATALOG.csv"""
        catalog = pd.read_csv(self.catalog_path, low_memory=False)
        
        # 过滤图像类型
        catalog = catalog[catalog['img_type'] == self.img_type]
        
        # 解析时间
        catalog['time_utc'] = pd.to_datetime(catalog['time_utc'])
        
        # 按日期范围过滤
        if self.start_date is not None:
            catalog = catalog[catalog['time_utc'] >= self.start_date]
        if self.end_date is not None:
            catalog = catalog[catalog['time_utc'] < self.end_date]
        
        # 过滤实际存在的文件
        catalog = self._filter_existing_files(catalog)
        
        return catalog.reset_index(drop=True)
    
    def _filter_existing_files(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """过滤掉不存在的文件"""
        existing_files = set()
        
        for _, row in catalog.iterrows():
            file_path = os.path.join(self.data_dir, row['file_name'])
            # 处理相对路径中可能的 img_type 前缀
            if not os.path.exists(file_path):
                # 尝试只使用文件名部分
                file_name = os.path.basename(row['file_name'])
                year = str(row['time_utc'].year)
                alt_path = os.path.join(self.data_dir, self.img_type, year, file_name)
                if os.path.exists(alt_path):
                    existing_files.add(row['file_name'])
            else:
                existing_files.add(row['file_name'])
        
        # 重新检查，只保留存在的文件对应的行
        valid_rows = []
        for idx, row in catalog.iterrows():
            file_path = self._get_file_path(row['file_name'])
            if file_path is not None:
                valid_rows.append(idx)
        
        return catalog.loc[valid_rows]
    
    def _get_file_path(self, file_name: str) -> Optional[str]:
        """获取文件的实际路径"""
        # 尝试直接路径
        file_path = os.path.join(self.data_dir, file_name)
        if os.path.exists(file_path):
            return file_path
        
        # 尝试简化路径 (vil/2017/xxx.h5 -> vil/2017/xxx.h5)
        parts = file_name.replace('\\', '/').split('/')
        if len(parts) >= 1:
            # 尝试 data_dir/img_type/year/filename
            simple_name = parts[-1]  # 只取文件名
            
            # 从文件名提取年份
            import re
            year_match = re.search(r'(\d{4})', simple_name)
            if year_match:
                year = year_match.group(1)
                alt_path = os.path.join(self.data_dir, self.img_type, year, simple_name)
                if os.path.exists(alt_path):
                    return alt_path
        
        return None
    
    def _build_file_dict(self) -> dict:
        """构建文件索引字典"""
        file_dict = {}
        for idx, row in self.catalog.iterrows():
            file_name = row['file_name']
            if file_name not in file_dict:
                file_dict[file_name] = []
            file_dict[file_name].append({
                'catalog_idx': idx,
                'file_index': row['file_index'],
                'time_utc': row['time_utc']
            })
        return file_dict
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.catalog)
    
    def load_event(self, idx: int) -> np.ndarray:
        """
        加载单个事件的图像序列
        
        Args:
            idx: 事件索引
            
        Returns:
            numpy array of shape (T, H, W) 或 (T, C, H, W)
        """
        row = self.catalog.iloc[idx]
        file_path = self._get_file_path(row['file_name'])
        
        if file_path is None:
            raise FileNotFoundError(f"Cannot find file: {row['file_name']}")
        
        file_index = row['file_index']
        
        with h5py.File(file_path, 'r') as f:
            # SEVIR VIL 数据格式: (N, T, H, W) 其中 N 是事件数
            data_key = self.img_type
            if data_key not in f:
                # 尝试其他可能的键名
                data_key = list(f.keys())[0]
            
            data = f[data_key][file_index]  # (T, H, W)
        
        # 预处理
        if self.preprocess:
            data = self._preprocess(data)
        
        return data
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        预处理数据
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 转换为 float32
        data = data.astype(np.float32)
        
        # 归一化
        if self.rescale_method == '01':
            # 归一化到 [0, 1]
            # VIL 原始值范围大约 0-255
            data = data / 255.0
        elif self.rescale_method == 'minmax':
            # Min-Max 归一化
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def sample_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        从完整序列中采样指定长度的子序列
        
        Args:
            data: 完整序列 (T, H, W)
            
        Returns:
            采样后的序列 (seq_len, H, W)
        """
        T = data.shape[0]
        
        if self.sample_mode == 'sequent':
            # 顺序采样
            if T >= self.seq_len:
                # 随机选择起始点
                start = np.random.randint(0, T - self.seq_len + 1)
                return data[start:start + self.seq_len]
            else:
                # 序列太短，用零填充
                result = np.zeros((self.seq_len,) + data.shape[1:], dtype=data.dtype)
                result[:T] = data
                return result
        else:
            # 随机采样 (带步长)
            indices = np.arange(0, T, self.stride)[:self.seq_len]
            if len(indices) < self.seq_len:
                # 填充
                result = np.zeros((self.seq_len,) + data.shape[1:], dtype=data.dtype)
                result[:len(indices)] = data[indices]
                return result
            return data[indices[:self.seq_len]]
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """
        获取一个批次的数据
        
        Args:
            indices: 事件索引列表
            
        Returns:
            批次数据 (B, T, H, W)
        """
        batch = []
        for idx in indices:
            data = self.load_event(idx)
            data = self.sample_sequence(data)
            batch.append(data)
        
        return np.stack(batch, axis=0)
    
    def iterate_batches(self):
        """
        迭代所有批次
        
        Yields:
            批次数据
        """
        indices = list(range(len(self)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.get_batch(batch_indices)


def test_dataloader():
    """测试数据加载器"""
    catalog_path = r"X:\datasets\sevir\CATALOG.csv"
    data_dir = r"X:\datasets\sevir\data"
    
    loader = SEVIRDataLoader(
        catalog_path=catalog_path,
        data_dir=data_dir,
        img_type='vil',
        seq_len=24,
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2017, 7, 1),
        verbose=True
    )
    
    print(f"Dataset size: {len(loader)}")
    
    if len(loader) > 0:
        # 加载一个样本
        sample = loader.load_event(0)
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")


if __name__ == '__main__':
    test_dataloader()
