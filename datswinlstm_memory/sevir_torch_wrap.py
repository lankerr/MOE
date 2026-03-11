"""
SEVIR PyTorch Dataset 封装
提供与 PyTorch DataLoader 兼容的接口
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import Optional, Tuple, Union
import h5py
import pandas as pd


class SEVIRTorchDataset(Dataset):
    """
    SEVIR PyTorch Dataset
    
    用于 PyTorch 训练的 SEVIR 数据集封装
    """
    
    def __init__(
        self,
        sevir_catalog: str,
        sevir_data_dir: str,
        raw_seq_len: int = 49,
        split_mode: str = "uneven",
        shuffle: bool = False,
        seq_len: int = 24,
        stride: int = 12,
        sample_mode: str = "sequent",
        batch_size: int = 1,
        layout: str = "NTCHW",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_type: np.dtype = np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        verbose: bool = False,
        **kwargs
    ):
        """
        初始化 SEVIR PyTorch Dataset
        
        Args:
            sevir_catalog: CATALOG.csv 文件路径
            sevir_data_dir: 数据目录路径
            raw_seq_len: 原始序列长度
            split_mode: 分割模式
            shuffle: 是否打乱
            seq_len: 输出序列长度
            stride: 采样步长
            sample_mode: 采样模式
            batch_size: 批次大小 (用于兼容性)
            layout: 数据布局 ('NTCHW' 或 'NTHWC')
            start_date: 开始日期
            end_date: 结束日期
            output_type: 输出数据类型
            preprocess: 是否预处理
            rescale_method: 归一化方法
            verbose: 是否打印信息
        """
        super().__init__()
        
        self.catalog_path = sevir_catalog
        self.data_dir = sevir_data_dir
        self.raw_seq_len = raw_seq_len
        self.seq_len = seq_len
        self.stride = stride
        self.sample_mode = sample_mode
        self.batch_size = batch_size
        self.layout = layout
        self.start_date = start_date
        self.end_date = end_date
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.shuffle = shuffle
        
        # 加载目录
        self.catalog = self._load_catalog()
        self.samples = self._build_sample_list()
        
        if self.verbose:
            print(f"SEVIRTorchDataset: {len(self.samples)} samples loaded")
    
    def _load_catalog(self) -> pd.DataFrame:
        """加载并过滤 CATALOG"""
        catalog = pd.read_csv(self.catalog_path, low_memory=False)
        
        # 只保留 VIL 数据
        catalog = catalog[catalog['img_type'] == 'vil']
        
        # 解析时间
        catalog['time_utc'] = pd.to_datetime(catalog['time_utc'])
        
        # 按日期过滤
        if self.start_date is not None:
            catalog = catalog[catalog['time_utc'] >= self.start_date]
        if self.end_date is not None:
            catalog = catalog[catalog['time_utc'] < self.end_date]
        
        return catalog.reset_index(drop=True)
    
    def _get_file_path(self, file_name: str) -> Optional[str]:
        """获取文件实际路径"""
        import re
        
        # 直接路径
        file_path = os.path.join(self.data_dir, file_name)
        if os.path.exists(file_path):
            return file_path
        
        # 提取文件名和年份
        parts = file_name.replace('\\', '/').split('/')
        simple_name = parts[-1]
        
        year_match = re.search(r'(\d{4})', simple_name)
        if year_match:
            year = year_match.group(1)
            alt_path = os.path.join(self.data_dir, 'vil', year, simple_name)
            if os.path.exists(alt_path):
                return alt_path
        
        return None
    
    def _build_sample_list(self) -> list:
        """构建样本列表"""
        samples = []
        
        for idx, row in self.catalog.iterrows():
            file_path = self._get_file_path(row['file_name'])
            if file_path is not None:
                samples.append({
                    'file_path': file_path,
                    'file_index': row['file_index'],
                    'time_utc': row['time_utc']
                })
        
        if self.shuffle:
            np.random.shuffle(samples)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            torch.Tensor of shape depending on layout
        """
        sample = self.samples[idx]
        
        # 从 HDF5 加载数据
        with h5py.File(sample['file_path'], 'r') as f:
            # 获取数据键
            data_key = 'vil' if 'vil' in f else list(f.keys())[0]
            data = f[data_key][sample['file_index']]  # (H, W, T)
        
        # 转换类型
        data = data.astype(self.output_type)
        
        # 转置轴: (H, W, T) -> (T, H, W)
        data = np.transpose(data, (2, 0, 1))
        
        # 预处理
        if self.preprocess:
            data = self._preprocess(data)
        
        # 采样序列
        data = self._sample_sequence(data)
        
        # 调整布局
        if self.layout == 'NTCHW':
            # (T, H, W) -> (T, 1, H, W)
            data = np.expand_dims(data, axis=1)
        elif self.layout == 'NTHWC':
            # (T, H, W) -> (T, H, W, 1)
            data = np.expand_dims(data, axis=-1)
        
        return torch.from_numpy(data)
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """预处理数据"""
        if self.rescale_method == '01':
            # VIL 值归一化到 [0, 1]
            data = data / 255.0
        elif self.rescale_method == 'minmax':
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def _sample_sequence(self, data: np.ndarray) -> np.ndarray:
        """采样序列"""
        T = data.shape[0]
        
        if self.sample_mode == 'sequent':
            if T >= self.seq_len:
                start = np.random.randint(0, T - self.seq_len + 1)
                return data[start:start + self.seq_len]
            else:
                result = np.zeros((self.seq_len,) + data.shape[1:], dtype=data.dtype)
                result[:T] = data
                return result
        else:
            indices = np.arange(0, min(T, self.seq_len * self.stride), self.stride)
            indices = indices[:self.seq_len]
            if len(indices) < self.seq_len:
                result = np.zeros((self.seq_len,) + data.shape[1:], dtype=data.dtype)
                result[:len(indices)] = data[indices]
                return result
            return data[indices]
    
    def get_torch_dataloader(
        self,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        获取 PyTorch DataLoader
        
        Args:
            num_workers: 工作进程数
            pin_memory: 是否锁定内存
            
        Returns:
            DataLoader 实例
        """
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )


def test_dataset():
    """测试数据集"""
    catalog_path = r"c:\Users\97290\Desktop\datasets\sevir\CATALOG.csv"
    data_dir = r"c:\Users\97290\Desktop\datasets\sevir\data"
    
    dataset = SEVIRTorchDataset(
        sevir_catalog=catalog_path,
        sevir_data_dir=data_dir,
        seq_len=36,
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2017, 7, 1),
        verbose=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")


if __name__ == '__main__':
    test_dataset()
