"""
重庆雷达数据 DataModule for Baseline Experiments
==================================================

适配重庆雷达数据到 EarthFormer baseline 实验。

数据格式:
- 预处理后: day_simple_YYYYMMDD.npy → (240, 384, 384)
- 每天最多240帧 (每6分钟一帧, 24小时)
- 与SEVIR相同的VIL归一化: [0, 1]

使用方法:
    # 在配置文件中设置:
    dataset:
      dataset_name: "chongqing"
      data_dir: "/path/to/chongqing/data"
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, List


class ChongqingRadarDataset(Dataset):
    """
    重庆雷达数据集

    支持多种时间窗口配置:
    - 2h→2h: 24帧输入 → 24帧输出 (与SEVIR一致)
    - 4h→4h: 48帧输入 → 48帧输出
    - 2h→3h: 24帧输入 → 36帧输出
    - 4h→3h: 48帧输入 → 36帧输出

    注意: 重庆数据每6分钟一帧, 而SEVIR每5分钟一帧
    因此帧数与时间对应关系不同:
    - SEVIR: 24帧 = 2小时 (5分钟间隔)
    - 重庆: 20帧 = 2小时 (6分钟间隔)
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        in_len: int = 24,
        out_len: int = 24,
        img_size: Tuple[int, int] = (384, 384),
        stride: int = 24,
        frame_interval_minutes: int = 6,  # 重庆数据6分钟间隔
    ):
        """
        Parameters
        ----------
        data_dir : str
            数据目录,包含 day_simple_YYYYMMDD.npy 文件
        mode : str
            'train', 'val', 或 'test'
        in_len : int
            输入序列长度
        out_len : int
            输出序列长度
        img_size : tuple
            图像尺寸 (H, W)
        stride : int
            滑动窗口步长
        frame_interval_minutes : int
            帧间隔分钟数 (重庆=6, SEVIR=5)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len
        self.img_size = img_size
        self.stride = stride
        self.frame_interval = frame_interval_minutes

        # 收集所有数据文件
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "day_simple_*.npy")))

        if not self.data_files:
            raise ValueError(f"[Chongqing] 未找到数据文件 in {data_dir}")

        print(f"[Chongqing] 找到 {len(self.data_files)} 个数据文件")

        # 划分数据集 (按日期)
        train_ratio = 0.7
        val_ratio = 0.15

        n_train = int(len(self.data_files) * train_ratio)
        n_val = int(len(self.data_files) * val_ratio)

        if mode == 'train':
            self.data_files = self.data_files[:n_train]
        elif mode == 'val':
            self.data_files = self.data_files[n_train:n_train + n_val]
        else:  # test
            self.data_files = self.data_files[n_train + n_val:]

        print(f"[Chongqing] {mode}集: {len(self.data_files)} 个文件")

        # 创建样本索引
        self.samples = self._create_samples()
        print(f"[Chongqing] {mode}集: {len(self.samples)} 个样本")

    def _create_samples(self) -> List[Tuple[int, int]]:
        """创建样本索引 (file_idx, start_frame)"""
        samples = []

        for file_idx, file_path in enumerate(self.data_files):
            # 加载数据获取形状
            try:
                data = np.load(file_path, mmap_mode='r')
                total_frames = data.shape[0]

                # 滑动窗口
                for start_idx in range(0, total_frames - self.seq_len + 1, self.stride):
                    if start_idx + self.seq_len <= total_frames:
                        samples.append((file_idx, start_idx))

            except Exception as e:
                print(f"[Chongqing] 警告: 无法读取 {file_path}: {e}")
                continue

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_idx = self.samples[idx]
        file_path = self.data_files[file_idx]

        # 加载数据
        data = np.load(file_path, mmap_mode='r')

        # 提取序列
        seq = data[start_idx:start_idx + self.seq_len]

        # 确保形状正确
        if seq.ndim == 2:
            seq = seq[..., np.newaxis]
        elif seq.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected shape: {seq.shape}")

        # 调整尺寸 (如果需要)
        if seq.shape[1:3] != self.img_size:
            # 使用简单的resize (可以后续优化为插值)
            from skimage.transform import resize
            resized = np.zeros((seq.shape[0], *self.img_size, seq.shape[3]), dtype=seq.dtype)
            for i in range(seq.shape[0]):
                for c in range(seq.shape[3]):
                    resized[i, :, :, c] = resize(seq[i, :, :, c], self.img_size,
                                                 preserve_range=True, order=1)
            seq = resized

        # 划分输入和输出
        x = seq[:self.in_len]  # (T_in, H, W, C)
        y = seq[self.in_len:self.seq_len]  # (T_out, H, W, C)

        # 转换为 Tensor: (H, W, T, C) for EarthFormer
        # EarthFormer 期望 NTHWC 格式, 但DataLoader返回的是单样本
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y


class ChongqingDataModule(pl.LightningDataModule):
    """
    重庆雷达数据 Lightning DataModule
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,
        in_len: int = 24,
        out_len: int = 24,
        img_size: Tuple[int, int] = (384, 384),
        stride: int = 24,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_len = in_len
        self.out_len = out_len
        self.img_size = img_size
        self.stride = stride
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ChongqingRadarDataset(
                self.data_dir, mode='train',
                in_len=self.in_len, out_len=self.out_len,
                img_size=self.img_size, stride=self.stride,
            )
            self.val_dataset = ChongqingRadarDataset(
                self.data_dir, mode='val',
                in_len=self.in_len, out_len=self.out_len,
                img_size=self.img_size, stride=self.stride,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ChongqingRadarDataset(
                self.data_dir, mode='test',
                in_len=self.in_len, out_len=self.out_len,
                img_size=self.img_size, stride=self.stride,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def get_datamodule(config, dataset_type: str = 'sevir'):
    """
    根据配置获取 DataModule

    Parameters
    ----------
    config : dict
        配置字典
    dataset_type : str
        'sevir' 或 'chongqing'

    Returns
    -------
    datamodule : pl.LightningDataModule
    """
    dataset_config = config.get('dataset', {})
    optim_config = config.get('optim', {})

    if dataset_type == 'chongqing':
        return ChongqingDataModule(
            data_dir=dataset_config.get('data_dir', './data/chongqing'),
            batch_size=optim_config.get('micro_batch_size', 1),
            num_workers=4,
            in_len=dataset_config.get('in_len', 24),
            out_len=dataset_config.get('out_len', 24),
            img_size=(dataset_config.get('img_height', 384),
                     dataset_config.get('img_width', 384)),
            stride=dataset_config.get('stride', 24),
        )
    else:
        # 使用 SEVIR 原始 DataModule
        from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
        return SEVIRLightningDataModule(
            data_dir=dataset_config.get('data_dir', './data/sevir'),
            batch_size=optim_config.get('micro_batch_size', 1),
            **dataset_config,
        )
