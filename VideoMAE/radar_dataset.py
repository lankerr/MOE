"""
SEVIR Radar Dataset for VideoMAE Pretraining
=============================================
从 SEVIR HDF5 文件中读取 VIL 雷达数据，
输出 VideoMAE 所需格式: (C, T, H, W) + mask

数据路径: X:\datasets\sevir
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
from typing import Optional
import torch.nn.functional as F


class SEVIRVideoMAEDataset(Dataset):
    """
    SEVIR VIL → VideoMAE 预训练数据集
    
    从 HDF5 中读取 (384, 384, 49) 的 VIL 事件,
    随机抽取 num_frames 帧, resize 到 input_size,
    输出 (C=1, T, H, W) 的张量 + tube mask
    """
    
    def __init__(
        self,
        sevir_root: str = r"X:\datasets\sevir",
        num_frames: int = 8,
        input_size: int = 128,
        mask_ratio: float = 0.9,
        tubelet_size: int = 2,
        patch_size: int = 16,
        split: str = "train",  # train / val / test
        max_samples: Optional[int] = None,
        augment: bool = True,  # 数据增强
    ):
        super().__init__()
        self.sevir_root = sevir_root
        self.num_frames = num_frames
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        
        # 加载 catalog
        catalog_path = os.path.join(sevir_root, "CATALOG.csv")
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"找不到 CATALOG.csv: {catalog_path}")
        
        catalog = pd.read_csv(catalog_path, low_memory=False)
        catalog = catalog[catalog['img_type'] == 'vil'].copy()
        catalog['time_utc'] = pd.to_datetime(catalog['time_utc'])
        
        # SEVIR 标准划分
        if split == "train":
            catalog = catalog[catalog['time_utc'] < '2019-06-01']
        elif split == "val":
            catalog = catalog[
                (catalog['time_utc'] >= '2019-06-01') &
                (catalog['time_utc'] < '2019-09-01')
            ]
        elif split == "test":
            catalog = catalog[catalog['time_utc'] >= '2019-09-01']
        
        # 构建样本列表
        self.samples = []
        for _, row in catalog.iterrows():
            fpath = self._resolve_path(row['file_name'])
            if fpath is not None:
                self.samples.append({
                    'file_path': fpath,
                    'file_index': int(row['file_index']),
                })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        # 掩码参数
        self.num_patches_per_frame = (input_size // patch_size) ** 2
        self.num_temporal_tokens = num_frames // tubelet_size
        self.total_tokens = self.num_patches_per_frame * self.num_temporal_tokens
        self.num_mask = int(self.total_tokens * mask_ratio)
        
        print(f"[SEVIRVideoMAE] split={split}, samples={len(self.samples)}, "
              f"frames={num_frames}, size={input_size}, "
              f"tokens={self.total_tokens}, mask={self.num_mask}/{self.total_tokens}")
    
    def _resolve_path(self, file_name: str) -> Optional[str]:
        """解析 HDF5 文件路径"""
        import re
        # 直接路径
        fpath = os.path.join(self.sevir_root, file_name)
        if os.path.exists(fpath):
            return fpath
        # data/vil/YYYY/xxx.h5
        parts = file_name.replace('\\', '/').split('/')
        simple = parts[-1]
        m = re.search(r'(\d{4})', simple)
        if m:
            alt = os.path.join(self.sevir_root, 'data', 'vil', m.group(1), simple)
            if os.path.exists(alt):
                return alt
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            frames: (C=1, T, H, W) float32, [0, 1]
            mask: (total_tokens,) bool — True=masked
        """
        sample = self.samples[idx]
        
        # 读取 HDF5
        with h5py.File(sample['file_path'], 'r') as f:
            key = 'vil' if 'vil' in f else list(f.keys())[0]
            raw = f[key][sample['file_index']]  # (384, 384, 49) uint8
        
        raw = raw.astype(np.float32) / 255.0  # [0, 1]
        
        # (H, W, T) → (T, H, W)
        raw = np.transpose(raw, (2, 0, 1))  # (49, 384, 384)
        T_total = raw.shape[0]
        
        # 随机抽取连续 num_frames 帧
        if T_total >= self.num_frames:
            start = np.random.randint(0, T_total - self.num_frames + 1)
            frames = raw[start:start + self.num_frames]
        else:
            frames = np.zeros((self.num_frames, 384, 384), dtype=np.float32)
            frames[:T_total] = raw
        
        # ★ 数据增强 (防止 8000 epochs 过拟合)
        if self.augment:
            # 随机水平翻转
            if np.random.rand() > 0.5:
                frames = frames[:, :, ::-1].copy()
            # 随机垂直翻转
            if np.random.rand() > 0.5:
                frames = frames[:, ::-1, :].copy()
            # 随机时间翻转 (反转时间序列)
            if np.random.rand() > 0.3:
                frames = frames[::-1].copy()
            # 随机裁剪 (从 384 中随机裁一个 crop_size 区域, 再 resize)
            crop_size = np.random.randint(int(384 * 0.7), 384)
            y0 = np.random.randint(0, 384 - crop_size + 1)
            x0 = np.random.randint(0, 384 - crop_size + 1)
            frames = frames[:, y0:y0+crop_size, x0:x0+crop_size]
        
        # (T, H, W) → (1, T, H, W) 单通道
        frames = torch.from_numpy(frames).unsqueeze(0)  # (1, T, crop_H, crop_W)
        
        # Resize 到 input_size
        cur_h, cur_w = frames.shape[2], frames.shape[3]
        if cur_h != self.input_size or cur_w != self.input_size:
            frames = F.interpolate(
                frames, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False
            )  # (1, T, input_size, input_size)
        
        # 生成 Tube Mask (时间方向一致的随机掩码)
        mask = self._generate_tube_mask()
        
        return frames, mask
    
    def _generate_tube_mask(self):
        """
        Tube Masking: 时间方向的 token 共享同一个 mask
        返回: (total_tokens,) 的 bool 数组, True=被遮挡
        """
        # 空间 token 数
        num_spatial = self.num_patches_per_frame
        # 随机选 mask_ratio 比例的空间位置
        num_spatial_mask = int(num_spatial * self.mask_ratio)
        
        spatial_mask = np.zeros(num_spatial, dtype=bool)
        mask_indices = np.random.choice(num_spatial, num_spatial_mask, replace=False)
        spatial_mask[mask_indices] = True
        
        # Tube: 时间方向复制
        mask = np.tile(spatial_mask, self.num_temporal_tokens)
        return mask


def test_dataset():
    """快速测试"""
    print("=" * 50)
    print("测试 SEVIRVideoMAEDataset")
    print("=" * 50)
    
    # 检查路径
    sevir_root = r"X:\datasets\sevir"
    if not os.path.exists(sevir_root):
        print(f"[跳过] SEVIR 数据不在: {sevir_root}")
        print("生成假数据进行测试...")
        return test_with_dummy_data()
    
    ds = SEVIRVideoMAEDataset(
        sevir_root=sevir_root,
        num_frames=8,
        input_size=128,
        mask_ratio=0.9,
        split="train",
        max_samples=10,
    )
    
    frames, mask = ds[0]
    print(f"frames shape: {frames.shape}")   # (1, 8, 128, 128)
    print(f"frames range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"mask shape:   {mask.shape}")      # (256,) for 8/2 * 8*8 = 4*64
    print(f"mask ratio:   {mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.2f}")
    print("数据集OK!")
    return True


def test_with_dummy_data():
    """用假数据测试 (无需 SEVIR)"""
    print("使用随机数据测试 pipeline...")
    
    num_frames = 8
    input_size = 128
    patch_size = 16
    tubelet_size = 2
    mask_ratio = 0.9
    
    # 模拟数据 (1, T, H, W)
    frames = torch.rand(1, num_frames, input_size, input_size)
    
    # 模拟 mask
    num_spatial = (input_size // patch_size) ** 2  # 64
    num_temporal = num_frames // tubelet_size       # 4
    total = num_spatial * num_temporal               # 256
    num_mask = int(total * mask_ratio)               # 230
    
    mask = np.zeros(total, dtype=bool)
    mask[np.random.choice(total, num_mask, replace=False)] = True
    
    print(f"frames: {frames.shape} — (C=1, T={num_frames}, H={input_size}, W={input_size})")
    print(f"mask:   {mask.shape} — {mask.sum()}/{total} masked ({mask_ratio*100:.0f}%)")
    print(f"visible tokens: {total - mask.sum()} (送入 Encoder)")
    
    # 估算显存
    embed_dim = 192  # ViT-Tiny
    visible = total - mask.sum()
    # 每个 token: embed_dim × 4 bytes (fp32) 或 ×2 (fp16)
    attn_matrix_bytes = visible * visible * 2  # fp16
    print(f"\nViT-Tiny (embed_dim={embed_dim}):")
    print(f"  Attention matrix: {visible}×{visible} = {attn_matrix_bytes/1024:.1f} KB (fp16)")
    print(f"  预估训练显存: ~3-4 GB — RTX 5070 (8.55GB) 可行 ✅")
    print(f"  预估推理显存: ~1-2 GB — RTX 3050Ti (4GB) 可行 ✅")
    return True


if __name__ == "__main__":
    test_dataset()
