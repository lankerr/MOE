import sys
import os
from torch.utils.data.distributed import DistributedSampler
from matplotlib import colors
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
sys.path.append('/data_8t/WSG/code/MS-RNN-main/util')
from sevir_config import datacfg
import datetime
from sevir_torch_wrap_t import SEVIRTorchDataset
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
import rasterio


class SEVIRLightningDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = kwargs.get('seq_len', 25)
        self.sample_mode = kwargs.get('sample_mode', 'sequent')
        self.stride = kwargs.get('stride', 12)
        self.batch_size = kwargs.get('batch_size', 1)
        self.layout = kwargs.get('layout', 'NTCHW')
        self.output_type = kwargs.get('output_type', np.float32)
        self.preprocess = kwargs.get('preprocess', True)
        self.rescale_method = kwargs.get('rescale_method', '01')
        self.verbose = kwargs.get('verbose', False)
        self.num_workers = kwargs.get('num_workers', 0)
        self.use_distributed = kwargs.get('use_distributed', torch.cuda.device_count() > 1)

        dataset_name = kwargs.get('dataset_name', 'sevir')
        self.setup_dataset(dataset_name)

        self.start_date = datetime.datetime(*kwargs.get('start_date')) if kwargs.get('start_date') else None
        self.train_val_split_date = datetime.datetime(*kwargs.get('train_val_split_date', (2019, 1, 1)))
        self.train_test_split_date = datetime.datetime(*kwargs.get('train_test_split_date', (2019, 6, 30)))
        self.end_date = datetime.datetime(*kwargs.get('end_date')) if kwargs.get('end_date') else None

    def setup_dataset(self, dataset_name):
        if dataset_name == "sevir":
            sevir_root_dir = os.path.join(datacfg.datasets_dir, "sevir")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevir_lr":
            sevir_root_dir = os.path.join(datacfg.datasets_dir, "sevir_lr")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevir_lr'.")
        self.dataset_name = dataset_name
        self.sevir_root_dir = sevir_root_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_root_dir):
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            print('no data')

    def setup(self, stage=None) -> None:
        common_args = dict(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,
        )

        self.sevir_train = SEVIRTorchDataset(shuffle=True, start_date=self.start_date,
                                             end_date=self.train_val_split_date, **common_args)
        self.sevir_val = SEVIRTorchDataset(shuffle=False, start_date=self.train_val_split_date,
                                           end_date=self.train_test_split_date, **common_args)
        self.sevir_test = SEVIRTorchDataset(shuffle=False, start_date=self.train_test_split_date,
                                            end_date=self.end_date, **common_args)
        self.sevir_predict = SEVIRTorchDataset(shuffle=False, start_date=self.train_test_split_date,
                                               end_date=self.end_date, **common_args)

        print(f'Train set size: {len(self.sevir_train)}')
        print(f'Validation set size: {len(self.sevir_val)}')
        print(f'Test set size: {len(self.sevir_test)}')

    def train_dataloader(self):
        if self.use_distributed:
            sampler = DistributedSampler(self.sevir_train)
            return sampler, DataLoader(self.sevir_train, batch_size=self.batch_size, sampler=sampler,
                              num_workers=self.num_workers)
        else:
            return DataLoader(self.sevir_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.use_distributed:
            sampler = DistributedSampler(self.sevir_val)
            return DataLoader(self.sevir_val, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(self.sevir_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.use_distributed:
            sampler = DistributedSampler(self.sevir_test)
            return DataLoader(self.sevir_test, batch_size=self.batch_size, sampler=sampler,
                              num_workers=self.num_workers)
        else:
            return DataLoader(self.sevir_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        if self.use_distributed:
            sampler = DistributedSampler(self.sevir_predict)
            return DataLoader(self.sevir_predict, batch_size=self.batch_size, sampler=sampler,
                              num_workers=self.num_workers)
        else:
            return DataLoader(self.sevir_predict, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)


def get_train_images(dm, num):
    return dm.sevir_train[num]


def load_sevir():
    dm = SEVIRLightningDataModule()
    dm.setup()
    train_sampler, train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    return train_loader, train_sampler, test_loader, val_loader

def vil_to_rainfall(x):
    x = x.clone()
    out = torch.full_like(x, torch.nan)
    mask0 = x <= 5
    mask1 = (x > 5) & (x <= 18)
    mask2 = (x > 18) & (x <= 254)
    out[mask0] = 0
    out[mask1] = (x[mask1] - 2) / 90.66
    out[mask2] = torch.exp((x[mask2] - 83.9) / 38.9)
    return out

# if __name__ == '__main__':
#     from gwr_loss import gwr_supervised_loss_single_fit, gwr_supervised_loss_multi_channel
#     from linear_loss import linear_regression_loss_windowed_multi_channel
#     from gwr_loss_boost import gwr_gpu_loss_fast_multi_channel, gwr_gpu_loss_global
#     dm = SEVIRLightningDataModule(use_distributed=False)
#     dm.setup()

#     for i in range(10):
#         input_img = get_train_images(dm, i)
#         vil = input_img['vil']
#         terrain = input_img['terrain']
#         print(vil.shape)
#         print(terrain.shape)

#     train_loader = dm.train_dataloader()
#     for batch in train_loader:
#         try:
#             vil = batch["vil"]           # [B, T, 1, H, W]
#             terrain = batch["terrain"]   # [2, H, W]
#             # terrain = terrain[0,:,:]
#             pred = torch.randn_like(vil)  # 模拟预测值

#             # loss = linear_regression_loss_windowed_multi_channel(pred, vil, terrain,window_size=64, device=vil.device)
#             # print("✅ Linear window loss:", loss.item())

#             # loss = gwr_gpu_loss_global(pred, vil, terrain, bandwidth=100, device=vil.device)
#             # print("✅ GWR boost gloal loss:", loss.item())

#             # loss = gwr_gpu_loss_fast_multi_channel(pred, vil, terrain,64,10.0,device=vil.device)
#             # print("✅ GWR boost window loss:", loss.item())

#             # loss = gwr_supervised_loss_multi_channel(pred, vil, terrain, device=vil.device)
#             # print("✅ GWR windowed loss:", loss.item())
#             # break
#             import time

#             # # === Linear Regression Loss ===
#             # start = time.time()
#             # loss = linear_regression_loss_windowed_multi_channel(pred, vil, terrain, window_size=64, device=vil.device)
#             # end = time.time()
#             # print(f"✅ Linear window loss: {loss.item():.6f} | Time: {end - start:.4f} sec")

#             # # === GWR Global Loss ===
#             # start = time.time()
#             # loss = gwr_gpu_loss_global(pred, vil, terrain, bandwidth=100, device=vil.device)
#             # end = time.time()
#             # print(f"✅ GWR boost global loss: {loss.item():.6f} | Time: {end - start:.4f} sec")

#             # # === GWR Fast Windowed Loss ===
#             # start = time.time()
#             # loss = gwr_gpu_loss_fast_multi_channel(pred, vil, terrain, 64, 10.0, device=vil.device)
#             # end = time.time()
#             # print(f"✅ GWR boost window loss: {loss.item():.6f} | Time: {end - start:.4f} sec")

#             # === MGWR Loss ===
#             start = time.time()
#             loss = gwr_supervised_loss_multi_channel(pred, vil, terrain, device=vil.device)
#             end = time.time()
#             print(f"✅ MGWR loss: {loss.item():.6f} | Time: {end - start:.4f} sec")

#             break


#         except Exception as e:
#             print(f"⚠️ GWR loss error: {e}")
#             break

#✅ Linear window loss: 0.013232 | Time: 2.8797 sec
#✅ GWR boost global loss: 0.027268 | Time: 24.0174 sec
#✅ GWR boost window loss: 0.015943 | Time: 29308.0947 sec

# if __name__ == '__main__':
#     dm = SEVIRLightningDataModule(use_distributed=False)
#     dm.setup()

#     train_loader = dm.train_dataloader()

#     for batch in train_loader:
#         try:
#             # === 1. 提取 VIL 和路径 ===
#             vil = batch['vil'][0]  # (1, 49, 1, 384, 384)
#             vil = vil.squeeze(0)  # (49, 1, 384, 384)
#             vil_avg = vil.mean(dim=0).squeeze(0)  # (384, 384)
#             vil_avg = vil_avg * 255
#             # vil_kgm2 = vil_to_rainfall(vil_avg)  # torch.Size([384, 384])
#             vil_kgm2 = vil_avg

#             print("VIL 平均值图 shape:", vil_kgm2.shape)

#             # === 2. 读取 DEM 地形 ===
#             terrain_path = batch['terrain_path'][0]
#             print("对应地形文件路径:", terrain_path)
#             with rasterio.open(terrain_path) as src:
#                 dem = src.read(1).astype(np.float32)  # shape: (384, 384)

#             # === 裁剪中心 128×128 区域
#             crop = slice(128, 256)
#             vil_kgm2 = vil_kgm2[crop, crop]
#             dem = dem[crop, crop]

#             # === 3. 准备数据和坐标 ===
#             h, w = vil_kgm2.shape
#             xx, yy = np.meshgrid(np.arange(w), np.arange(h))
#             coords = np.vstack([xx.flatten(), yy.flatten()]).astype(np.float32).T
#             y = vil_kgm2.flatten().numpy()
#             x = dem.flatten().reshape(-1, 1)
#             mask = ~np.isnan(y)

#             y_valid = y[mask]
#             x_valid = x[mask]
#             coords_valid = coords[mask]

#             print(f"有效像素数量: {len(y_valid)} / {384*384}")

#             # === 4. 标准化 + GWR 带宽选择 ===
#             scaler = StandardScaler()
#             x_std = scaler.fit_transform(x_valid)
#             if x_std.ndim == 1:
#                 x_std = x_std.reshape(-1, 1)
#             if np.std(x_valid) == 0:
#                 print("跳过：DEM 高度无变化")
#                 continue  # 当前样本不具备回归意义

#             print("开始带宽选择...")
#             bw = Sel_BW(coords_valid, y_valid, x_std, n_jobs=1).search(bw_min=10)
#             print(f"选定带宽: {bw}")

#             # === 5. 执行 GWR 拟合 ===
#             gwr_model = GWR(coords_valid, y_valid, x_std, bw, n_jobs=1)
#             results = gwr_model.fit()
#             print("GWR 拟合完成。")

#             # === 6. 生成斜率图 ===
#             slope_map = np.full((384 * 384,), np.nan)
#             mask_full = np.full((384, 384), False)
#             mask_full[crop, crop] = mask.reshape(128, 128)
#             slope_map[mask_full.flatten()] = results.params[:, 1]
#             slope_map = slope_map.reshape(384, 384)

#             print("局部斜率分布:")
#             print("  最小值:", np.nanmin(slope_map))
#             print("  最大值:", np.nanmax(slope_map))
#             print("  平均值:", np.nanmean(slope_map))

#             # 可视化
#             import matplotlib.pyplot as plt
#             plt.imshow(slope_map, cmap='coolwarm')
#             plt.title("GWR 地形斜率对降水影响")
#             plt.colorbar(label='Slope')
#             plt.show()

#             break  # 找到一个成功样本后退出

#         except Exception as e:
#             print(f"当前样本 GWR 失败，跳过：{e}")
#             continue
