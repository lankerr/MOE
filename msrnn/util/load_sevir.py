import sys
import os
sys.path.append('/data_8t/WSG/code/MS-RNN-main/util')
from torch.utils.data.distributed import DistributedSampler
from matplotlib import colors
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from sevir_config import datacfg
import datetime
from sevir_torch_wrap import SEVIRTorchDataset
import torch.distributed as dist


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
        dataset_name = kwargs.get('dataset_name', 'sevir')
        self.setup_dataset(dataset_name)
        self.start_date = datetime.datetime(*kwargs.get('start_date')) if kwargs.get('start_date') else None
        self.train_val_split_date = datetime.datetime(*kwargs.get('train_val_split_date', (2019, 1, 1)))
        self.train_test_split_date = datetime.datetime(*kwargs.get('train_test_split_date', (2019, 6, 1)))
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
        self.sevir_train = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=True,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.start_date,
            end_date=self.train_val_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        self.sevir_val = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_val_split_date,
            end_date=self.train_test_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        self.sevir_test = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        
        self.sevir_predict = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,)
        
        print(f'Train set size: {len(self.sevir_train)}')
        print(f'Validation set size: {len(self.sevir_val)}')
        print(f'Test set size: {len(self.sevir_test)}')

    def train_dataloader(self):
        # sampler = DistributedSampler(self.sevir_train) if torch.cuda.device_count() > 1 else None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.sevir_train)
        else:
            sampler = None

        return sampler,DataLoader(self.sevir_train, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        # sampler = DistributedSampler(self.sevir_val) if torch.cuda.device_count() > 1 else None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.sevir_val)
        else:
            sampler = None
        return sampler,DataLoader(self.sevir_val, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        # sampler = DistributedSampler(self.sevir_test) if torch.cuda.device_count() > 1 else None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.sevir_test)
        else:
            sampler = None
        return DataLoader(self.sevir_test, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        sampler = DistributedSampler(self.sevir_predict) if torch.cuda.device_count() > 1 else None
        return DataLoader(self.sevir_predict, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)

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
    train_sampler,train_loader = dm.train_dataloader()
    valid_sampler,valid_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    return train_loader, train_sampler, test_loader, valid_loader



if __name__ == '__main__':
    # model, result = train_sevir()
    # print(result)
    dm = SEVIRLightningDataModule()
    dm.setup()
    input_img = get_train_images(dm, 0)
    print(input_img.shape)

    train_dataloader = dm.train_dataloader()

    for batch in train_dataloader:
        print(batch)
        break