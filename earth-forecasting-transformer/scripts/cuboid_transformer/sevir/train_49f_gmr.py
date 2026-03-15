"""
Earthformer 49Frame GMR-Conv 实验训练脚本
=========================================
在 Earthformer 49帧 baseline 基础上，将所有 Conv2d 替换为 GMR_Conv2d
(Gaussian Mixture Ring Convolution, arXiv:2504.02819)。

GMR-Conv 支持连续旋转等变 (任意角度)，drop-in 替换 nn.Conv2d，
无需 FieldType/GeometricTensor 包装。

Transformer 部分完全不变。

用法:
    cd earth-forecasting-transformer/scripts/cuboid_transformer/sevir
    conda activate rtx3050ti_cu128
    python -u train_49f_gmr.py --cfg cfg_sevir_49frame_gmr.yaml --epochs 10 --save exp_49f_gmr_baseline
"""

import warnings
warnings.filterwarnings("ignore", ".*isinstance.*LeafSpec.*")
warnings.filterwarnings("ignore", ".*StructuralSimilarityIndexMeasure.*")
warnings.filterwarnings("ignore", ".*indexing with dtype torch.uint8.*")
import matplotlib
matplotlib.use('Agg')
import logging
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import omegaconf

# PyTorch 2.6+ weights_only=True default fix
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.visualization.sevir.sevir_vis_seq import save_example_vis_results
from earthformer.metrics.sevir import SEVIRSkillScore
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule

# === GMR-Conv 替换 ===
# 修复: 只替换 initial_encoder + final_decoder，不动 FFN 的 1x1 Conv
from gmr_patch_embed import patch_model_with_gmr_embed, count_gmr_embed_layers

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")


class CuboidSEVIRGMRModule(pl.LightningModule):
    """Earthformer + GMR-Conv: 将 Conv2d 替换为连续旋转等变 GMR_Conv2d"""

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super().__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        # 1. 先创建标准 Earthformer 模型
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        # 2. 用 GMR-Conv 替换 initial_encoder + final_decoder (BUG FIX: 不替换 FFN 的 1x1 Conv)
        patch_model_with_gmr_embed(
            self.torch_nn_module,
            in_chans=model_cfg["input_shape"][-1],
            embed_dim=model_cfg["base_units"],
        )
        stats = count_gmr_embed_layers(self.torch_nn_module)
        print(f"[GMR-Conv] 已替换 initial_encoder + final_decoder: {stats}")

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode, seq_len=self.out_len, layout=self.layout,
            threshold_list=self.threshold_list, metrics_list=self.metrics_list, eps=1e-4)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode, seq_len=self.out_len, layout=self.layout,
            threshold_list=self.threshold_list, metrics_list=self.metrics_list, eps=1e-4)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                from shutil import copyfile
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = OmegaConf.create({"in_len": 37, "out_len": 12, "layout": "NTHWC"})
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        return OmegaConf.create({
            "dataset_name": "sevir", "img_height": 384, "img_width": 384,
            "in_len": 37, "out_len": 12, "seq_len": 49,
            "plot_stride": 2, "interval_real_time": 5,
            "sample_mode": "sequent", "stride": 37, "layout": "NTHWC",
            "start_date": [2017, 6, 13], "train_val_split_date": [2017, 8, 15],
            "train_test_split_date": [2017, 9, 15], "end_date": [2017, 10, 15],
            "metrics_mode": "0",
            "metrics_list": ['csi', 'pod', 'sucr', 'bias'],
            "threshold_list": [16, 74, 133, 160, 181, 219],
        })

    @staticmethod
    def get_optim_config():
        return OmegaConf.create({
            "total_batch_size": 8, "micro_batch_size": 1, "seed": 0,
            "method": "adamw", "lr": 0.001, "wd": 0.0,
            "gradient_clip_val": 1.0, "max_epochs": 50,
            "lr_scheduler_mode": "cosine", "min_lr_ratio": 1.0e-3,
            "warmup_min_lr_ratio": 0.0, "warmup_percentage": 0.2,
            "early_stop": True, "early_stop_mode": "min",
            "early_stop_patience": 20, "save_top_k": 1,
        })

    @staticmethod
    def get_logging_config():
        return OmegaConf.create({
            "logging_prefix": "Earthformer_49Frame_GMR",
            "monitor_lr": True, "monitor_device": False,
            "track_grad_norm": -1, "use_wandb": False,
        })

    @staticmethod
    def get_trainer_config():
        return OmegaConf.create({
            "check_val_every_n_epoch": 1, "log_step_ratio": 0.001,
            "precision": "bf16-mixed",
        })

    @staticmethod
    def get_vis_config():
        return OmegaConf.create({
            "train_example_data_idx_list": [0],
            "val_example_data_idx_list": [80],
            "test_example_data_idx_list": [0, 80, 160, 240, 320, 400],
            "eval_example_only": False, "plot_stride": 2,
        })

    @staticmethod
    def get_model_config():
        return OmegaConf.create({
            "input_shape": [37, 384, 384, 1],
            "target_shape": [12, 384, 384, 1],
            "base_units": 64, "block_units": None, "scale_alpha": 1.0,
            "enc_depth": [1, 1], "dec_depth": [1, 1],
            "enc_use_inter_ffn": True, "dec_use_inter_ffn": True,
            "dec_hierarchical_pos_embed": False,
            "downsample": 2, "downsample_type": "patch_merge",
            "upsample_type": "upsample",
            "num_global_vectors": 8, "use_dec_self_global": False,
            "dec_self_update_global": True, "use_dec_cross_global": False,
            "use_global_vector_ffn": False, "use_global_self_attn": True,
            "separate_global_qkv": True, "global_dim_ratio": 1,
            "self_pattern": "axial", "cross_self_pattern": "axial",
            "cross_pattern": "cross_1x1", "dec_cross_last_n_frames": None,
            "attn_drop": 0.1, "proj_drop": 0.1, "ffn_drop": 0.1, "num_heads": 4,
            "ffn_activation": "gelu", "gated_ffn": False,
            "norm_layer": "layer_norm", "padding_type": "zeros",
            "pos_embed_type": "t+h+w", "use_relative_pos": True,
            "self_attn_use_final_proj": True, "dec_use_first_self_attn": False,
            "z_init_method": "zeros", "checkpoint_level": 2,
            "initial_downsample_type": "stack_conv",
            "initial_downsample_activation": "leaky",
            "initial_downsample_stack_conv_num_layers": 3,
            "initial_downsample_stack_conv_dim_list": [4, 16, 64],
            "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
            "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],
            "attn_linear_init_mode": "0", "ffn_linear_init_mode": "0",
            "conv_init_mode": "0", "down_up_linear_init_mode": "0",
            "norm_init_mode": "0",
        })

    @staticmethod
    def get_total_num_steps(epoch, num_samples, total_batch_size):
        return epoch * num_samples // total_batch_size

    def forward(self, batch):
        data = batch['vil']
        in_seq = data[:, :self.in_len, ...]
        target_seq = data[:, self.in_len:self.in_len + self.out_len, ...]
        pred_seq = self.torch_nn_module(in_seq)
        loss = F.mse_loss(pred_seq, target_seq) + F.l1_loss(pred_seq, target_seq)
        return pred_seq, loss, in_seq, target_seq

    def training_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq = self(batch)
        micro_batch_size = pred_seq.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx, in_seq=in_seq, target_seq=target_seq, pred_seq=pred_seq.detach(),
                mode="train")
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq = self(batch)
        micro_batch_size = pred_seq.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx, in_seq=in_seq, target_seq=target_seq, pred_seq=pred_seq.detach(),
                mode="val")
        self.valid_mse(pred_seq.squeeze(-1), target_seq.squeeze(-1))
        self.valid_mae(pred_seq.squeeze(-1), target_seq.squeeze(-1))
        self.valid_score.update(pred_seq, target_seq)
        self.log('valid_loss', loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log("valid_mse", valid_mse, prog_bar=True, on_epoch=True)
        self.log("valid_mae", valid_mae, prog_bar=True, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        valid_score = self.valid_score.compute()
        self.valid_score.reset()

    def test_step(self, batch, batch_idx):
        pred_seq, loss, in_seq, target_seq = self(batch)
        micro_batch_size = pred_seq.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx, in_seq=in_seq, target_seq=target_seq, pred_seq=pred_seq.detach(),
                mode="test")
        self.test_mse(pred_seq.squeeze(-1), target_seq.squeeze(-1))
        self.test_mae(pred_seq.squeeze(-1), target_seq.squeeze(-1))
        self.test_score.update(pred_seq, target_seq)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log("test_mse", test_mse, prog_bar=True, on_epoch=True)
        self.log("test_mae", test_mae, prog_bar=True, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        test_score = self.test_score.compute()
        self.test_score.reset()

    def configure_optimizers(self):
        # GMR-Conv sigma 参数不需要 weight decay
        from GMR_Conv import GMR_Conv2d
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # GMR sigma 参数单独处理 (不做 weight decay)
        sigma_params = []
        other_no_decay = []
        other_decay = []
        for n, p in self.torch_nn_module.named_parameters():
            if 'gaussian_sigma' in n or 'sigma' in n:
                sigma_params.append(p)
            elif n in decay_parameters:
                other_decay.append(p)
            else:
                other_no_decay.append(p)

        optimizer_grouped_parameters = [
            {"params": other_decay, "weight_decay": self.wd},
            {"params": other_no_decay, "weight_decay": 0.0},
            {"params": sigma_params, "weight_decay": 0.0, "lr": self.lr * 0.1},  # sigma 用更小学习率
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.wd)
        warmup_iter = int(np.round(self.warmup_percentage * self.total_num_steps))
        if self.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda(
                warmup_steps=warmup_iter, min_lr_ratio=getattr(self, 'warmup_min_lr_ratio', 0.0)))
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.total_num_steps - warmup_iter),
                                                  eta_min=self.lr * self.min_lr_ratio)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        else:
            lr_scheduler_config = None
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def save_vis_step_end(self, data_idx, in_seq, target_seq, pred_seq, mode="train"):
        if mode == "train":
            example_data_idx_list = self.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}!")
        if data_idx in example_data_idx_list:
            save_example_vis_results(
                save_dir=self.example_save_dir, save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                in_seq=in_seq.detach().float().cpu().numpy(),
                target_seq=target_seq.detach().float().cpu().numpy(),
                pred_seq=pred_seq.detach().float().cpu().numpy(),
                layout=self.layout, plot_stride=self.oc.vis.plot_stride,
                label=self.logging_prefix, interval_real_time=self.oc.dataset.interval_real_time)

    def set_trainer_kwargs(self, devices=1, accumulate_grad_batches=1, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs
        callbacks = [
            TQDMProgressBar(refresh_rate=1),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor="valid_loss_epoch", patience=self.oc.optim.early_stop_patience,
                          mode=self.oc.optim.early_stop_mode, verbose=True),
            ModelCheckpoint(dirpath=os.path.join(self.save_dir, "checkpoints"),
                            filename='{epoch:03d}', monitor="valid_loss_epoch",
                            mode=self.oc.optim.early_stop_mode,
                            save_top_k=self.oc.optim.save_top_k, save_last=True, verbose=True),
        ]
        logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir, name="tb_logs")
        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_kwargs = dict(
            devices=devices, max_epochs=max_epochs,
            callbacks=callbacks, logger=logger,
            log_every_n_steps=log_every_n_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision,
        )
        return trainer_kwargs


# ============================================================
#  Data Module — 与 E2CNN 实验相同
# ============================================================

sys.path.append(r"c:\Users\Lenovo\Desktop\MOE\datswinlstm_memory")
from sevir_torch_wrap import SEVIRTorchDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class AlignedSEVIRDataModule(LightningDataModule):
    def __init__(self, dataset_oc, micro_batch_size, num_workers):
        super().__init__()
        import datetime
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        sevir_catalog = r"X:\datasets\sevir\CATALOG.csv"
        sevir_data_dir = r"X:\datasets\sevir\data"
        self.train_dataset = SEVIRTorchDataset(
            sevir_catalog=sevir_catalog, sevir_data_dir=sevir_data_dir,
            seq_len=dataset_oc['in_len'] + dataset_oc['out_len'],
            batch_size=micro_batch_size,
            start_date=datetime.datetime(*dataset_oc['start_date']),
            end_date=datetime.datetime(*dataset_oc['train_val_split_date']),
            shuffle=True, verbose=True, layout="NTHWC")
        self.val_dataset = SEVIRTorchDataset(
            sevir_catalog=sevir_catalog, sevir_data_dir=sevir_data_dir,
            seq_len=dataset_oc['in_len'] + dataset_oc['out_len'],
            batch_size=micro_batch_size,
            start_date=datetime.datetime(*dataset_oc['train_val_split_date']),
            end_date=datetime.datetime(*dataset_oc['train_test_split_date']),
            shuffle=False, verbose=True, layout="NTHWC")
        self.num_train_samples = len(self.train_dataset)

    def collate_fn(self, batch):
        return {"vil": torch.stack(batch, dim=0)}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.micro_batch_size,
                          shuffle=True, drop_last=True, collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.micro_batch_size,
                          shuffle=False, drop_last=False, collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        # BUG FIX: 返回 test_dataset 而不是 val_dataset
        return DataLoader(self.test_dataset, batch_size=self.micro_batch_size,
                          shuffle=False, drop_last=False, collate_fn=self.collate_fn,
                          num_workers=self.num_workers)

    def prepare_data(self): pass
    def setup(self, stage=None): pass


# ============================================================
#  Main
# ============================================================


def get_parser():
    parser = argparse.ArgumentParser(description='Earthformer 49F GMR-Conv Baseline')
    parser.add_argument('--save', default='exp_49f_gmr_baseline', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default='cfg_sevir_49frame_gmr.yaml', type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str)
    return parser


def main():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    parser = get_parser()
    args = parser.parse_args()

    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = args.epochs if args.epochs is not None else oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_oc = OmegaConf.to_object(CuboidSEVIRGMRModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = args.epochs or 50
        seed = 0

    seed_everything(seed, workers=True)
    num_workers = 0

    dm = AlignedSEVIRDataModule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers)
    dm.prepare_data()
    dm.setup()

    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidSEVIRGMRModule.get_total_num_steps(
        epoch=max_epochs, num_samples=dm.num_train_samples, total_batch_size=total_batch_size)

    pl_module = CuboidSEVIRGMRModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)

    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs)

    trainer = Trainer(**trainer_kwargs)

    if args.test:
        assert args.ckpt_name is not None
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    else:
        ckpt_path = None
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not found, training from scratch.")
                ckpt_path = None

        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)

        best_path = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        if not best_path or not os.path.exists(best_path):
            best_path = os.path.join(pl_module.save_dir, "checkpoints", "last.ckpt")
        if os.path.exists(best_path):
            state_dict = pl_ckpt_to_pytorch_state_dict(
                checkpoint_path=best_path, map_location=torch.device("cpu"),
                delete_prefix_len=len("torch_nn_module."))
            torch.save(state_dict, os.path.join(pl_module.save_dir, "earthformer_sevir_gmr.pt"))
            print(f"[GMR-Conv] Best model saved: {pl_module.save_dir}/earthformer_sevir_gmr.pt")

        trainer.test(model=pl_module, datamodule=dm)


if __name__ == '__main__':
    main()
