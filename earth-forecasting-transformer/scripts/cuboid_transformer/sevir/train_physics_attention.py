"""
Training script for Physics-Guided Attention experiments on SEVIR dataset.

This script compares three architectures:
1. Baseline: Original EarthFormer
2. PGSA: Physics-Guided Sparse Attention (15dBZ masking)
3. PGSA+DPCBA: PGSA + Density-Proximity Cross-Block Attention

Usage:
    python train_physics_attention.py --variant baseline
    python train_physics_attention.py --variant pgsa
    python train_physics_attention.py --variant pgsa_dpcba
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from earthformer.config import config as earthformer_config
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformer
from earthformer.cuboid_transformer.physics_attention import (
    PhysicsGuidedSparseAttention,
    PGSAWrapper,
    create_pgsa_cuboid_attention,
    DensityProximityCrossBlockAttention,
    DPCBAWrapper,
    create_dpcba_cuboid_attention,
)


class PhysicsAttentionCuboidTransformer(CuboidTransformer):
    """
    Extended CuboidTransformer with physics-guided attention options.
    """

    def __init__(
        self,
        use_pgsa: bool = False,
        use_dpcba: bool = False,
        dbz_threshold: float = 15.0,
        num_connections: int = 4,
        **kwargs
    ):
        """
        Parameters
        ----------
        use_pgsa : bool
            Enable Physics-Guided Sparse Attention
        use_dpcba : bool
            Enable Density-Proximity Cross-Block Attention
        dbz_threshold : float
            dBZ threshold for PGSA
        num_connections : int
            Number of cross-block connections for DPCBA
        **kwargs
            Arguments for base CuboidTransformer
        """
        self.use_pgsa = use_pgsa
        self.use_dpcba = use_dpcba
        self.dbz_threshold = dbz_threshold
        self.num_connections = num_connections

        # Initialize base transformer
        super().__init__(**kwargs)

        # Replace attention layers with physics-guided variants
        if use_pgsa or use_dpcba:
            self._replace_attention_layers()

    def _replace_attention_layers(self):
        """Replace standard attention layers with physics-guided variants."""
        from earthformer.cuboid_transformer.cuboid_transformer_patterns import CuboidSelfAttentionPatterns

        # Find all CuboidSelfAttentionLayer instances
        for name, module in self.named_modules():
            if hasattr(module, '__class__') and 'CuboidSelfAttentionLayer' in module.__class__.__name__:
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.get_submodule(parent_name) if parent_name else self

                # Create physics-guided replacement
                if self.use_pgsa and self.use_dpcba:
                    # Combined PGSA + DPCBA
                    new_layer = create_pgsa_cuboid_attention(
                        dim=module.dim,
                        num_heads=module.num_heads,
                        cuboid_size=module.cuboid_size,
                        dbz_threshold=self.dbz_threshold,
                        masking_mode='hybrid',
                    )
                    # Add DPCBA wrapper
                    new_layer = DPCBAWrapper(
                        base_attention_layer=new_layer,
                        num_connections=self.num_connections,
                        enable_dpcba=True,
                    )
                elif self.use_pgsa:
                    # PGSA only
                    new_layer = create_pgsa_cuboid_attention(
                        dim=module.dim,
                        num_heads=module.num_heads,
                        cuboid_size=module.cuboid_size,
                        dbz_threshold=self.dbz_threshold,
                        masking_mode='hybrid',
                    )
                elif self.use_dpcba:
                    # DPCBA only
                    new_layer = create_dpcba_cuboid_attention(
                        dim=module.dim,
                        num_heads=module.num_heads,
                        cuboid_size=module.cuboid_size,
                        num_connections=self.num_connections,
                        enable_dpcba=True,
                    )
                else:
                    continue

                # Replace the layer
                setattr(parent, child_name, new_layer)
                print(f"Replaced {name} with physics-guided attention")

    def forward(self, x, dbz_values=None):
        """Forward with optional dBZ values for physics masking."""
        # This would require modifying the base class forward
        # For now, call parent forward
        return super().forward(x)


class SEVIRDataModule(pl.LightningDataModule):
    """
    SEVIR data module for physics attention experiments.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        in_len: int = 13,
        out_len: int = 12,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len

    def setup(self, stage=None):
        # Import SEVIR dataset (adapt to your setup)
        from earthformer.datasets.sevir import SEVIRTorchDataset
        # Implementation depends on your data setup
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)


def get_model_config(
    variant: str,
    input_shape: tuple = (13, 384, 384, 1),
    base_units: int = 64,
) -> Dict[str, Any]:
    """
    Get model configuration for the specified variant.

    Parameters
    ----------
    variant : str
        One of: 'baseline', 'pgsa', 'pgsa_dpcba'
    input_shape : tuple
        Input shape (T, H, W, C)
    base_units : int
        Base feature dimension

    Returns
    -------
    config : dict
        Model configuration
    """
    T, H, W, C = input_shape

    # Base configuration (SEVIR from EarthFormer paper)
    config = {
        # Model structure
        'input_shape': input_shape,
        'base_units': base_units,
        'num_blocks': [3, 3, 3, 3],  # Number of blocks per stage
        'embed_dims': [base_units * (2 ** i) for i in range(4)],  # [64, 128, 256, 512]
        'patch_embed_stride': (1, 4, 4),  # Initial downsampling
        'encoder_depths': [2, 2, 6, 2],
        'decoder_depths': [2, 2, 2, 2],
        'num_heads': [2, 4, 8, 16],

        # Attention pattern
        'att_pattern_list': [
            ['axial', 'axial', 'axial', 'axial'],
            ['axial', 'axial', 'axial', 'axial'],
            ['video_swin_2x8', 'video_swin_2x8', 'video_swin_2x8', 'video_swin_2x8'],
            ['video_swin_2x8', 'video_swin_2x8', 'video_swin_2x8', 'video_swin_2x8'],
        ],
        'use_global_vector': True,
        'global_vector_mode': 'local',

        # Physics attention flags
        'use_pgsa': variant in ['pgsa', 'pgsa_dpcba'],
        'use_dpcba': variant == 'pgsa_dpcba',
        'dbz_threshold': 15.0,
        'num_connections': 4,

        # Training
        'output_shape': (12, 384, 384, 1),  # 12-frame prediction
        'loss_mode': 'mse+logcosh',
        'metrics': ['mse', 'mae'],
    }

    return config


def create_experiment_name(variant: str) -> str:
    """Create experiment name based on variant."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sevir_physics_attention_{variant}_{timestamp}"


def train_physics_attention(
    variant: str = 'baseline',
    data_dir: str = './data/sevir',
    max_epochs: int = 50,
    gpus: int = 1,
    **kwargs
):
    """
    Train a physics attention variant.

    Parameters
    ----------
    variant : str
        Model variant: 'baseline', 'pgsa', 'pgsa_dpcba'
    data_dir : str
        Path to SEVIR data
    max_epochs : int
        Maximum training epochs
    gpus : int
        Number of GPUs to use
    **kwargs
        Additional training arguments
    """
    print(f"\n{'='*60}")
    print(f"Training Physics Attention Variant: {variant.upper()}")
    print(f"{'='*60}\n")

    # Get model configuration
    model_cfg = get_model_config(variant, **kwargs)

    # Create model
    model = PhysicsAttentionCuboidTransformer(**model_cfg)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data module
    datamodule = SEVIRDataModule(
        data_dir=data_dir,
        batch_size=kwargs.get('batch_size', 1),
        num_workers=kwargs.get('num_workers', 4),
    )

    # Setup logging
    exp_name = create_experiment_name(variant)
    logger = TensorBoardLogger(
        save_dir='./logs',
        name=exp_name,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./checkpoints/{exp_name}',
        filename=f'{{epoch}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=kwargs.get('accumulate_grad_batches', 8),
        **kwargs
    )

    # Train
    print("\nStarting training...\n")
    trainer.fit(model, datamodule=datamodule)

    # Test
    print("\nRunning test evaluation...\n")
    test_results = trainer.test(model, datamodule=datamodule)

    return model, test_results


def compare_variants(
    variants: list = ['baseline', 'pgsa', 'pgsa_dpcba'],
    **kwargs
):
    """
    Train and compare multiple variants.

    Parameters
    ----------
    variants : list
        List of variant names to train
    **kwargs
        Training arguments
    """
    results = {}

    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"# Training variant: {variant}")
        print(f"{'#'*60}\n")

        try:
            model, test_results = train_physics_attention(
                variant=variant,
                **kwargs
            )
            results[variant] = {
                'model': model,
                'test_results': test_results,
            }
        except Exception as e:
            print(f"Error training {variant}: {e}")
            results[variant] = {'error': str(e)}

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    for variant, res in results.items():
        if 'test_results' in res:
            print(f"\n{variant.upper()}:")
            for metric, value in res['test_results'][0].items():
                print(f"  {metric}: {value:.4f}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train physics attention variants')
    parser.add_argument('--variant', type=str, default='baseline',
                       choices=['baseline', 'pgsa', 'pgsa_dpcba', 'all'],
                       help='Model variant to train')
    parser.add_argument('--data_dir', type=str, default='./data/sevir',
                       help='Path to SEVIR data')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--accumulate_grad_batches', type=int, default=8,
                       help='Gradient accumulation steps')

    args = parser.parse_args()

    if args.variant == 'all':
        compare_variants(
            variants=['baseline', 'pgsa', 'pgsa_dpcba'],
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )
    else:
        train_physics_attention(
            variant=args.variant,
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )
