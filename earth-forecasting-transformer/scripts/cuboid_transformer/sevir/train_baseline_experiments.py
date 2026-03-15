"""
Baseline Experiments Training Script
====================================

支持的实验配置:
- 高优先级 2h→2h (1:1): cfg_sevir_baseline_2h_to_2h.yaml
- 高优先级 4h→4h (1:1): cfg_sevir_baseline_4h_to_4h.yaml
- 中优先级 2h→3h (1:1.5): cfg_sevir_baseline_2h_to_3h.yaml
- 低优先级 4h→3h (4:3): cfg_sevir_baseline_4h_to_3h.yaml

使用方法:
    # 运行单个实验
    python train_baseline_experiments.py --exp 2h_to_2h

    # 运行所有高优先级实验
    python train_baseline_experiments.py --priority high

    # 运行所有实验
    python train_baseline_experiments.py --all

    # 快速测试 (1 epoch)
    python train_baseline_experiments.py --exp 2h_to_2h --test_run
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from earthformer.cuboid_transformer.cuboid_transformer_model import CuboidTransformerModel
from utils.dataset_wrapper import SEVIRDataModule


# 实验配置映射
EXPERIMENT_CONFIGS = {
    # SEVIR 数据集实验
    '2h_to_2h': {
        'config': 'cfg_sevir_baseline_2h_to_2h.yaml',
        'priority': 'high',
        'description': 'SEVIR 2h→2h (1:1) - 与70%论文一致',
    },
    '4h_to_4h': {
        'config': 'cfg_sevir_baseline_4h_to_4h.yaml',
        'priority': 'high',
        'description': 'SEVIR 4h→4h (1:1) - 测试可预测性上限',
    },
    '2h_to_3h': {
        'config': 'cfg_sevir_baseline_2h_to_3h.yaml',
        'priority': 'medium',
        'description': 'SEVIR 2h→3h (1:1.5) - 探索性实验',
    },
    '4h_to_3h': {
        'config': 'cfg_sevir_baseline_4h_to_3h.yaml',
        'priority': 'low',
        'description': 'SEVIR 4h→3h (4:3) - 探索性实验',
    },
    # 重庆数据集实验
    'chongqing_2h_to_2h': {
        'config': 'cfg_chongqing_baseline_2h_to_2h.yaml',
        'priority': 'high',
        'description': '重庆雷达 2h→2h - 山地地形预报',
    },
}


def load_config(config_path):
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_trainer(config, exp_name, output_dir):
    """创建 PyTorch Lightning Trainer"""
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'{exp_name}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=config.get('optim', {}).get('save_top_k', 1),
        save_last=True,
    )

    # AutoStop callback (自适应早停)
    from earthformer.utils.adaptive_lr import AutoStopCallback
    early_stop_patience = config.get('optim', {}).get('early_stop_patience', 15)
    auto_stop = AutoStopCallback(
        patience=early_stop_patience,
        min_delta=1e-4,
        monitor='val_loss',
        mode='min',
        min_lr=1e-6,
        verbose=True,
    )

    # Wrap as Lightning callback
    class AutoStopLightningCallback(pl.Callback):
        def __init__(self, auto_stop_inner):
            self.auto_stop = auto_stop_inner

        def on_validation_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get('val_loss')
            current_lr = trainer.optimizers[0].param_groups[0]['lr']

            # 获取梯度norm (需要从模型中获取)
            grad_norm = None
            if hasattr(pl_module, 'grad_norm'):
                grad_norm = pl_module.grad_norm

            if val_loss is not None:
                should_stop = self.auto_stop(float(val_loss), float(current_lr), grad_norm)
                if should_stop:
                    trainer.should_stop = True

    early_stop_callback = AutoStopLightningCallback(auto_stop)

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=exp_name,
    )

    # Trainer
    trainer_config = config.get('trainer', {})
    accumulate_grad_batches = trainer_config.get('accumulate_grad_batches', 1)

    trainer = pl.Trainer(
        max_epochs=config.get('optim', {}).get('max_epochs', 50),
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=int(trainer_config.get('log_step_ratio', 0.001) * 1000),
        precision=trainer_config.get('precision', 'bf16-mixed'),
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=config.get('optim', {}).get('gradient_clip_val', 1.0),
        gradient_clip_algorithm='norm',
        deterministic=True,
        enable_progress_bar=True,
    )

    return trainer


def run_experiment(exp_name, config_path, test_run=False):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    # 加载配置
    config = load_config(config_path)

    # 覆盖 test_run 设置
    if test_run:
        config['optim']['max_epochs'] = 1
        config['optim']['save_top_k'] = 1
        print("[TEST RUN] Only 1 epoch for testing\n")

    # 输出目录
    output_dir = Path('./outputs') / f'baseline_{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    import yaml
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Data Module - 支持 SEVIR 和重庆数据
    print("Initializing Data Module...")
    data_config = config.get('dataset', {})
    optim_config = config.get('optim', {})

    dataset_name = data_config.get('dataset_name', 'sevir')

    if dataset_name == 'chongqing':
        print("[Chongqing] Using Chongqing radar data...")
        from chongqing_datamodule import ChongqingDataModule
        datamodule = ChongqingDataModule(
            data_dir=data_config.get('data_dir', './data/chongqing'),
            batch_size=optim_config.get('micro_batch_size', 1),
            num_workers=4,
            in_len=data_config.get('in_len', 24),
            out_len=data_config.get('out_len', 24),
            img_size=(data_config.get('img_height', 384),
                     data_config.get('img_width', 384)),
            stride=data_config.get('stride', 24),
        )
    else:
        print("[SEVIR] Using SEVIR data...")
        from earthformer.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
        datamodule = SEVIRLightningDataModule(
            data_dir=data_config.get('data_dir', './data/sevir'),
            batch_size=optim_config.get('micro_batch_size', 1),
            num_workers=4,
            **data_config,
        )

    # Model
    print("Initializing Model...")
    model_config = config.get('model', {})
    model = CuboidTransformerModel(
        **model_config,
    )

    # Trainer
    print("Initializing Trainer...")
    trainer = create_trainer(config, exp_name, output_dir)

    # Train
    print(f"\nStarting training for {config['optim']['max_epochs']} epochs...")
    print(f"Output directory: {output_dir}\n")

    trainer.fit(model, datamodule=datamodule)

    # Test
    print("\nRunning test evaluation...")
    test_results = trainer.test(model, datamodule=datamodule, verbose=True)

    print(f"\n{'='*60}")
    print(f"Experiment {exp_name} completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    return test_results


def main():
    parser = argparse.ArgumentParser(description='Baseline Experiments Training')
    parser.add_argument('--exp', type=str, choices=list(EXPERIMENT_CONFIGS.keys()),
                       help='Experiment to run')
    parser.add_argument('--config', type=str, help='Custom config file path')
    parser.add_argument('--priority', type=str, choices=['high', 'medium', 'low'],
                       help='Run all experiments of given priority')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--test_run', action='store_true',
                       help='Test run with 1 epoch')

    args = parser.parse_args()

    # 确定要运行的实验
    experiments_to_run = []

    if args.config:
        # 自定义配置
        experiments_to_run.append(('custom', args.config))
    elif args.exp:
        experiments_to_run.append((args.exp, EXPERIMENT_CONFIGS[args.exp]['config']))
    elif args.priority:
        for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
            if exp_config['priority'] == args.priority:
                experiments_to_run.append((exp_name, exp_config['config']))
    elif args.all:
        for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
            experiments_to_run.append((exp_name, exp_config['config']))
    else:
        parser.print_help()
        print("\n[ERROR] Please specify --exp, --priority, --all, or --config")
        return

    # 打印实验计划
    print(f"\n{'='*60}")
    print(f"BASELINE EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"\nExperiment plan:")
    for exp_name, config_path in experiments_to_run:
        desc = EXPERIMENT_CONFIGS.get(exp_name, {}).get('description', 'Custom')
        print(f"  - {exp_name}: {desc}")
    print(f"{'='*60}\n")

    # 运行实验
    results = {}
    for exp_name, config_path in experiments_to_run:
        try:
            results[exp_name] = run_experiment(exp_name, config_path, args.test_run)
        except Exception as e:
            print(f"[ERROR] Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    for exp_name, result in results.items():
        if result:
            print(f"{exp_name}: {result}")
        else:
            print(f"{exp_name}: Failed")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
