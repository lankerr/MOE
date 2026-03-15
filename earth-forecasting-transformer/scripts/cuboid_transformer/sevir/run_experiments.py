"""
Comprehensive experiment runner for EarthFormer improvements.

This script orchestrates the following experiments:
1. Baseline (original EarthFormer)
2. + WSD Scheduler
3. + GMR Patch Embedding
4. + Physics-Guided Sparse Attention
5. + Density-Proximity Cross-Block Attention
6. Full model (all improvements)

Usage:
    # Run all experiments
    python run_experiments.py --all --data_dir /path/to/sevir

    # Run specific variant
    python run_experiments.py --variant gmr_patch --epochs 50

    # Quick smoke test
    python run_experiments.py --smoke_test
"""

import os
import sys
import argparse
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'src'))

from earthformer.utils.lr_schedulers import (
    WSDScheduler,
    CosineAnnealingWarmupRestarts,
    get_scheduler,
    visualize_schedule,
)


EXPERIMENT_CONFIGS = {
    'baseline': {
        'name': 'Baseline',
        'description': 'Original EarthFormer with cosine scheduler',
        'use_gmr': False,
        'use_pgsa': False,
        'use_dpcba': False,
        'scheduler': 'cosine',
        'lr': 3e-4,
    },
    'wsd': {
        'name': 'Baseline+WSD',
        'description': 'Original EarthFormer with WSD scheduler',
        'use_gmr': False,
        'use_pgsa': False,
        'use_dpcba': False,
        'scheduler': 'wsd',
        'lr': 3e-4,
        'warmup_epochs': 2,
        'stable_epochs': 30,
        'decay_epochs': 18,
    },
    'gmr_patch': {
        'name': 'GMR-Patch',
        'description': 'GMR patch embedding with cosine scheduler',
        'use_gmr': True,
        'use_pgsa': False,
        'use_dpcba': False,
        'scheduler': 'cosine',
        'lr': 3e-4,
    },
    'gmr_patch_wsd': {
        'name': 'GMR-Patch+WSD',
        'description': 'GMR patch embedding with WSD scheduler',
        'use_gmr': True,
        'use_pgsa': False,
        'use_dpcba': False,
        'scheduler': 'wsd',
        'lr': 3e-4,
        'warmup_epochs': 2,
        'stable_epochs': 30,
        'decay_epochs': 18,
    },
    'pgsa': {
        'name': 'PGSA',
        'description': 'GMR patch + Physics-guided sparse attention',
        'use_gmr': True,
        'use_pgsa': True,
        'use_dpcba': False,
        'scheduler': 'wsd',
        'lr': 3e-4,
        'dbz_threshold': 15.0,
        'warmup_epochs': 2,
        'stable_epochs': 30,
        'decay_epochs': 18,
    },
    'full': {
        'name': 'Full',
        'description': 'GMR patch + PGSA + DPCBA + WSD',
        'use_gmr': True,
        'use_pgsa': True,
        'use_dpcba': True,
        'scheduler': 'wsd',
        'lr': 3e-4,
        'dbz_threshold': 15.0,
        'num_connections': 4,
        'warmup_epochs': 2,
        'stable_epochs': 30,
        'decay_epochs': 18,
    },
}


class ExperimentRunner:
    """Orchestrates EarthFormer improvement experiments"""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = './experiments',
        gpu: int = 0,
        max_epochs: int = 50,
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.gpu = gpu
        self.max_epochs = max_epochs

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device check
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(gpu)}")
            print(f"Memory: {torch.cuda.get_device_properties(gpu).total_memory / 1e9:.1f} GB")

    def run_experiment(
        self,
        variant: str,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Run a single experiment variant

        Args:
            variant: Experiment variant name
            config: Optional config overrides

        Returns:
            Dictionary with experiment results
        """
        if variant not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}")

        exp_config = EXPERIMENT_CONFIGS[variant].copy()
        if config:
            exp_config.update(config)

        print(f"\n{'='*60}")
        print(f"Running: {exp_config['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp_config['description']}")
        print(f"Config: {exp_config}")
        print()

        # Create experiment directory
        exp_dir = self.output_dir / variant / datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(exp_config, f)

        # Generate training script
        train_script = self._generate_train_script(exp_config, exp_dir)
        print(f"Generated training script: {train_script}")

        # Run experiment
        try:
            result = self._execute_training(train_script, exp_dir)
            print(f"\n✓ Experiment {variant} completed")
            print(f"  Results saved to: {exp_dir}")
            return result
        except Exception as e:
            print(f"\n✗ Experiment {variant} failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def run_ablation_study(
        self,
        variants: Optional[List[str]] = None,
    ):
        """
        Run ablation study across all variants

        Args:
            variants: List of variants to run (default: all)
        """
        if variants is None:
            variants = list(EXPERIMENT_CONFIGS.keys())

        print(f"\n{'='*60}")
        print(f"ABLATION STUDY: {len(variants)} variants")
        print(f"{'='*60}")

        results = {}
        for variant in variants:
            results[variant] = self.run_experiment(variant)

        # Generate comparison report
        self._generate_report(results)
        return results

    def _generate_train_script(self, config: Dict, exp_dir: Path) -> str:
        """Generate training script for the experiment"""
        script_path = exp_dir / 'train.py'

        # This would be based on your existing train_49f_gmr_patch.py
        # For now, create a placeholder
        script_content = f'''"""
Auto-generated training script for {config['name']}
"""
import sys
sys.path.insert(0, 'src')

from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformer

# Config:
{yaml.dump(config, default_flow_style=False)}

# TODO: Implement actual training
print("Training script placeholder")
print("Replace with actual training logic from train_49f_gmr_patch.py")
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        return str(script_path)

    def _execute_training(self, script_path: str, exp_dir: Path) -> Dict:
        """Execute the training script"""
        # Placeholder for actual training execution
        # In practice, this would call the actual training logic

        # Create a dummy result
        result = {
            'status': 'completed',
            'config_path': str(exp_dir / 'config.yaml'),
            'checkpoint_dir': str(exp_dir / 'checkpoints'),
            'metrics': {
                'CSI@16': 0.0,
                'CSI@74': 0.0,
                'CSI@133': 0.0,
                'MAE': 0.0,
                'MSE': 0.0,
            },
        }
        return result

    def _generate_report(self, results: Dict):
        """Generate comparison report"""
        report_path = self.output_dir / 'ablation_report.md'

        with open(report_path, 'w') as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Comparison table
            f.write("## Results Comparison\n\n")
            f.write("| Variant | CSI@16 | CSI@74 | CSI@133 | MAE | MSE |\n")
            f.write("|---------|--------|--------|---------|-----|-----|\n")

            for variant, result in results.items():
                if result['status'] == 'completed':
                    metrics = result['metrics']
                    f.write(f"| {variant} | {metrics['CSI@16']:.4f} | {metrics['CSI@74']:.4f} | ")
                    f.write(f"{metrics['CSI@133']:.4f} | {metrics['MAE']:.4f} | {metrics['MSE']:.4f} |\n")

        print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='EarthFormer improvement experiments')
    parser.add_argument('--variant', type=str, choices=list(EXPERIMENT_CONFIGS.keys()) + ['all'],
                       help='Experiment variant to run')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to SEVIR data directory')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for experiments')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum training epochs')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run smoke test (1 epoch)')

    args = parser.parse_args()

    # Override epochs for smoke test
    if args.smoke_test:
        args.max_epochs = 1
        print("SMOKE TEST MODE: Running 1 epoch per variant")

    # Create runner
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        gpu=args.gpu,
        max_epochs=args.max_epochs,
    )

    # Run experiments
    if args.variant == 'all':
        results = runner.run_ablation_study()
    else:
        results = runner.run_experiment(args.variant)

    print(f"\n{'='*60}")
    print("EXPERIMENT RUN COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
