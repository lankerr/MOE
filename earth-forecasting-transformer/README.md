# Earth Forecasting Transformer (Earthformer) — SEVIR Training

Fork of [amazon-science/earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer) with modifications for running on Windows with NVIDIA Blackwell architecture (RTX 5070).

## Key Modifications

| Change                             | File                       | Description                                                                                                                             |
| ---------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **DDP Strategy**             | `train_cuboid_sevir.py`  | Uses native PyTorch Lightning DDPStrategy (no apex)                                                                                     |
| **Gradient Checkpointing**   | `cuboid_transformer.py`  | Added `use_reentrant=False` to all 14 `checkpoint.checkpoint()` calls                                                               |
| **Meshgrid Fix**             | `cuboid_transformer.py`  | Added `indexing='ij'` to `torch.meshgrid()` calls                                                                                   |
| **Progress Bar**             | `train_cuboid_sevir.py`  | `TQDMProgressBar(refresh_rate=10)` for clean terminal output                                                                          |
| **Model Summary**            | `train_cuboid_sevir.py`  | Disabled FLOPs model summary (avoids triton/precision warnings)                                                                         |
| **Windows DataLoader**       | `train_cuboid_sevir.py`  | `num_workers=0` (h5py objects cannot be pickled on Windows spawn)                                                                     |
| **PyTorch 2.6+ Checkpoints** | `train_cuboid_sevir.py`  | Monkey-patched `torch.load` with `weights_only=False` to allow PyTorch Lightning to load OmegaConf hyperparameters from checkpoints |
| **8GB VRAM Config**          | `cfg_sevir_rtx5070.yaml` | Optimized config for 8GB VRAM GPUs                                                                                                      |
| **Dataset Path**             | `config.py`              | Auto-detects Windows/Linux paths                                                                                                        |

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 5070 Laptop, Blackwell architecture)
- **CUDA**: 12.8+
- **Python**: 3.12+
- **PyTorch**: 2.11+ (nightly with Blackwell/SM100 support)
- **PyTorch Lightning**: 2.6+

## Environment Setup

```bash
# Create conda environment (example for CUDA 12.8)
conda create -n rtx5070_cu128 python=3.12
conda activate rtx5070_cu128

# Install PyTorch (nightly for Blackwell support)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
pip install pytorch_lightning torchmetrics omegaconf einops numpy pandas matplotlib PyYAML boto3 scikit-learn scikit-image h5py yacs scipy

# Install earthformer in development mode
cd earth-forecasting-transformer
pip install -e .
```

## Dataset Preparation

Download the [SEVIR dataset](https://sevir.mit.edu/) and place it under a `datasets` directory:

```
datasets/
└── sevir/
    ├── CATALOG.csv
    └── data/
        └── vil/
            ├── 2017/
            │   ├── SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5
            │   └── ...
            ├── 2018/
            │   └── ...
            └── 2019/
                └── ...
```

Update the dataset path in `src/earthformer/config.py`:

```python
cfg.datasets_dir = r"C:\path\to\your\datasets"  # Windows
# or
cfg.datasets_dir = "/path/to/your/datasets"       # Linux
```

## Training

### Quick Start (8GB VRAM GPU)

```bash
python scripts/cuboid_transformer/sevir/train_cuboid_sevir.py \
    --cfg scripts/cuboid_transformer/sevir/cfg_sevir_rtx5070.yaml \
    --gpus 1 \
    --save sevir_rtx5070_run
```

### Full Model (16GB+ VRAM GPU)

```bash
python scripts/cuboid_transformer/sevir/train_cuboid_sevir.py \
    --cfg scripts/cuboid_transformer/sevir/cfg_sevir.yaml \
    --gpus 1 \
    --save sevir_full_run
```

### Multi-GPU Training

```bash
python scripts/cuboid_transformer/sevir/train_cuboid_sevir.py \
    --cfg scripts/cuboid_transformer/sevir/cfg_sevir.yaml \
    --gpus 2 \
    --save sevir_multi_gpu_run
```

## Configuration Comparison

| Parameter            | `cfg_sevir.yaml` (Full) | `cfg_sevir_rtx5070.yaml` (8GB) |
| -------------------- | ------------------------- | -------------------------------- |
| `base_units`       | 128                       | 64                               |
| `micro_batch_size` | 1                         | 1                                |
| `total_batch_size` | 32                        | 8                                |
| `checkpoint_level` | 2                         | 2                                |
| `precision`        | 16-mixed                  | 16-mixed                         |
| `max_epochs`       | 100                       | 50                               |
| `num_heads`        | 4                         | 4                                |
| Est. VRAM usage      | ~14GB                     | ~6GB                             |

## Experiment Outputs

Training outputs are saved to `experiments/` directory:

```
experiments/
└── cuboid_sevir/
    └── sevir_rtx5070_run/
        ├── checkpoints/     # Model checkpoints
        ├── config.yaml      # Saved config
        └── tensorboard/     # Training logs
```

## Citation

```bibtex
@inproceedings{gao2022earthformer,
    title={Earthformer: Exploring Space-Time Transformers for Earth System Forecasting},
    author={Gao, Zhihan and Shi, Xingjian and Wang, Hao and Zhu, Yi and Wang, Yuyang and Li, Mu and Yeung, Dit-Yan},
    booktitle={NeurIPS},
    year={2022}
}
```
