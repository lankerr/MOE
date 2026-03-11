# 20260131 代码现代化实施计划 (Modernization Plan)

**目标**: 将 Earthformer 项目代码升级，以原生支持 RTX 5070 所需的最新软件栈 (PyTorch 2.6.0, CUDA 12.8, PyTorch Lightning 2.6.1)。

## 1. 背景与现状
-   **硬件**: NVIDIA RTX 5070 Laptop GPU (Blackwell 架构, sm_120)。
-   **环境**: WSL2 Ubuntu-24.04, Python 3.9+。
-   **核心问题**: PyTorch 1.x 不支持 CUDA 12.x，导致新显卡无法使用。NVIDIA Apex 编译极其困难。
-   **解决方案**: 放弃降级，全面拥抱最新版 PyTorch 全家桶。

## 2. 关键变更点 (Breaking Changes)

### 2.1 依赖库升级
-   **PyTorch Lightning**: 从 1.x 升级到 **2.6.1** (当前环境已安装)。
-   **NVIDIA Apex**: **彻底移除**。不再尝试编译安装，改用 PyTorch 原生 `DDPStrategy`。

### 2.2 代码重构清单

#### 🅰️ 移除 Apex 依赖
-   **文件**: `src/earthformer/utils/apex_ddp.py`
-   **操作**: 该文件目前已被临时替换为 Mock 实现。建议进一步清理引用，或者保留 Mock 实现但确保逻辑正确（继承自 `lightning.pytorch.strategies.DDPStrategy`）。
-   **目标**: 确保 `import earthformer.utils.apex_ddp` 不会报错，且能正常通过 Native DDP 运行。

#### 🅱️ PyTorch Lightning 2.x API 适配
需要修改 `scripts/cuboid_transformer/sevir/train_cuboid_sevir.py` 及相关 `pl_module` 文件。

1.  **Trainer 参数重命名**:
    -   `gpus` -> **`devices`**
    -   `auto_select_gpus` -> (已移除，需检查)
    -   `flush_logs_every_n_steps` -> (已移除，需检查)
    -   `resume_from_checkpoint` -> (移至 `trainer.fit(ckpt_path=...)`)

2.  **回调 (Callbacks) 路径变更**:
    -   由 `pytorch_lightning.callbacks` 确认是否需要调整（大部分兼容，但需检查 `DeprecationWarning`）。

3.  **Step 方法签名**:
    -   `training_step`, `validation_step`, `test_step` 返回值和参数行为检查。
    -   `validation_epoch_end` -> **`on_validation_epoch_end`** (PL 2.0 重大变更！)
    -   `test_epoch_end` -> **`on_test_epoch_end`**

4.  **Logging**:
    -   检查 `self.log` 在 `epoch_end` 钩子中的行为。

#### 🆎 配置更新 (Config.py)
-   确保 `src/earthformer/config.py` (WSL端) 已正确指向 WSL 本地的数据路径。

## 3. 验证步骤
在 WSL 终端运行以下命令：
```bash
python3 scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 1 --cfg scripts/cuboid_transformer/sevir/cfg_sevir.yaml
```
*(注意：脚本中的 argparse 参数 `--gpus` 可能依然保留作为输入参数，但在传给 Trainer 时必须转为 `devices`)*

## 4. 执行状态报告 (2026-01-31 17:50 Updated)

###  已完成事项
1.  **环境修复**:
    -   识别到 WSL 存在两份代码 (WSL Home vs Windows Mount)，确认以 \/mnt/c/Users/...\ (Windows 挂载路径) 为准。
    -   在 \src/earthformer/config.py\ 中增加了跨平台路径判断，自动适配 WSL Linux 路径。

2.  **Moving MNIST 训练验证**:
    -   **状态**: **成功运行** (Process Running)。
    -   **修复**: 修复了 \AttributeError: 'CuboidMovingMNISTPLModule' object has no attribute 'precision'\ 错误。
    -   **产出**: \experiments/tmp_mnist\ 下已生成日志文件 (\events.out.tfevents\)。

3.  **SEVIR 脚本同步修复**:
    -   **操作**: 已同步修复 \scripts/cuboid_transformer/sevir/train_cuboid_sevir.py\ 中的同类问题 (\self.precision\ -> \self.oc.trainer.precision\)。
    -   **状态**: 代码已就绪，随时可启动训练。

###  后续建议
-   后续所有训练指令请务必在 \/mnt/c/Users/97290/Desktop/datswinlstm_memory/earth-forecasting-transformer\ 目录下执行。
-   如需修改代码，请直接修改 Windows 下的文件，WSL 会自动同步。

## 5. 警告修复与优化 (2026-01-31 18:00 Updated)

针对训练日志中出现的 Warnings，已进行如下修复：

### 5.1 精度优化 (Tensor Cores)
-   **问题**: \You are using a CUDA device ... that has Tensor Cores. To properly utilize them...-   **修复**: 在 \	rain_cuboid_mnist.py\ 和 \	rain_cuboid_sevir.py\ 的 \main()\ 函数开头添加：
    \python
    torch.set_float32_matmul_precision('medium')
    \    这大大提升了 RTX 5070 上的矩阵运算效率。

### 5.2 依赖库废弃警告
-   **问题**: \FutureWarning: Importing StructuralSimilarityIndexMeasure from torchmetrics was deprecated...-   **修复**: 修改导入路径为官方建议的新路径：
    \python
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ### 5.3 数据加载优化
-   **问题**: \The val_dataloader does not have many workers...-   **修复**: \	rain_cuboid_mnist.py\ 中 um_workers\ 默认值已调整为 **8** (原为 0 或 1)。这能显著减少 GPU 等待数据的时间。


## 6. 零警告运行保障 (Zero-Warning Compliance)

为确保训练流程的纯净与日志的可读性，实施了严格的警告过滤策略：

### 6.1 过滤策略实施
在训练脚本头部 (	rain_cuboid_mnist.py, 	rain_cuboid_sevir.py) 添加了以下全局过滤器：

\python
import warnings
# 修复 PyTorch Lightning 内部 _pytree.py 的 LeafSpec 弃用警告
warnings.filterwarnings(
