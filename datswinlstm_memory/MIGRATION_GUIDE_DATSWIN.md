# DatSwinLSTM-Memory RTX 5070 适配指南

本指南帮助你在 RTX 5070 环境下运行 `datswinlstm_memory` 项目。

## 1. 环境准备

推荐使用 **`extrapolation_p39`** 环境 (Python 3.9 + PyTorch 2.x)。

```bash
conda activate extrapolation_p39
```

## 2. 关键修改点 (已自动应用)

我们已经对代码进行了如下优化，以适配 RTX 5070 和本地环境：

1.  **路径配置 (`config.py`)**:
    *   数据集路径已自动指向: `c:\Users\97290\Desktop\datswinlstm_memory` (请确保 SEVIR 数据在此目录下或其子目录 `sevir/` 中)。

2.  **精度优化 (Tensor Cores)**:
    *   我们在 `train.py` 和 `run_sevir_...py` 中添加了 `torch.set_float32_matmul_precision('medium')`。
    *   这将大幅提升 RTX 50 系列显卡的训练速度 (最高 2-3 倍)。

3.  **分布式兼容性**:
    *   修复了 `run_sevir_multiGpu_distributedSampler.py` 中 `DDPStrategy` 的导入问题，兼容 PyTorch Lightning 新老版本。

## 3. 运行指令

### 3.1 快速测试 (CPU/单卡调试)

使用简化版的 `train.py` 进行快速测试，确保数据和模型通路正常。

```bash
# 运行 1 个 epoch，使用较小的序列长度快速验证
python train.py --epochs 1 --batch_size 1 --seq_len 13 --input_frames 6 --output_frames 7
```

### 3.2 完整训练 (使用 PyTorch Lightning)

使用完整脚本进行实验。

```bash
python run_sevir_multiGpu_distributedSampler.py
```

## 4. 常见问题

*   **RuntimeError: CUDA out of memory**: 尝试减小 `--batch_size`。
*   **FileNotFoundError**: 请检查 `config.py` 中的 `datasets_dir` 是否正确指向了你的数据文件夹根目录。

## 5. WSL (Windows Subsystem for Linux) 运行指南

为了利用 RTX 5070 的 Blackwell 架构特性，建议在 WSL 环境下运行。

### 5.1 环境配置

由于 WSL 中可能缺少预装的 Python 环境，建议安装基础依赖：

```bash
# 获取管理员权限并安装 pip (如果尚未安装)
sudo apt-get update && sudo apt-get install -y python3-pip

# 安装项目依赖 (PyTorch for CUDA 11.8+, 兼容 5070)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install pytorch-lightning timm einops h5py pandas matplotlib scikit-image
```

### 5.2 运行训练

代码已自动适配 WSL 路径 (`/mnt/c/...`)。

```bash
# 检查依赖
python3 check_deps.py

# 启动训练
python3 run_sevir_multiGpu_distributedSampler.py
```

### 5.3 网络问题排查 (重要)

如果遇到 `Connection refused` 或下载卡住：

1.  **检查本机代理软件 (Clash/v2ray)**:
    *   确保开启了 **"Allow LAN" (允许局域网连接)** 功能。
    *   记录显示的端口号 (通常是 7890)。

2.  **获取主机 IP**:
    在 WSL 终端运行：
    ```bash
    cat /etc/resolv.conf | grep nameserver
    # 输出示例: nameserver 10.255.255.254
    ```

3.  **配置并重试**:
    ```bash
    export hostip=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
    export http_proxy="http://${hostip}:7890"
    export https_proxy="http://${hostip}:7890"
    
    # 手动安装
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```
