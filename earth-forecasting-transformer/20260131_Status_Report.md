# 项目状态全记录报告 (2026-01-31)

**核心目标**: 在搭载 **RTX 5070 (Blackwell架构)** 显卡的笔记本上运行 Earthformer 雷达预测模型。

---

## 📅 背景与核心问题 (Why WSL?)
我们之所以大费周章切换到 WSL2 环境，是因为遇到了以下不可逾越的硬件兼容性问题：

1.  **硬件太新**: RTX 5070 采用最新的 Blackwell 架构 (Compute Capability sm_120)。
2.  **Windows版 PyTorch 落后**: Windows 平台上的 PyTorch 稳定版 (CUDA 12.1/12.4) **不支持** sm_120 架构。
    - *表现*: 报错 `RuntimeError: CUDA error: no kernel image is available for execution on the device`。
3.  **解决方案**: 转战 **Linux (WSL2)** 生态。
    - Linux 平台拥有最新的开发版支持。
    - 我们成功在 WSL2 中获取了 **PyTorch Nightly + CUDA 12.8**，它是目前唯一能驱动 RTX 5070 进行计算的版本。

---

## ✅ 已完成的里程碑 (Milestones)

### 1. WSL2 环境完美构建
- **系统**: Ubuntu 24.04 (代号: `extrapolation`)。
- **网络修复**: 解决了最棘手的 VPN/代理连接问题。
    - 配置: `%UserProfile%\.wslconfig` 开启了 `networkingMode=mirrored`，让 WSL 共享 Windows 的网络环境。
- **文件挂载**: 实现了**无缝数据访问**。
    - **不下载**: 100GB 的 SEVIR 数据集保留在 Windows 桌面 (`C:\Users\97290\Desktop\datasets\sevir`)。
    - **直接读**: 通过配置代码直接读取 `/mnt/c/Users/97290/Desktop/datasets`，节省了大量磁盘空间。

### 2. 显卡计算验证成功 (Major Success)
- **PyTorch 版本**: `2.7.0.dev + cu128` (核心关键点)。
- **验证结果**: 运行 `simple_gpu_test.py` 成功。
    - 模型加载 ✅
    - Tensor 移入显卡 ✅
    - 前向传播 (Forward Pass) ✅
    - **结论**: RTX 5070 在此环境下可用！

---

## 🚧 当前进度与接下来的挑战

### 1. 代码环境配置 (进行中)
目前卡在 Python 库的版本依赖上：
- **Earthformer 代码**: 基于旧版 PyTorch Lightning (1.x) 开发。
- **当前安装**: 安装了最新版 PyTorch Lightning (2.6.0)。
- **报错**: `ModuleNotFoundError: No module named 'pytorch_lightning.utilities.cloud_io'`。
- **修复方案**: 需要降级 PyTorch Lightning 到 `1.9.5` (已验证该版本兼容代码且不影响新版 PyTorch)。

### 2. 配置文件修复
- `config.py` 需要使用 `yacs` 库构建配置节点，以支持代码中的导入方式。我们已经准备好修复脚本。

---

## 🛠️ 环境信息速查 (Context for Next Session)

如果你需要重新开始或让 AI 接手，请直接把下面这段话发给它：

> "环境是 WSL2 (Ubuntu-24.04)，用户 'extrapolation'。
> 显卡是 RTX 5070，必须使用 PyTorch Nightly (cu128) 才能运行。
> 项目代码在 `~/earth-forecasting-transformer`。
> 数据集在 Windows 桌面，通过 `/mnt/c/Users/97290/Desktop/datasets` 访问。
> 目前已验证 GPU 可用，但需要将 'pytorch-lightning' 降级到 1.x 版本以适配 Earthformer 代码。"

---

**文件位置**: `C:\Users\97290\Desktop\datswinlstm_memory\earth-forecasting-transformer\20260131_Status_Report.md`
