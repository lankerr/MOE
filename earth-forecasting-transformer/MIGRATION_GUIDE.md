# Earthformer Huawei Ascend (NPU) Migration Guide

此指南帮助你将代码从 NVIDIA GPU 环境迁移到华为升腾 (Ascend NPU) 智算平台。

## 1. 平台环境准备 (在智算平台上操作)

1.  登录 **SCOW 智算平台**。
2.  点击 **作业 -> 应用**，创建 **VSCode** 应用。
3.  **镜像选择**：
    *   请务必选择华为官方支持 PyTorch 的镜像：
    *   `app-store-images.pku.edu.cn/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10` (根据教程推荐)
4.  **资源申请**：
    *   选择 **单节点**，加速卡数量根据需求选择 (例如 1 张 910B)。
5.  启动并进入 VSCode 界面。

## 2. 代码与数据上传

为了方便你上传，老公已经帮你把核心代码打包好了！

1.  **上传代码包**：
    *   在本地电脑找到文件：`C:\Users\97290\Desktop\datswinlstm_memory\earth-forecasting-transformer\earthformer_code.zip`
    *   在平台的 **VSCode** 界面中，直接将这个 `zip` 文件拖进去。
    *   在 VSCode 的终端 (Terminal) 里运行解压命令：
        ```bash
        unzip earthformer_code.zip -d earth-forecasting-transformer
        ```

2.  **确认数据位置**：
    *   平台通常有共享存储或者会有教程（如 Tutorial 5）教你挂载数据。请确认 SEVIR 数据集在平台上的路径。
    *   如果需要重新下载，Ascend 环境通常也支持 AWS CLI (需要配置)。

## 3. 环境依赖安装 (在智算平台 Terminal 中)

进入项目目录并安装依赖：

```bash
cd ~/earth-forecasting-transformer
# 升级 pip
pip install --upgrade pip

# 安装项目依赖 (不包括 torch, 因为镜像自带了 NPU 版 torch)
pip install -r requirements.txt
```

> **注意**：不要运行 `pip install torch`！镜像里已经预装了适配 NPU 的 `torch` 和 `torch_npu`。如果覆盖安装会导致无法使用 NPU。

## 4. 代码适配修改

为了在 NPU 上运行，需要对代码做少量修改。

### 4.1 修改 `scripts/cuboid_transformer/sevir/train_cuboid_sevir.py`

在文件开头添加 `torch_npu` 导入，并禁用 CUDA 特有功能。

```python
# [NEW] 在 import torch 下面增加
import torch
import torch_npu  # 必须导入这个才能识别 NPU
from torch_npu.contrib import transfer_to_npu # 可选，自动转换
```

在 `main()` 函数中，**删除或注释掉** NVIDIA 特有的设置：

```python
def main():
    # [DELETE] 注释掉这行，NPU 不支持 cudnn benchmark
    # torch.backends.cudnn.benchmark = True 
    
    # [MODIFY] 确保 matmul 精度设置兼容 (可选)
    # torch.set_float32_matmul_precision('medium') 
```

### 4.2 修改 `scripts/cuboid_transformer/sevir/cfg_sevir.yaml`

修改 `trainer` 部分，显式指定加速器为 NPU。

```yaml
trainer:
    # [MODIFY] 将 gpu 改为 npu，或者 auto (取决于 PyTorch Lightning 版本)
    accelerator: "npu" 
    devices: 1
    # [KEEP] 混合精度在 NPU 上支持良好
    precision: 16 
```

> **提示**：如果你的 `pytorch_lightning` 版本较老不支持 `accelerator="npu"`，请尝试 `accelerator="auto"`，或者查阅对应版本的文档。通常新版 PL 会自动识别。

## 5. 运行训练

使用以下命令运行 (路径根据实际情况调整)：

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 scripts/cuboid_transformer/sevir/train_cuboid_sevir.py --gpus 1 --cfg scripts/cuboid_transformer/sevir/cfg_sevir.yaml
```

**常见问题排查**：
*   **HCCL Error**: 通常是多卡通信问题，单卡训练一般不会遇到。
*   **Device Error**: 确保导入了 `import torch_npu`。
