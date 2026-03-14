# ============================================================
# VideoMAE ViT-Tiny 环境搭建脚本
# 在 MOE 根目录运行: .\setup_videomae.ps1
# ============================================================

Write-Host "=" * 60
Write-Host "VideoMAE ViT-Tiny 环境搭建"
Write-Host "=" * 60

# 1. 克隆 VideoMAE
if (-not (Test-Path "VideoMAE")) {
    Write-Host "`n[1/4] 克隆 VideoMAE 仓库..."
    git clone https://github.com/MCG-NJU/VideoMAE.git
} else {
    Write-Host "`n[1/4] VideoMAE 已存在, 跳过克隆"
}

# 2. 安装依赖 (使用 conda 环境 rtx3050ti_cu128)
Write-Host "`n[2/4] 安装依赖..."
Write-Host "请确保已激活 conda 环境: conda activate rtx3050ti_cu128"
& C:\Users\Lenovo\.conda\envs\rtx3050ti_cu128\python.exe -m pip install timm==0.4.12 einops tensorboardX h5py --quiet

# 3. 准备数据目录
$dataDir = "X:\datasets\sevir"
if (Test-Path $dataDir) {
    Write-Host "`n[3/4] SEVIR 数据目录已存在: $dataDir"
} else {
    Write-Host "`n[3/4] 警告: SEVIR 数据目录不存在: $dataDir"
    Write-Host "      请确保 X:\datasets\sevir 下有 CATALOG.csv 和 data/ 目录"
}

# 4. 验证
Write-Host "`n[4/4] 验证安装..."
& C:\Users\Lenovo\.conda\envs\rtx3050ti_cu128\python.exe -c @"
import torch, timm, einops, h5py
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"})')
print(f'显存:     {torch.cuda.get_device_properties(0).total_mem / 1024**3:.2f} GB' if torch.cuda.is_available() else '')
print(f'timm:     {timm.__version__}')
print(f'einops:   {einops.__version__}')
print(f'h5py:     {h5py.__version__}')
print('所有依赖安装成功!')
"@

Write-Host "`n" + "=" * 60
Write-Host "搭建完成! 下一步:"
Write-Host "  cd VideoMAE"
Write-Host "  conda activate rtx3050ti_cu128"
Write-Host "  python run_pretrain_radar.py --debug   # 先CPU调试"
Write-Host "  python run_pretrain_radar.py            # GPU训练"
Write-Host "=" * 60
