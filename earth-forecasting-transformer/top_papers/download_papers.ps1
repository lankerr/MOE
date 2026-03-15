"""
顶刊论文自动下载脚本 (Windows PowerShell)

使用方法:
    1. 右键点击此文件 → "使用 PowerShell 运行"
    2. 或在 PowerShell 中运行: .\download_papers.ps1
"""

# 设置错误处理
$ErrorActionPreference = "Stop"

# 创建目录结构
$baseDir = "top_papers"
$dirs = @(
    "$baseDir\nature\dgmr",
    "$baseDir\nature\nowcastnet",
    "$baseDir\nature\pangu_weather",
    "$baseDir\nature\graphcast",
    "$baseDir\nature\fourcastnet",
    "$baseDir\science\graphcast",
    "$baseDir\top_conferences\transformer",
    "$baseDir\top_conferences\token_efficient",
    "$baseDir\top_conferences\equivariant"
)

Write-Host "=== 创建目录结构 ===" -ForegroundColor Green
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  创建: $dir" -ForegroundColor Gray
    }
}

# 论文下载列表
$papers = @(
    # Nature 级别
    @{ Name = "DGMR"; Url = "https://arxiv.org/pdf/2104.00954.pdf"; Path = "$baseDir\nature\dgmr\DGMR_Nature2021.pdf" },
    @{ Name = "NowcastNet"; Url = "https://www.nature.com/articles/s41586-023-06184-4.pdf"; Path = "$baseDir\nature\nowcastnet\NowcastNet_Nature2023.pdf" },
    @{ Name = "Pangu-Weather"; Url = "https://arxiv.org/pdf/2211.02556.pdf"; Path = "$baseDir\nature\pangu_weather\PanguWeather_Nature2023.pdf" },
    @{ Name = "GraphCast"; Url = "https://arxiv.org/pdf/2212.12794.pdf"; Path = "$baseDir\nature\graphcast\GraphCast_Nature2023.pdf" },
    @{ Name = "FourCastNet"; Url = "https://arxiv.org/pdf/2202.11214.pdf"; Path = "$baseDir\nature\fourcastnet\FourCastNet_Nature2022.pdf" },

    # Science 级别
    @{ Name = "GraphCast (Science)"; Url = "https://www.science.org/doi/pdf/10.1126/science.adi2336"; Path = "$baseDir\science\graphcast\GraphCast_Science2023.pdf" },

    # Transformer 架构
    @{ Name = "EarthFormer"; Url = "https://arxiv.org/pdf/2207.05833.pdf"; Path = "$baseDir\top_conferences\transformer\EarthFormer_NeurIPS2022.pdf" },
    @{ Name = "Swin Transformer"; Url = "https://arxiv.org/pdf/2103.14030.pdf"; Path = "$baseDir\top_conferences\transformer\SwinTransformer_ICLR2022.pdf" },
    @{ Name = "ViT"; Url = "https://arxiv.org/pdf/2010.11929.pdf"; Path = "$baseDir\top_conferences\transformer\ViT_ICLR2021.pdf" },

    # Token 高效
    @{ Name = "ToMe"; Url = "https://arxiv.org/pdf/2209.15559.pdf"; Path = "$baseDir\top_conferences\token_efficient\ToMe_ICLR2023.pdf" },
    @{ Name = "EViT"; Url = "https://arxiv.org/pdf/2204.08616.pdf"; Path = "$baseDir\top_conferences\token_efficient\EViT_ICLR2022.pdf" },
    @{ Name = "DynamicViT"; Url = "https://arxiv.org/pdf/2106.01304.pdf"; Path = "$baseDir\top_conferences\token_efficient\DynamicViT_NeurIPS2021.pdf" },

    # 等变网络
    @{ Name = "E2CNN"; Url = "https://arxiv.org/pdf/1802.08219.pdf"; Path = "$baseDir\top_conferences\equivariant\E2CNN_NeurIPS2018.pdf" },
    @{ Name = "Steerable CNN"; Url = "https://arxiv.org/pdf/1804.08258.pdf"; Path = "$baseDir\top_conferences\equivariant\SteerableCNN_ICLR2018.pdf" }
)

Write-Host "`n=== 开始下载论文 ===" -ForegroundColor Green
Write-Host "共 $($papers.Count) 篇论文`n"

$successCount = 0
$failCount = 0

foreach ($paper in $papers) {
    # 检查是否已存在
    if (Test-Path $paper.Path) {
        Write-Host "[$($paper.Name)] " -ForegroundColor Cyan -NoNewline
        Write-Host "已存在" -ForegroundColor Gray
        $successCount++
        continue
    }

    try {
        Write-Host "[$($paper.Name)] " -ForegroundColor Cyan -NoNewline
        Write-Host "下载中..." -ForegroundColor Yellow -NoNewline

        # 下载文件
        Invoke-WebRequest -Uri $paper.Url -OutFile $paper.Path -TimeoutSec 60

        # 获取文件大小
        $size = (Get-Item $paper.Path).Length / 1MB
        Write-Host "`r[$($paper.Name)] " -ForegroundColor Cyan -NoNewline
        Write-Host "完成 ($('{0:.2f}' -f $size) MB)" -ForegroundColor Green

        $successCount++
    }
    catch {
        Write-Host "`r[$($paper.Name)] " -ForegroundColor Cyan -NoNewline
        Write-Host "失败: $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
}

# 总结
Write-Host "`n=== 下载完成 ===" -ForegroundColor Green
Write-Host "成功: $successCount / $($papers.Count)" -ForegroundColor Green
if ($failCount -gt 0) {
    Write-Host "失败: $failCount" -ForegroundColor Red
}

Write-Host "`n论文目录: $((Get-Location).Path)\$baseDir" -ForegroundColor Gray
Write-Host "查看 README.md 获取更多信息" -ForegroundColor Gray

# 暂停以查看结果
Write-Host "`n按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
