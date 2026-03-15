"""
简化的论文下载报告和环境检查
"""
import sys
from pathlib import Path

print("=" * 60)
print("顶刊论文下载报告")
print("=" * 60)

# 检查已下载的论文
base_dir = Path('.')
papers = [
    'nature/dgmr/DGMR_Nature2021.pdf',
    'nature/pangu_weather/PanguWeather_Nature2023.pdf',
    'nature/graphcast/GraphCast_Nature2023.pdf',
    'nature/fourcastnet/FourCastNet_Nature2022.pdf',
    'top_conferences/transformer/EarthFormer_NeurIPS2022.pdf',
    'top_conferences/token_efficient/ToMe_ICLR2023.pdf',
    'top_conferences/equivariant/E2CNN_NeurIPS2018.pdf',
    'top_conferences/transformer/SwinTransformer_ICLR2022.pdf',
    'top_conferences/token_efficient/EViT_ICLR2022.pdf',
    'top_conferences/token_efficient/DynamicViT_NeurIPS2021.pdf',
    'top_conferences/transformer/ViT_ICLR2021.pdf',
    'top_conferences/equivariant/SteerableCNN_ICLR2018.pdf',
]

# 未下载的论文（需要学校账号）
pending = [
    ('nature/nowcastnet/NowcastNet_Nature2023.pdf', 'https://www.nature.com/articles/s41586-023-06184-4.pdf'),
    ('science/graphcast/GraphCast_Science2023.pdf', 'https://www.science.org/doi/pdf/10.1126/science.adi2336'),
]

success_count = 0
for paper in papers:
    path = base_dir / paper
    if path.exists():
        size = path.stat().st_size / 1024 / 1024
        print(f"[OK] {paper} ({size:.2f} MB)")
        success_count += 1
    else:
        print(f"[X] {paper} (not found)")

print()
print(f"已下载: {success_count}/{len(papers)}")
print()

if pending:
    print("需要学校账号下载:")
    for path, url in pending:
        print(f"  - {path}")
        print(f"    URL: {url}")
    print()

print("=" * 60)
