"""
论文下载脚本 - 自动下载顶刊论文 PDF

使用方法:
    python download_papers.py --all
    python download_papers.py --category lr_scheduler
    python download_papers.py --paper wsd

依赖:
    pip install requests beautifulsoup4 arxiv
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


# 论文配置
PAPERS = {
    # 学习率调度
    'wsd': {
        'title': 'Universal Dynamics of Warmup Stable Decay',
        'url': 'https://arxiv.org/abs/2401.11079',
        'pdf_url': 'https://arxiv.org/pdf/2401.11079.pdf',
        'category': 'lr_scheduler',
        'year': 2024,
        'venue': 'arxiv',
    },
    'adaptive_lr': {
        'title': 'WHEN, WHY AND HOW MUCH? Learning Rate Schedules',
        'url': 'https://openreview.net/forum?id=xxxxx',
        'category': 'lr_scheduler',
        'year': 2024,
        'venue': 'OpenReview',
    },

    # Token 高效处理
    'tome': {
        'title': 'Token Merging for Fast Vision Processing',
        'url': 'https://arxiv.org/abs/2209.15559',
        'pdf_url': 'https://arxiv.org/pdf/2209.15559.pdf',
        'category': 'token_efficient',
        'year': 2023,
        'venue': 'ICLR',
    },
    'evit': {
        'title': 'Efficient Vision Transformer with Token Pruning',
        'url': 'https://arxiv.org/abs/2204.08616',
        'pdf_url': 'https://arxiv.org/pdf/2204.08616.pdf',
        'category': 'token_efficient',
        'year': 2022,
        'venue': 'ICLR',
    },
    'dynamicvit': {
        'title': 'DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsity',
        'url': 'https://arxiv.org/abs/2106.01304',
        'pdf_url': 'https://arxiv.org/pdf/2106.01304.pdf',
        'category': 'token_efficient',
        'year': 2021,
        'venue': 'NeurIPS',
    },
    'adaptive_merge': {
        'title': 'Adaptive Token Merging for Efficient Transformers',
        'url': 'https://arxiv.org/abs/2409.09955',
        'pdf_url': 'https://arxiv.org/pdf/2409.09955.pdf',
        'category': 'token_efficient',
        'year': 2024,
        'venue': 'arxiv',
    },
    'tram': {
        'title': 'Token Reduction via Attention-based Multilayer Network',
        'url': 'https://www.sciencedirect.com/science/article/pii/S0957417425010711',
        'category': 'token_efficient',
        'year': 2025,
        'venue': 'Expert Systems with Applications',
    },

    # 气象 AI
    'earthformer': {
        'title': 'EarthFormer: Exploring Space-Time Transformers for Earth System Forecasting',
        'url': 'https://arxiv.org/abs/2207.05833',
        'pdf_url': 'https://arxiv.org/pdf/2207.05833.pdf',
        'category': 'weather_ai',
        'year': 2022,
        'venue': 'NeurIPS',
    },
    'nowcastnet': {
        'title': 'NowcastNet: A Deep Learning Approach for Precipitation Nowcasting',
        'url': 'https://www.nature.com/articles/s41586-023-06184-4',
        'category': 'weather_ai',
        'year': 2023,
        'venue': 'Nature',
    },
    'prediff': {
        'title': 'PreDiff: Precipitation Nowcasting with Latent Diffusion Models',
        'url': 'https://arxiv.org/abs/2309.15025',
        'pdf_url': 'https://arxiv.org/pdf/2309.15025.pdf',
        'category': 'weather_ai',
        'year': 2023,
        'venue': 'NeurIPS',
    },
    'pangu': {
        'title': 'Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks',
        'url': 'https://www.nature.com/articles/s41586-023-06184-4',
        'category': 'weather_ai',
        'year': 2023,
        'venue': 'Nature',
    },

    # 等变网络
    'e2cnn': {
        'title': 'Equivariant CNNs for the Rotation Group',
        'url': 'https://arxiv.org/abs/1802.08219',
        'pdf_url': 'https://arxiv.org/pdf/1802.08219.pdf',
        'category': 'equivariant',
        'year': 2018,
        'venue': 'NeurIPS',
    },
    'steerable_cnn': {
        'title': 'Steerable CNNs for Rotation Equivariance',
        'url': 'https://arxiv.org/abs/1804.08258',
        'pdf_url': 'https://arxiv.org/pdf/1804.08258.pdf',
        'category': 'equivariant',
        'year': 2018,
        'venue': 'ICLR',
    },
}


def download_pdf(url: str, output_path: Path) -> bool:
    """下载 PDF 文件"""
    try:
        print(f"  下载: {url}")
        print(f"  保存: {output_path}")

        # 使用 requests 下载
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # 保存文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  ✓ 完成 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def create_paper_markdown(papers_dir: Path):
    """创建论文索引 Markdown 文件"""
    index_path = papers_dir / "README.md"

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# 核心论文索引\n\n")
        f.write("## 论文分类\n\n")

        # 按类别分组
        categories = {}
        for key, paper in PAPERS.items():
            cat = paper['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((key, paper))

        for cat, papers in categories.items():
            cat_name = {
                'lr_scheduler': '学习率调度',
                'token_efficient': 'Token 高效处理',
                'weather_ai': '气象 AI',
                'equivariant': '等变网络',
            }.get(cat, cat)

            f.write(f"### {cat_name}\n\n")

            for key, paper in papers:
                pdf_file = papers_dir / cat / f"{key}.pdf"
                status = "✓" if pdf_file.exists() else "✗"

                f.write(f"- [{status}] **{paper['title']}** ")
                f.write(f"({paper['venue']} {paper['year']})\n")
                f.write(f"  - Arxiv: {paper['url']}\n")
                if pdf_file.exists():
                    f.write(f"  - PDF: `{pdf_file}`\n")
                f.write("\n")

    print(f"\n索引文件已创建: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='下载顶刊论文')
    parser.add_argument('--all', action='store_true', help='下载所有论文')
    parser.add_argument('--category', type=str, choices=set(p['category'] for p in PAPERS.values()),
                       help='按类别下载')
    parser.add_argument('--paper', type=str, choices=list(PAPERS.keys()),
                       help='下载特定论文')
    parser.add_argument('--output', type=str, default='papers',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    papers_dir = Path(args.output)

    # 确定要下载的论文
    papers_to_download = []

    if args.all:
        papers_to_download = list(PAPERS.items())
    elif args.category:
        papers_to_download = [(k, v) for k, v in PAPERS.items() if v['category'] == args.category]
    elif args.paper:
        papers_to_download = [(args.paper, PAPERS[args.paper])]
    else:
        print("请指定 --all, --category 或 --paper")
        return

    # 下载论文
    print(f"准备下载 {len(papers_to_download)} 篇论文...\n")

    success_count = 0
    for key, paper in papers_to_download:
        print(f"\n[{key.upper()}] {paper['title']}")
        print(f"  来源: {paper['venue']} {paper['year']}")

        if 'pdf_url' not in paper:
            print(f"  ⚠ 暂无直接 PDF 链接")
            continue

        output_path = papers_dir / paper['category'] / f"{key}.pdf"

        if output_path.exists():
            print(f"  ✓ 已存在")
            success_count += 1
            continue

        if download_pdf(paper['pdf_url'], output_path):
            success_count += 1

    # 创建索引
    create_paper_markdown(papers_dir)

    # 总结
    print(f"\n{'='*60}")
    print(f"下载完成: {success_count}/{len(papers_to_download)}")
    print(f"论文目录: {papers_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
