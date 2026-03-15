import urllib.request
from pathlib import Path

base_dir = Path('.')
downloads = [
    ('https://arxiv.org/pdf/2104.00954.pdf', 'nature/dgmr/DGMR_Nature2021.pdf'),
    ('https://arxiv.org/pdf/2211.02556.pdf', 'nature/pangu_weather/PanguWeather_Nature2023.pdf'),
    ('https://arxiv.org/pdf/2212.12794.pdf', 'nature/graphcast/GraphCast_Nature2023.pdf'),
    ('https://arxiv.org/pdf/2202.11214.pdf', 'nature/fourcastnet/FourCastNet_Nature2022.pdf'),
    ('https://arxiv.org/pdf/2207.05833.pdf', 'top_conferences/transformer/EarthFormer_NeurIPS2022.pdf'),
    ('https://arxiv.org/pdf/2209.15559.pdf', 'top_conferences/token_efficient/ToMe_ICLR2023.pdf'),
    ('https://arxiv.org/pdf/1802.08219.pdf', 'top_conferences/equivariant/E2CNN_NeurIPS2018.pdf'),
    ('https://arxiv.org/pdf/2103.14030.pdf', 'top_conferences/transformer/SwinTransformer_ICLR2022.pdf'),
    ('https://arxiv.org/pdf/2204.08616.pdf', 'top_conferences/token_efficient/EViT_ICLR2022.pdf'),
    ('https://arxiv.org/pdf/2106.01304.pdf', 'top_conferences/token_efficient/DynamicViT_NeurIPS2021.pdf'),
    ('https://arxiv.org/pdf/2010.11929.pdf', 'top_conferences/transformer/ViT_ICLR2021.pdf'),
    ('https://arxiv.org/pdf/1804.08258.pdf', 'top_conferences/equivariant/SteerableCNN_ICLR2018.pdf'),
]

success = []
failed = []

for url, path in downloads:
    try:
        output_path = base_dir / path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading: {path}')
        urllib.request.urlretrieve(url, output_path)
        size = output_path.stat().st_size / 1024 / 1024
        print(f'  Success ({size:.2f} MB)')
        success.append(path)
    except Exception as e:
        print(f'  Failed: {e}')
        failed.append((path, str(e)))

print()
print('=== Download Report ===')
print(f'Success: {len(success)}/{len(success)+len(failed)}')
if failed:
    print('Failed (use school account):')
    for path, err in failed:
        print(f'  {path}')
