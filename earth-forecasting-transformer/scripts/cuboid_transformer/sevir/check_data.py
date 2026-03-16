import numpy as np
import glob

data_dir = r'C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple'
files = sorted(glob.glob(data_dir + '/day_simple_*.npy'))

if files:
    print(f'[Chongqing Data] Found {len(files)} files')
    print(f'[Chongqing Data] First file: {files[0]}')

    data = np.load(files[0], mmap_mode='r')
    print(f'[Chongqing Data] Shape: {data.shape}')
    print(f'[Chongqing Data] Dtype: {data.dtype}')
    print(f'[Chongqing Data] Range: [{data.min():.4f}, {data.max():.4f}]')
    print(f'[Chongqing Data] Mean: {data.mean():.4f}')
    print(f'[Chongqing Data] Image size: {data.shape[1]} x {data.shape[2]}')
    print(f'[Chongqing Data] Frames: {data.shape[0]}')

    # Check first 5 files
    print(f'[Chongqing Data] First 5 files:')
    for f in files[:5]:
        d = np.load(f, mmap_mode='r')
        print(f'  {f}: {d.shape}')
