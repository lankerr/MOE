"""
显存基准测试: 逐步升配置，找出 RTX 3050Ti / RTX 5070 的极限
"""
import torch
import numpy as np
import gc
import sys
sys.path.insert(0, '.')
from run_pretrain_radar import VideoMAEPretrainModel

device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {gpu_name} ({gpu_mem:.2f} GB)")
print()

header = f"{'Config':<46s} | {'Params':>7s} | {'Tokens':>10s} | {'VRAM':>10s} | Fits?"
print(header)
print("-" * len(header))

configs = [
    # 既然显存这么宽裕，直接测大配置 + 大batch
    # --- ViT-Small 极限推 ---
    ("Small 384x384  8f  b=2  mask90%",   384,  8, 384, 12, 6, 192, 4,  2, 0.9),
    ("Small 384x384  8f  b=4  mask90%",   384,  8, 384, 12, 6, 192, 4,  4, 0.9),
    ("Small 384x384 16f  b=1  mask90%",   384, 16, 384, 12, 6, 192, 4,  1, 0.9),
    ("Small 384x384 16f  b=2  mask90%",   384, 16, 384, 12, 6, 192, 4,  2, 0.9),
    ("Small 384x384 16f  b=4  mask90%",   384, 16, 384, 12, 6, 192, 4,  4, 0.9),
    ("Small 384x384 20f  b=2  mask90%",   384, 20, 384, 12, 6, 192, 4,  2, 0.9),
    # --- ViT-Base 极限推 ---
    ("Base  192x192 16f  b=2  mask90%",   192, 16, 768, 12, 12, 384, 4, 2, 0.9),
    ("Base  192x192 16f  b=4  mask90%",   192, 16, 768, 12, 12, 384, 4, 4, 0.9),
    ("Base  256x256  8f  b=2  mask90%",   256,  8, 768, 12, 12, 384, 4, 2, 0.9),
    ("Base  256x256 16f  b=2  mask90%",   256, 16, 768, 12, 12, 384, 4, 2, 0.9),
    ("Base  384x384  8f  b=1  mask90%",   384,  8, 768, 12, 12, 384, 4, 1, 0.9),
    ("Base  384x384  8f  b=2  mask90%",   384,  8, 768, 12, 12, 384, 4, 2, 0.9),
    ("Base  384x384 16f  b=1  mask90%",   384, 16, 768, 12, 12, 384, 4, 1, 0.9),
    ("Base  384x384 16f  b=2  mask90%",   384, 16, 768, 12, 12, 384, 4, 2, 0.9),
    # --- ViT-Base 80% mask (更多visible) ---
    ("Base  384x384 16f  b=1  mask80%",   384, 16, 768, 12, 12, 384, 4, 1, 0.8),
    ("Base  384x384 16f  b=2  mask80%",   384, 16, 768, 12, 12, 384, 4, 2, 0.8),
]

results = []

for name, isz, nf, edim, dep, heads, ddim, dd, bs, mr in configs:
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    n_sp = (isz // 16) ** 2
    n_t = nf // 2
    n_total = n_sp * n_t
    n_vis = int(n_total * (1 - mr))
    
    try:
        model = VideoMAEPretrainModel(
            img_size=isz, patch_size=16, tubelet_size=2, in_chans=1,
            encoder_embed_dim=edim, encoder_depth=dep, encoder_num_heads=heads,
            decoder_embed_dim=ddim, decoder_depth=dd, decoder_num_heads=heads,
        ).to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[X] {name:<44s} |       | CREATE OOM")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        raise
    
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    model.train()
    oom = False
    
    for step in range(3):
        frames = torch.rand(bs, 1, nf, isz, isz, device=device)
        sp_mask = np.zeros(n_sp, dtype=bool)
        sp_mask[np.random.choice(n_sp, int(n_sp * mr), replace=False)] = True
        mask_np = np.tile(sp_mask, n_t)
        mask = torch.from_numpy(mask_np).unsqueeze(0).expand(bs, -1).contiguous().to(device)
        
        opt.zero_grad()
        try:
            with torch.amp.autocast('cuda'):
                loss = model(frames, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom = True
                torch.cuda.empty_cache()
                break
            raise
        del frames, mask
    
    if oom:
        peak_gb = 999
        peak_str = "OOM!"
        fit = "3050 X   5070 ?"
    else:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_str = f"{peak_gb:.2f} GB"
        if peak_gb < 3.5:
            fit = "3050 OK  5070 OK"
        elif peak_gb < 8.0:
            fit = "3050 X   5070 OK"
        else:
            fit = "3050 X   5070 X"
    
    tag = "v" if not oom else "X"
    line = f"[{tag}] {name:<44s} | {total_p:5.1f}M | {n_total:4d}->{n_vis:3d} | {peak_str:>10s} | {fit}"
    print(line)
    results.append((name, total_p, n_total, n_vis, peak_gb, fit))
    
    del model, opt, scaler
    torch.cuda.empty_cache()
    gc.collect()

print()
print("=" * 80)
print("总结: RTX 5070 (8.55GB) 可用配置:")
print("=" * 80)
for name, params, ntot, nvis, peak, fit in results:
    if peak < 8.0:
        print(f"  OK  {name}  |  {peak:.2f} GB  |  {params:.1f}M params")
print()
print("推荐给 RTX 5070 的最大配置 = 上表中 peak 最接近 8GB 但不超过的那个")
