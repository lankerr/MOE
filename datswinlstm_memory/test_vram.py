"""显存峰值对比测试"""
import torch
import argparse
import gc

device = torch.device("cuda")
print("=" * 60)
print("显存峰值对比测试 - 寻找 8GB 显卡最大可用配置")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

from models.DATSwinLSTM_D_Memory import Memory

def test_config(name, img_size, embed_dim, depths_down, depths_up, heads, window_size, out_len, short_len, long_len, memory_ch):
    print(f"【{name}】{img_size}x{img_size}, embed_dim={embed_dim}")
    model_args = argparse.Namespace(
        input_img_size=img_size, patch_size=4, input_channels=1,
        embed_dim=embed_dim, depths_down=depths_down, depths_up=depths_up,
        heads_number=heads, window_size=window_size, out_len=out_len
    )
    try:
        model = Memory(model_args, memory_channel_size=memory_ch, short_len=short_len, long_len=long_len).to(device)
        x = torch.randn(1, short_len, 1, img_size, img_size).to(device)
        memory_x = torch.randn(1, long_len, 1, img_size, img_size).to(device)
        
        torch.cuda.reset_peak_memory_stats()
        y = model(x, memory_x, phase=2)
        if isinstance(y, list): y = torch.stack(y)
        loss = y.mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {params:,}")
        print(f"  训练峰值: {peak:.2f} GB ✅")
        
        del model, x, memory_x, y, loss
        torch.cuda.empty_cache()
        gc.collect()
        return peak
    except RuntimeError as e:
        torch.cuda.empty_cache()
        gc.collect()
        if "out of memory" in str(e):
            print(f"  ❌ OOM!")
        else:
            print(f"  错误: {e}")
        return None
    print()

# 测试各种配置
test_config("轻量 A", 128, 64, [2,2], [2,2], [4,4], 4, 12, 8, 24, 256)
print()
test_config("中等 B", 192, 64, [2,2], [2,2], [4,4], 4, 12, 8, 24, 256)
print()
test_config("中等 C", 256, 64, [2,2], [2,2], [4,4], 4, 12, 8, 24, 256)
print()
test_config("大型 D", 320, 64, [2,2], [2,2], [4,4], 4, 12, 8, 24, 256)
print()
test_config("大型 E", 384, 64, [2,2], [2,2], [4,4], 4, 12, 8, 24, 256)

print("=" * 60)
