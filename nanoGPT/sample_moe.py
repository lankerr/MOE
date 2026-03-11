"""
MoE 模型采样 + 质量评估脚本
用法:
  python sample_moe.py --ckpt out-exp2-vanilla-moe/ckpt.pt
  python sample_moe.py --ckpt out-exp3-full-moe/ckpt.pt --start "太祖高皇帝"
"""

import os
import sys
import math
import pickle
import argparse
from collections import Counter

import numpy as np
import torch
from model_moe import GPTConfig, GPT, AUX_MANAGER

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--dataset', type=str, default='history24_char')
parser.add_argument('--start', type=str, default='诸葛亮字孔明')
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--max_new_tokens', type=int, default=400)
parser.add_argument('--temperature', type=float, default=0.75)
parser.add_argument('--top_k', type=int, default=200)
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

device = args.device
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# ─── 加载模型 ────────────────────────────────────────────
print(f"加载: {args.ckpt}")
checkpoint = torch.load(args.ckpt, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"模型: {n_params:.1f}M 参数, n_exp={model.config.n_exp}, act={model.config.activation}")

# ─── 加载词表 ────────────────────────────────────────────
data_dir = os.path.join('data', args.dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

# ─── 1. Val PPL ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  1. 验证集困惑度 (Val PPL)")
print(f"{'='*60}")

val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
block_size = model.config.block_size
batch_size = 32

losses = []
with torch.no_grad():
    for _ in range(args.eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())

val_loss = np.mean(losses)
val_ppl = math.exp(min(val_loss, 20))
print(f"  Val Loss: {val_loss:.4f}")
print(f"  Val PPL:  {val_ppl:.2f}")

# ─── 2. 生成样本 ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  2. 文本生成 (start=\"{args.start}\")")
print(f"{'='*60}")

start_ids = encode(args.start)
x = torch.tensor([start_ids], dtype=torch.long, device=device)

all_texts = []
for i in range(args.num_samples):
    with torch.no_grad():
        with ctx:
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    text = decode(y[0].tolist())
    all_texts.append(text)
    print(f"\n--- 样本 {i+1} ---")
    print(text)

# ─── 3. 生成质量指标 ─────────────────────────────────────
print(f"\n{'='*60}")
print(f"  3. 生成质量指标")
print(f"{'='*60}")

for i, text in enumerate(all_texts):
    # 重复率
    chars = list(text)
    bigrams = [text[j:j+2] for j in range(len(text)-1)]
    trigrams = [text[j:j+3] for j in range(len(text)-2)]
    fourgrams = [text[j:j+4] for j in range(len(text)-3)]

    def repeat_rate(ngrams):
        if not ngrams:
            return 0
        c = Counter(ngrams)
        return sum(v - 1 for v in c.values()) / len(ngrams)

    def distinct_n(ngrams):
        if not ngrams:
            return 0
        return len(set(ngrams)) / len(ngrams)

    r2 = repeat_rate(bigrams)
    r4 = repeat_rate(fourgrams)
    d1 = distinct_n(chars)
    d2 = distinct_n(bigrams)

    # 标点统计
    puncts = set('，。、；：？！""''（）《》—…')
    punct_rate = sum(1 for c in text if c in puncts) / max(len(text), 1)

    print(f"\n  样本 {i+1} ({len(text)}字):")
    print(f"    2-gram重复率: {r2:.3f} | 4-gram重复率: {r4:.3f}")
    print(f"    Distinct-1: {d1:.3f} | Distinct-2: {d2:.3f}")
    print(f"    标点率: {punct_rate:.1%}")

# ─── 4. 专家负载 (仅MoE) ─────────────────────────────────
if model.config.n_exp > 1:
    print(f"\n{'='*60}")
    print(f"  4. 专家负载分布")
    print(f"{'='*60}")

    model.reset_expert_stats()
    # 跑一些数据
    with torch.no_grad():
        for _ in range(20):
            ix = torch.randint(len(val_data) - block_size, (batch_size,))
            xb = torch.stack([torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix])
            yb = torch.stack([torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
            xb, yb = xb.to(device), yb.to(device)
            with ctx:
                model(xb, yb)

    stats = model.get_expert_stats()
    for layer_name, s in stats.items():
        pcts = s['pct']
        balance = s['balance']
        print(f"\n  {layer_name}: 均衡度={balance:.4f}")
        for j, p in enumerate(pcts):
            bar_len = int(p / 100 * model.config.n_exp * 20)
            bar = '█' * bar_len + '░' * max(0, 20 - bar_len)
            print(f"    E{j}: {bar} {p:.1f}%")

# ─── 5. 生成PPL ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  5. 生成文本自身困惑度")
print(f"{'='*60}")

for i, text in enumerate(all_texts):
    ids = encode(text)
    if len(ids) < 2:
        continue
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with ctx:
            _, loss = model(seq[:, :-1], seq[:, 1:])
    gen_ppl = math.exp(min(loss.item(), 20))
    print(f"  样本 {i+1}: PPL={gen_ppl:.2f}")

print(f"\n{'='*60}")
print(f"  完成!")
print(f"{'='*60}")
