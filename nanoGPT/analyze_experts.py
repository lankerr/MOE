"""
专家分析脚本 — 负载均衡、专家消融、专家特化分析
用法:
  python analyze_experts.py --ckpt out-exp3-full-moe/ckpt.pt
"""

import os
import sys
import math
import json
import argparse
import pickle
from collections import defaultdict, Counter

import numpy as np
import torch
from model_moe import GPTConfig, GPT, AUX_MANAGER

# ─── 参数 ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
parser.add_argument('--dataset', type=str, default='history24_char')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=512)
args = parser.parse_args()

device = args.device
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# ─── 加载模型 ────────────────────────────────────────────
print(f"加载 checkpoint: {args.ckpt}")
checkpoint = torch.load(args.ckpt, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

n_exp = model.config.n_exp
out_dir = os.path.dirname(args.ckpt)
print(f"模型: n_exp={n_exp}, top_k={model.config.moe_top_k}, act={model.config.activation}")
print(f"总参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ─── 数据加载 ────────────────────────────────────────────
data_dir = os.path.join('data', args.dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta.get('stoi', {})
itos = meta.get('itos', {})

val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(val_data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((val_data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# ─── 1. 基线 PPL ─────────────────────────────────────────
@torch.no_grad()
def compute_ppl():
    losses = []
    for _ in range(args.eval_iters):
        X, Y = get_batch()
        with ctx:
            _, loss = model(X, Y)
        losses.append(loss.item())
    avg = np.mean(losses)
    return math.exp(min(avg, 20)), avg

print("\n" + "="*60)
print("  1. 基线困惑度")
print("="*60)
base_ppl, base_loss = compute_ppl()
print(f"  Val Loss: {base_loss:.4f}, Val PPL: {base_ppl:.2f}")

# ─── 2. 负载均衡分析 ─────────────────────────────────────
print("\n" + "="*60)
print("  2. 负载均衡分析")
print("="*60)

model.reset_expert_stats()
# 跑数据收集统计
for _ in range(args.eval_iters):
    X, Y = get_batch()
    with ctx:
        model(X, Y)

expert_stats = model.get_expert_stats()
balance_report = {}
for layer_name, stats in expert_stats.items():
    pcts = stats['pct']
    balance = stats['balance']
    print(f"\n  {layer_name}:")
    print(f"    均衡度: {balance:.4f} (1.0=完美均匀)")
    bar = ""
    for j, p in enumerate(pcts):
        bar_len = int(p / 100 * n_exp * 20)
        bar += f"    E{j}: {'█' * bar_len}{'░' * (20 - bar_len)} {p:.1f}%\n"
    print(bar, end='')
    balance_report[layer_name] = {
        'pcts': pcts.tolist() if hasattr(pcts, 'tolist') else list(pcts),
        'balance': float(balance),
    }

# ─── 3. 专家消融实验 ─────────────────────────────────────
print("\n" + "="*60)
print("  3. 专家消融实验 (逐个移除专家，观察PPL变化)")
print("="*60)

if n_exp <= 1:
    print("  非 MoE 模型，跳过消融实验")
    ablation_results = {}
else:
    ablation_results = {}
    # 找所有MoE层
    moe_layers = []
    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'moe') and block.moe is not None:
            moe_layers.append(i)

    print(f"  MoE 层: {moe_layers}")
    print(f"  基线 PPL: {base_ppl:.2f}\n")

    # 对每层的每个专家做消融
    for layer_idx in moe_layers:
        layer_results = []
        for exp_idx in range(n_exp):
            # 消融
            model.ablate_expert(layer_idx, exp_idx)
            abl_ppl, abl_loss = compute_ppl()
            # 恢复 (重新加载权重)
            block_sd = {k: v for k, v in checkpoint['model'].items() if f'h.{layer_idx}.' in k}
            cleaned = {}
            for k, v in block_sd.items():
                nk = k.replace('_orig_mod.', '')
                cleaned[nk] = v
            model.load_state_dict(cleaned, strict=False)

            delta = abl_ppl - base_ppl
            pct = delta / base_ppl * 100
            importance = "⚠ 关键" if pct > 5 else ("→ 重要" if pct > 2 else "  一般")
            print(f"  Layer {layer_idx} Expert {exp_idx}: PPL={abl_ppl:.2f} (Δ={delta:+.2f}, {pct:+.1f}%) {importance}")
            layer_results.append({
                'expert': exp_idx, 'ppl': abl_ppl, 'delta': delta, 'pct_change': pct
            })
        ablation_results[f'layer_{layer_idx}'] = layer_results
        # 找最重要和最不重要的
        sorted_r = sorted(layer_results, key=lambda x: x['delta'], reverse=True)
        print(f"  → Layer {layer_idx} 最关键专家: E{sorted_r[0]['expert']} (Δ={sorted_r[0]['delta']:+.2f})")
        print(f"  → Layer {layer_idx} 最冗余专家: E{sorted_r[-1]['expert']} (Δ={sorted_r[-1]['delta']:+.2f})")
        print()

# ─── 4. 专家特化分析 ─────────────────────────────────────
print("\n" + "="*60)
print("  4. 专家特化分析 (不同主题文本激活哪些专家)")
print("="*60)

# 从语料中提取一些主题性强的片段
theme_prompts = {
    '战争': '大军出征，将士奋勇杀敌，攻城掠地，斩首万级',
    '政治': '天子御殿，群臣朝拜，宰相奏事，诏曰',
    '人物': '字子长，少好学，年十岁则诵古文',
    '天文': '日有食之，星辰移动，天象示警，彗星见于东方',
    '地理': '其地东临大海，西接沙漠，南至岭南，北抵长城',
}

if n_exp <= 1:
    print("  非 MoE 模型，跳过特化分析")
    specialization = {}
else:
    specialization = {}
    for theme, prompt in theme_prompts.items():
        # 编码
        ids = [stoi.get(c, 0) for c in prompt]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:] + [0]], dtype=torch.long, device=device)

        model.reset_expert_stats()
        with torch.no_grad():
            with ctx:
                model(x, y)

        stats = model.get_expert_stats()
        print(f"\n  【{theme}】: \"{prompt[:20]}...\"")
        theme_stats = {}
        for layer_name, s in stats.items():
            top_exp = np.argsort(s['pct'])[::-1][:3]
            top_str = ", ".join([f"E{e}({s['pct'][e]:.1f}%)" for e in top_exp])
            print(f"    {layer_name}: {top_str}")
            theme_stats[layer_name] = s['pct'].tolist() if hasattr(s['pct'], 'tolist') else list(s['pct'])
        specialization[theme] = theme_stats

# ─── 5. 保存报告 ─────────────────────────────────────────
report = {
    'checkpoint': args.ckpt,
    'n_exp': n_exp,
    'moe_top_k': model.config.moe_top_k,
    'activation': model.config.activation,
    'base_ppl': base_ppl,
    'base_loss': base_loss,
    'load_balance': balance_report,
    'ablation': ablation_results,
    'specialization': specialization,
}

report_path = os.path.join(out_dir, 'expert_analysis.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(f"\n分析报告已保存: {report_path}")

# ─── 6. 总览 ─────────────────────────────────────────────
print("\n" + "="*60)
print("  总览")
print("="*60)
print(f"  基线 PPL: {base_ppl:.2f}")
if balance_report:
    avg_balance = np.mean([v['balance'] for v in balance_report.values()])
    print(f"  平均负载均衡度: {avg_balance:.4f}")
if ablation_results:
    all_deltas = [r['delta'] for layer in ablation_results.values() for r in layer]
    print(f"  消融 PPL 变化范围: [{min(all_deltas):+.2f}, {max(all_deltas):+.2f}]")
print("="*60)
