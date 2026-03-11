"""
实验结果汇总对比脚本
读取所有实验的 training_log.json 和 expert_analysis.json
生成对比表格
"""

import os
import json
import math

experiments = [
    ('out-exp1-dense',       'Exp1: Dense 基线'),
    ('out-exp2-vanilla-moe', 'Exp2: Vanilla MoE'),
    ('out-exp3-full-moe',    'Exp3: Full MoE'),
    ('out-exp4a-relu2',      'Exp4a: MoE+ReLU²'),
    ('out-exp4b-swiglu',     'Exp4b: MoE+SwiGLU'),
]

print("=" * 80)
print("  MoE 实验结果对比")
print("=" * 80)

results = []
for out_dir, name in experiments:
    log_path = os.path.join(out_dir, 'training_log.json')
    if not os.path.exists(log_path):
        print(f"  {name}: 未找到 {log_path}, 跳过")
        continue

    with open(log_path, 'r', encoding='utf-8') as f:
        log = json.load(f)

    val_loss = log.get('final_val_loss', None)
    if val_loss is None and log.get('evals'):
        val_loss = min(e['val_loss'] for e in log['evals'])

    val_ppl = math.exp(min(val_loss, 20)) if val_loss else None
    time_min = log.get('total_time_min', None)

    cfg = log.get('config', {})
    n_exp = cfg.get('n_exp', 1)
    act = cfg.get('activation', 'gelu')
    aux = cfg.get('use_aux_loss', False)

    results.append({
        'name': name,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'time_min': time_min,
        'n_exp': n_exp,
        'activation': act,
        'aux_loss': aux,
    })

if not results:
    print("  还没有实验完成，请先运行训练!")
    exit()

# 排序: 按 val_ppl 从低到高
results.sort(key=lambda x: x['val_ppl'] if x['val_ppl'] else 999)

print(f"\n{'排名':<4} {'实验':<24} {'Val Loss':<10} {'Val PPL':<10} {'用时(min)':<10} {'专家':<6} {'激活':<8} {'辅助损失':<8}")
print("-" * 80)
for i, r in enumerate(results):
    rank = f"#{i+1}"
    vl = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
    vp = f"{r['val_ppl']:.2f}" if r['val_ppl'] else "N/A"
    tm = f"{r['time_min']:.1f}" if r['time_min'] else "N/A"
    aux = "是" if r['aux_loss'] else "否"
    print(f"{rank:<4} {r['name']:<24} {vl:<10} {vp:<10} {tm:<10} {r['n_exp']:<6} {r['activation']:<8} {aux:<8}")

# 最佳
best = results[0]
print(f"\n🏆 最佳: {best['name']} (Val PPL={best['val_ppl']:.2f})")

# 与 Dense 对比
dense = next((r for r in results if 'Dense' in r['name']), None)
if dense:
    print(f"\n相对 Dense 基线 (PPL={dense['val_ppl']:.2f}):")
    for r in results:
        if 'Dense' not in r['name'] and r['val_ppl']:
            diff = r['val_ppl'] - dense['val_ppl']
            pct = diff / dense['val_ppl'] * 100
            arrow = "↑" if diff > 0 else "↓"
            print(f"  {r['name']}: PPL {arrow}{abs(diff):.2f} ({pct:+.1f}%)")

# 负载均衡
print(f"\n{'='*80}")
print("  负载均衡对比 (来自 expert_analysis.json)")
print(f"{'='*80}")
for out_dir, name in experiments:
    ea_path = os.path.join(out_dir, 'expert_analysis.json')
    if not os.path.exists(ea_path):
        continue
    with open(ea_path, 'r', encoding='utf-8') as f:
        ea = json.load(f)
    lb = ea.get('load_balance', {})
    if lb:
        avg_bal = sum(v['balance'] for v in lb.values()) / len(lb)
        print(f"  {name}: 平均均衡度={avg_bal:.4f}")

print(f"\n{'='*80}")
