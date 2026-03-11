"""
二十四史 GPT 模型综合评估脚本
============================================================
评估维度:
  1. 验证集困惑度 (Val PPL)  — 语言建模能力的金标准
  2. 生成文本困惑度 (Gen PPL) — 模型对自己生成内容的确信度
  3. 重复率 (Repetition)      — n-gram重复程度，越低越好
  4. Distinct-n               — 词汇多样性，越高越好
  5. 与原文的分布相似度       — 字频分布是否接近真实语料
  6. 平均句长 & 标点率        — 生成文本的结构合理性

用法:
  python evaluate_model.py --out_dir=out-history24-char --num_samples=10
  python evaluate_model.py --out_dir=out-history24-bpe  --num_samples=10
"""

import os
import sys
import math
import pickle
import numpy as np
from collections import Counter
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# ─── 默认参数 ────────────────────────────────────────────────
init_from = 'resume'
out_dir = 'out-history24-char'
num_samples = 10           # 生成样本数（越多越稳定）
max_new_tokens = 500       # 每个样本生成长度
temperature = 0.8
top_k = 200
seed = 42
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# prompts 用于生成评估样本
prompts = ["黄帝", "太宗皇帝", "诸葛亮", "曹操", "刘备", "司马懿", "李世民", "赵匡胤", "朱元璋", "岳飞"]
exec(open('configurator.py').read())
# ─────────────────────────────────────────────────────────────

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ═══════════════════════════════════════════════════════════
#  加载模型与分词器
# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("  二十四史 GPT 综合评估")
print("=" * 65)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"错误: 找不到 checkpoint {ckpt_path}")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
block_size = gptconf.block_size

# 分词器
dataset_name = checkpoint['config']['dataset']
meta_path = os.path.join('data', dataset_name, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

is_bpe = meta.get('tokenizer') == 'sentencepiece'
if is_bpe:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(meta['model_path'])
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda l: sp.decode(l)
    tokenizer_name = f"SentencePiece BPE (vocab={meta['vocab_size']})"
else:
    stoi, itos = meta['stoi'], meta['itos']
    UNK_ID = stoi.get('<UNK>', 0)
    encode = lambda s: [stoi.get(c, UNK_ID) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    tokenizer_name = f"字符级 (vocab={meta['vocab_size']})"

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  模型: {out_dir} ({n_params:.1f}M params)")
print(f"  分词器: {tokenizer_name}")
print(f"  训练步数: {checkpoint.get('iter_num', '?')}")
print(f"  训练最佳val_loss: {checkpoint.get('best_val_loss', '?'):.4f}")
print()


# ═══════════════════════════════════════════════════════════
#  指标1: 验证集困惑度 (最重要的指标)
# ═══════════════════════════════════════════════════════════
print("─" * 65)
print("  [1/6] 验证集困惑度 (Val Perplexity)")
print("─" * 65)

val_data_path = os.path.join('data', dataset_name, 'val.bin')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
val_tokens = len(val_data)

# 在验证集上计算PPL（多个随机块取平均）
eval_iters = 100
total_loss = 0.0
for i in range(eval_iters):
    ix = torch.randint(len(val_data) - block_size, (32,))
    x_val = torch.stack([torch.from_numpy((val_data[j:j+block_size]).astype(np.int64)) for j in ix])
    y_val = torch.stack([torch.from_numpy((val_data[j+1:j+1+block_size]).astype(np.int64)) for j in ix])
    x_val, y_val = x_val.to(device), y_val.to(device)
    with torch.no_grad():
        with ctx:
            logits, loss = model(x_val, y_val)
    total_loss += loss.item()

val_loss = total_loss / eval_iters
val_ppl = math.exp(val_loss)
print(f"  Val Loss:  {val_loss:.4f}")
print(f"  Val PPL:   {val_ppl:.2f}")
print(f"  (在 {val_tokens:,} 个验证token上, {eval_iters}次随机采样)")
print()


# ═══════════════════════════════════════════════════════════
#  指标2~6: 生成质量评估
# ═══════════════════════════════════════════════════════════
print("─" * 65)
print(f"  [2-6] 生成质量评估 ({num_samples} 个样本)")
print("─" * 65)

generated_texts = []
gen_ppls = []

with torch.no_grad():
    with ctx:
        for i in range(min(num_samples, len(prompts))):
            prompt = prompts[i % len(prompts)]
            start_ids = encode(prompt)
            x_gen = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y_gen = model.generate(x_gen, max_new_tokens, temperature=temperature, top_k=top_k)
            gen_ids = y_gen[0].tolist()
            text = decode(gen_ids)
            generated_texts.append(text)

            # 生成文本的perplexity
            seq = torch.tensor(gen_ids, dtype=torch.long, device=device).unsqueeze(0)
            seq_len = seq.size(1)
            chunk_nll = 0.0
            chunk_cnt = 0
            for start_pos in range(0, seq_len - 1, block_size):
                end_pos = min(start_pos + block_size, seq_len - 1)
                inp = seq[:, start_pos:end_pos]
                tgt = seq[:, start_pos + 1:end_pos + 1]
                logits, _ = model(inp, tgt)
                nll = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction='sum')
                chunk_nll += nll.item()
                chunk_cnt += tgt.size(1)
            avg_nll = chunk_nll / chunk_cnt if chunk_cnt > 0 else float('nan')
            gen_ppls.append(math.exp(avg_nll) if avg_nll < 100 else float('inf'))

            print(f"  样本 {i+1}: prompt='{prompt}' → {len(text)}字, PPL={gen_ppls[-1]:.2f}")

# 合并所有生成的文本
all_text = ''.join(generated_texts)
all_chars = list(all_text)
total_gen_chars = len(all_chars)

# ─── 指标2: 生成PPL汇总 ──────────────────────────────────
print()
print(f"  [2] 生成PPL (Self-Perplexity)")
print(f"      平均: {sum(gen_ppls)/len(gen_ppls):.2f}")
print(f"      最低: {min(gen_ppls):.2f}  最高: {max(gen_ppls):.2f}")
print(f"      (衡量模型对自己生成内容的确信度，越低=越确信)")


# ─── 指标3: 重复率 (Repetition Rate) ────────────────────────
print()
print(f"  [3] 重复率 (Repetition Rate)")

def repetition_rate(text, n):
    """计算 n-gram 的重复率: 1 - (unique n-grams / total n-grams)"""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    unique = len(set(ngrams))
    total = len(ngrams)
    return 1.0 - unique / total

for n in [2, 3, 4, 5, 8]:
    rep = repetition_rate(all_text, n)
    bar = "█" * int(rep * 30) + "░" * (30 - int(rep * 30))
    print(f"      {n}-gram: {rep:.4f}  {bar}")
print(f"      (越低=越少重复, 理想值: 2-gram<0.90, 4-gram<0.70)")


# ─── 指标4: Distinct-n (多样性) ──────────────────────────────
print()
print(f"  [4] Distinct-n (词汇多样性)")

def distinct_n(text, n):
    """Distinct-n: unique n-grams / total n-grams"""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

for n in [1, 2, 3, 4]:
    d = distinct_n(all_text, n)
    bar = "█" * int(d * 30) + "░" * (30 - int(d * 30))
    print(f"      Distinct-{n}: {d:.4f}  {bar}")
print(f"      (越高=词汇越丰富, 理想值: D-1>0.05, D-2>0.20)")


# ─── 指标5: 与原文分布相似度 ─────────────────────────────────
print()
print(f"  [5] 字频分布相似度 (与原始语料对比)")

# 加载原始语料的一小段做参考
ref_texts = []
corpus_dir = os.path.join('..', '二十四史txt')
if os.path.exists(corpus_dir):
    for fn in sorted(os.listdir(corpus_dir))[:3]:  # 取前3本做参考
        fp = os.path.join(corpus_dir, fn)
        try:
            with open(fp, 'r', encoding='gb18030', errors='ignore') as f:
                ref_texts.append(f.read()[:200000])  # 每本取前20万字
        except:
            pass

if ref_texts:
    ref_text = ''.join(ref_texts)

    # 字频分布的Jensen-Shannon散度
    gen_counter = Counter(all_text)
    ref_counter = Counter(ref_text)

    # 取并集的字符
    all_vocab = set(gen_counter.keys()) | set(ref_counter.keys())
    gen_total = sum(gen_counter.values())
    ref_total = sum(ref_counter.values())

    gen_dist = np.array([gen_counter.get(c, 0) / gen_total for c in all_vocab])
    ref_dist = np.array([ref_counter.get(c, 0) / ref_total for c in all_vocab])

    # JS散度 (对称版KL散度, 0=完全相同, ln2≈0.693=完全不同)
    m = 0.5 * (gen_dist + ref_dist)
    # 避免log(0)
    eps = 1e-12
    kl_pm = np.sum(gen_dist * np.log((gen_dist + eps) / (m + eps)))
    kl_qm = np.sum(ref_dist * np.log((ref_dist + eps) / (m + eps)))
    js_div = 0.5 * (kl_pm + kl_qm)
    js_similarity = 1.0 - js_div / np.log(2)  # 归一化到 [0, 1]

    # 字符覆盖率
    gen_unique = set(all_text)
    ref_unique = set(ref_text)
    coverage = len(gen_unique & ref_unique) / len(ref_unique) if ref_unique else 0

    print(f"      JS相似度: {js_similarity:.4f}  (1.0=完美匹配, >0.85=良好)")
    print(f"      字符覆盖率: {coverage:.4f}  (生成文本覆盖了参考语料 {coverage*100:.1f}% 的字符)")
    print(f"      生成独特字: {len(gen_unique)} / 参考独特字: {len(ref_unique)}")
else:
    print(f"      [跳过] 未找到参考语料 ({corpus_dir})")


# ─── 指标6: 结构合理性 ───────────────────────────────────────
print()
print(f"  [6] 文本结构合理性")

# 标点符号集
punctuation = set('，。；：！？、""''（）《》…——\n')
punct_count = sum(1 for c in all_text if c in punctuation)
punct_rate = punct_count / total_gen_chars if total_gen_chars > 0 else 0

# 句子分割（按句号、问号、感叹号）
import re
sentences = re.split(r'[。！？]', all_text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0
std_sent_len = np.std([len(s) for s in sentences]) if sentences else 0

# 段落（按换行）
paragraphs = [p.strip() for p in all_text.split('\n') if len(p.strip()) > 0]

print(f"      总生成字数: {total_gen_chars}")
print(f"      标点率: {punct_rate:.4f}  ({punct_rate*100:.1f}%)")
print(f"      平均句长: {avg_sent_len:.1f} 字 (σ={std_sent_len:.1f})")
print(f"      句子数: {len(sentences)}")
print(f"      段落数: {len(paragraphs)}")
print(f"      (参考: 古文标点率约5-8%, 平均句长10-30字)")


# ═══════════════════════════════════════════════════════════
#  综合评分
# ═══════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  综合评估报告")
print("=" * 65)

# 打分 (每项0-100, 权重加总)
scores = {}

# PPL评分: val_ppl < 10: 100分, < 30: 80, < 60: 65, < 100: 50, < 200: 35, else 20
if val_ppl < 10:
    scores['val_ppl'] = 100
elif val_ppl < 30:
    scores['val_ppl'] = 80 + 20 * (30 - val_ppl) / 20
elif val_ppl < 60:
    scores['val_ppl'] = 65 + 15 * (60 - val_ppl) / 30
elif val_ppl < 100:
    scores['val_ppl'] = 50 + 15 * (100 - val_ppl) / 40
elif val_ppl < 200:
    scores['val_ppl'] = 35 + 15 * (200 - val_ppl) / 100
else:
    scores['val_ppl'] = max(20, 35 - (val_ppl - 200) / 50)

# 重复率评分 (4-gram): < 0.5: 100, < 0.7: 75, < 0.85: 50, else 30
rep4 = repetition_rate(all_text, 4)
if rep4 < 0.5:
    scores['repetition'] = 100
elif rep4 < 0.7:
    scores['repetition'] = 75 + 25 * (0.7 - rep4) / 0.2
elif rep4 < 0.85:
    scores['repetition'] = 50 + 25 * (0.85 - rep4) / 0.15
else:
    scores['repetition'] = max(10, 50 * (1.0 - rep4) / 0.15)

# Distinct-2评分: > 0.5: 100, > 0.3: 80, > 0.15: 60, > 0.05: 40, else 20
d2 = distinct_n(all_text, 2)
if d2 > 0.5:
    scores['diversity'] = 100
elif d2 > 0.3:
    scores['diversity'] = 80 + 20 * (d2 - 0.3) / 0.2
elif d2 > 0.15:
    scores['diversity'] = 60 + 20 * (d2 - 0.15) / 0.15
elif d2 > 0.05:
    scores['diversity'] = 40 + 20 * (d2 - 0.05) / 0.1
else:
    scores['diversity'] = max(10, 40 * d2 / 0.05)

# 分布相似度评分
if ref_texts:
    scores['distribution'] = min(100, js_similarity * 110)
else:
    scores['distribution'] = 50  # 无参考时给中间分

# 结构评分: 标点率5-8%最佳, 句长10-30最佳
punct_score = 100 - min(100, abs(punct_rate - 0.065) / 0.065 * 60)
sent_score = 100 - min(100, abs(avg_sent_len - 20) / 20 * 50)
scores['structure'] = max(10, (punct_score + sent_score) / 2)

# 权重
weights = {
    'val_ppl': 0.35,        # 验证PPL最重要
    'repetition': 0.20,     # 重复问题很扎眼
    'diversity': 0.15,      # 多样性
    'distribution': 0.15,   # 与原文接近度
    'structure': 0.15,      # 结构合理
}

total_score = sum(scores[k] * weights[k] for k in weights)

labels = {
    'val_ppl': '验证集困惑度',
    'repetition': '低重复率',
    'diversity': '词汇多样性',
    'distribution': '分布相似度',
    'structure': '结构合理性',
}

for k in weights:
    w_pct = int(weights[k] * 100)
    bar = "█" * int(scores[k] / 100 * 20) + "░" * (20 - int(scores[k] / 100 * 20))
    print(f"  {labels[k]:>10s} ({w_pct}%): {scores[k]:5.1f}/100  {bar}")

print(f"\n  {'总分':>10s}:       {total_score:.1f}/100")

# 等级判定
if total_score >= 85:
    grade = "A  (优秀 — 可产出高质量文本)"
elif total_score >= 70:
    grade = "B  (良好 — 基本连贯，偶有重复)"
elif total_score >= 55:
    grade = "C  (一般 — 能模仿风格，但有明显缺陷)"
elif total_score >= 40:
    grade = "D  (较差 — 需要更多训练或调参)"
else:
    grade = "E  (很差 — 模型尚未学到有意义的模式)"

print(f"  {'等级':>10s}:       {grade}")
print()
print("  改进建议:")
worst = min(scores, key=scores.get)
if worst == 'val_ppl':
    print("  → PPL偏高: 增加训练步数、增大模型、或尝试更低学习率")
elif worst == 'repetition':
    print("  → 重复率高: 增大dropout、降低temperature、或加入repetition penalty")
elif worst == 'diversity':
    print("  → 多样性低: 提高temperature、扩大top_k、或增加训练数据")
elif worst == 'distribution':
    print("  → 与原文偏差大: 模型可能过拟合或欠拟合，检查训练数据处理")
elif worst == 'structure':
    print("  → 结构不理想: 检查分词器对标点的处理，或增加训练步数")

print("=" * 65)
