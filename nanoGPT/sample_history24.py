"""
二十四史模型采样脚本 - 支持字符级和SentencePiece BPE两种分词器
带 Perplexity 计算：衡量模型对自己生成文本的困惑度
用法:
  python sample_history24.py --out_dir=out-history24-char --start="黄帝"
  python sample_history24.py --out_dir=out-history24-bpe --start="黄帝"
"""
import os
import math
import pickle
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out-history24-char'
start = "黄帝"
num_samples = 3
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 加载模型
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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

# 加载分词器
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

if meta.get('tokenizer') == 'sentencepiece':
    # SentencePiece BPE 分词器
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(meta['model_path'])
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda l: sp.decode(l)
    tokenizer_name = f"SentencePiece BPE (vocab={meta['vocab_size']})"
else:
    # 字符级分词器
    stoi, itos = meta['stoi'], meta['itos']
    UNK_ID = stoi.get('<UNK>', 0)
    encode = lambda s: [stoi.get(c, UNK_ID) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    tokenizer_name = f"字符级 (vocab={meta['vocab_size']})"

print(f"分词器: {tokenizer_name}")


def compute_perplexity(token_ids, model, block_size, device, ctx):
    """
    计算给定 token 序列的 perplexity。
    input[i] -> 预测 target[i] = token_ids[i+1]
    对长序列按 block_size 分块，取平均 NLL。
    """
    seq = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = seq.size(1)
    if seq_len < 2:
        return float('nan'), float('nan'), 0

    total_nll = 0.0
    total_tokens = 0

    # 分块处理: input = seq[i:i+block_size], target = seq[i+1:i+block_size+1]
    for i in range(0, seq_len - 1, block_size):
        end = min(i + block_size, seq_len - 1)  # end for input
        input_chunk = seq[:, i:end]          # (1, chunk_len)
        target_chunk = seq[:, i + 1:end + 1]  # (1, chunk_len) — 每个位置的下一个token

        with torch.no_grad():
            with ctx:
                # 传入 targets 使模型输出所有位置的 logits（而非仅最后一个）
                logits, _ = model(input_chunk, target_chunk)
                nll = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_chunk.view(-1),
                    reduction='sum'
                )

        total_nll += nll.item()
        total_tokens += target_chunk.size(1)

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('nan')
    ppl = math.exp(avg_nll) if avg_nll < 100 else float('inf')
    return ppl, avg_nll, total_tokens


# 编码起始文本
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(f"\n起始文本: '{start}'")
print(f"编码为 {len(start_ids)} tokens")
print(f"生成 {num_samples} 个样本，每个 {max_new_tokens} tokens")
print(f"temperature={temperature}, top_k={top_k}")
print("=" * 60)

# 生成 + 计算perplexity
all_ppls = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_ids = y[0].tolist()
            text = decode(generated_ids)

            # 计算整个序列(prompt+generated)的 perplexity
            ppl, avg_nll, n_tokens = compute_perplexity(
                generated_ids, model, block_size, device, ctx
            )
            all_ppls.append(ppl)

            print(f"\n{'─' * 50}")
            print(f"  样本 {k+1}  │  PPL={ppl:.2f}  │  NLL={avg_nll:.4f}  │  {n_tokens} tokens")
            print(f"{'─' * 50}")
            print(text)
            print()

# 汇总统计
print("=" * 60)
print(f"  汇总: {num_samples} 个样本")
print(f"  平均 Perplexity: {sum(all_ppls)/len(all_ppls):.2f}")
print(f"  最低 PPL: {min(all_ppls):.2f}  │  最高 PPL: {max(all_ppls):.2f}")
print(f"  (PPL 越低 = 模型对自己生成内容越确信)")
print("=" * 60)
