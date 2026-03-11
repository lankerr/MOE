"""
MoE 训练脚本 — 基于 nanoGPT train.py，使用 model_moe.py
支持所有 MoE 实验变体，自动记录负载均衡统计。

用法:
  python train_moe.py config/exp2_vanilla_moe.py
  python train_moe.py config/exp3_full_moe.py
  python train_moe.py config/exp4a_relu2.py
"""

import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_moe import GPTConfig, GPT, AUX_MANAGER

# ─── 默认配置 ────────────────────────────────────────────
# I/O
out_dir = 'out'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
# wandb
wandb_log = False
wandb_project = 'moe-history24'
wandb_run_name = 'run'
# data
dataset = 'history24_char'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512
# model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False
# MoE
n_exp = 1
moe_top_k = 2
moe_start_layer = 2
moe_stride = 1
use_aux_loss = False
aux_loss_weight = 0.01
use_router_z_loss = False
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
activation = 'gelu'
# optimizer
learning_rate = 1e-3
max_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 3000
min_lr = 1e-4
# system
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# ─────────────────────────────────────────────────────────

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model init
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"vocab_size = {meta_vocab_size} (from {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    # MoE args
    n_exp=n_exp, moe_top_k=moe_top_k, moe_start_layer=moe_start_layer,
    moe_stride=moe_stride, use_aux_loss=use_aux_loss, aux_loss_weight=aux_loss_weight,
    use_router_z_loss=use_router_z_loss, router_z_loss_weight=router_z_loss_weight,
    use_noisy_top_k=use_noisy_top_k, train_capacity=train_capacity,
    eval_capacity=eval_capacity, activation=activation,
)

if init_from == 'scratch':
    print("Initializing new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'n_exp', 'moe_top_k', 'moe_start_layer', 'moe_stride', 'activation']:
        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# ─── 评估函数 ─────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ─── LR 调度 ─────────────────────────────────────────────
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ─── 训练日志 ─────────────────────────────────────────────
training_log = {
    'config': config,
    'model_args': {k: v for k, v in model_args.items() if v is not None},
    'evals': [],
    'load_balance': [],
}

# ─── 训练循环 ─────────────────────────────────────────────
X, Y = get_batch('train')
t0 = time.time()
train_start = time.time()
local_iter_num = 0
running_mfu = -1.0
val_loss_history = []

print(f"\n{'='*65}")
print(f"  开始训练: {out_dir}")
print(f"  实验: n_exp={n_exp}, top_k={moe_top_k}, act={activation}")
print(f"  aux_loss={use_aux_loss}, z_loss={use_router_z_loss}")
print(f"  max_iters={max_iters}, batch={batch_size}, block={block_size}")
print(f"{'='*65}\n")

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ─── 评估 ─────────────────────────────────────────
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        overfit_gap = val_loss - train_loss
        elapsed = time.time() - train_start
        eta = (max_iters - iter_num) * (elapsed / max(iter_num, 1)) / 60
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if 'cuda' in device else 0

        val_improved = val_loss < best_val_loss
        val_loss_history.append(val_loss.item())

        # 专家负载统计
        expert_info = ""
        if n_exp > 1:
            expert_stats = raw_model.get_expert_stats()
            if expert_stats:
                # 取第一个MoE层展示
                first_layer = list(expert_stats.keys())[0]
                stats = expert_stats[first_layer]
                pcts = stats['pct']
                balance = stats['balance']
                expert_info = f"  负载均衡 ({first_layer}): {balance:.3f} (1.0=完美)"
                bar_parts = []
                for j, p in enumerate(pcts):
                    bar_parts.append(f"E{j}:{p:.1f}%")
                expert_info += f"\n  专家分配: {' | '.join(bar_parts)}"
                training_log['load_balance'].append({
                    'step': iter_num, 'stats': expert_stats
                })
            raw_model.reset_expert_stats()

        print(f"\n{'='*65}")
        print(f"  Step {iter_num}/{max_iters} | {elapsed/60:.1f}min | ETA {eta:.1f}min")
        print(f"  Train Loss: {train_loss:.4f} (PPL={train_ppl:.1f})")
        print(f"  Val   Loss: {val_loss:.4f} (PPL={val_ppl:.1f}) {'✓ BEST' if val_improved else ''}")
        print(f"  Overfit Gap: {overfit_gap:+.4f} {'⚠' if overfit_gap > 0.3 else '✓'}")
        print(f"  LR: {lr:.6f} | GPU: {gpu_mem:.2f}GB")
        if expert_info:
            print(expert_info)
        print(f"{'='*65}\n")

        training_log['evals'].append({
            'step': iter_num, 'train_loss': train_loss.item(),
            'val_loss': val_loss.item(), 'train_ppl': train_ppl, 'val_ppl': val_ppl,
        })

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

        # 保存训练日志
        with open(os.path.join(out_dir, 'training_log.json'), 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)

    if iter_num == 0 and eval_only:
        break

    # ─── 前向/反向 ─────────────────────────────────────
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

# ─── 训练完成，保存最终日志 ────────────────────────────────
if master_process:
    total_time = (time.time() - train_start) / 60
    training_log['total_time_min'] = total_time
    training_log['final_val_loss'] = best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss
    with open(os.path.join(out_dir, 'training_log.json'), 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n训练完成! 用时 {total_time:.1f}min, best val_loss={best_val_loss:.4f}")

if ddp:
    destroy_process_group()
