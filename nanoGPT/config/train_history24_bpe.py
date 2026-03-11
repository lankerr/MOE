# 二十四史 SentencePiece BPE GPT训练配置
# 25M参数模型，适配 RTX 5070 8GB VRAM

out_dir = 'out-history24-bpe'
eval_interval = 500
eval_iters = 100
log_interval = 10

# 只在验证loss改善时保存
always_save_checkpoint = False

wandb_log = False
wandb_project = 'history24-bpe'
wandb_run_name = 'bpe-25M'

dataset = 'history24_bpe'
gradient_accumulation_steps = 4  # 有效batch = 4 * 32 = 128
batch_size = 32
block_size = 512  # 上下文512 tokens

# ~25M 参数模型（vocab更大，embedding层更大，稍微减层保持总参数相近）
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 6e-4  # BPE模型用稍低的lr
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 200

bias = False

# 系统设置
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Windows上Triton不可用
