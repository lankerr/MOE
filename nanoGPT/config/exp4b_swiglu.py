# 实验4b: SwiGLU 激活函数 (Full MoE + SwiGLU)
# Qwen3/Alibaba 风格，带 gate_proj 的门控激活

out_dir = 'out-exp4b-swiglu'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'moe-history24'
wandb_run_name = 'exp4b-swiglu'

dataset = 'history24_char'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

# 模型结构
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# MoE: Full + SwiGLU
n_exp = 8
moe_top_k = 2
moe_start_layer = 2
moe_stride = 1
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = True
train_capacity = 1.25
eval_capacity = 2.0
activation = 'swiglu'

# 训练
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

# 系统
device = 'cuda'
dtype = 'bfloat16'
compile = False
