# 实验4a: ReLU² 激活函数 (Full MoE + ReLU Squared)
# 来自 nanoMoE 默认设置，据称在 MoE 中效果好于 GELU

out_dir = 'out-exp4a-relu2'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'moe-history24'
wandb_run_name = 'exp4a-relu2'

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

# MoE: Full + ReLU²
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
activation = 'relu2'

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
