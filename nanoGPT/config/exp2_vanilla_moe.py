# 实验2: Vanilla MoE (无辅助损失，最朴素的MoE)
# 8个专家，top-2 路由，不加任何辅助损失

out_dir = 'out-exp2-vanilla-moe'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'moe-history24'
wandb_run_name = 'exp2-vanilla-moe'

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

# MoE: 朴素版
n_exp = 8
moe_top_k = 2
moe_start_layer = 2
moe_stride = 1
use_aux_loss = False
use_router_z_loss = False
use_noisy_top_k = False
activation = 'gelu'

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
