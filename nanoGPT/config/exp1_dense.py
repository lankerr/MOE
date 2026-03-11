# 实验1: Dense 基线 (n_exp=1, 普通Transformer)
# 用 MoE 框架但不开启 MoE，作为公平对比的基线
# 参数量 ~25M，与原 char 模型一致

out_dir = 'out-exp1-dense'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'moe-history24'
wandb_run_name = 'exp1-dense-baseline'

dataset = 'history24_char'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

# 模型结构 (Dense)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# MoE 关闭
n_exp = 1
moe_top_k = 1
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
