# 二十四史 字符级GPT训练配置
# 25M参数模型，适配 RTX 5070 8GB VRAM

out_dir = 'out-history24-char'
eval_interval = 250  # 每250步评估一次，展示指标
eval_iters = 100
log_interval = 10

# 只在验证loss改善时保存
always_save_checkpoint = False

wandb_log = False
wandb_project = 'history24-char'
wandb_run_name = 'char-25M'

dataset = 'history24_char'
gradient_accumulation_steps = 4  # 有效batch = 4 * 32 = 128
batch_size = 32
block_size = 512  # 上下文512个字符

# ~25M 参数模型
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1  # 数据量相对模型偏小，加点正则化

learning_rate = 1e-3  # 字符级小模型可以用稍大的lr
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

bias = False  # 不用bias，稍快一点

# 系统设置
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Windows上Triton不可用，关闭compile
