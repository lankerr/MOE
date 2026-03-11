# run_all_experiments.py — 顺序执行所有 MoE 实验
import subprocess
import sys
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

experiments = [
    ('config/exp1_dense.py',       'Exp1: Dense 基线'),
    ('config/exp2_vanilla_moe.py', 'Exp2: Vanilla MoE'),
    ('config/exp3_full_moe.py',    'Exp3: Full MoE'),
    ('config/exp4a_relu2.py',      'Exp4a: MoE+ReLU²'),
    ('config/exp4b_swiglu.py',     'Exp4b: MoE+SwiGLU'),
]

total = len(experiments)
for i, (cfg, name) in enumerate(experiments):
    print(f"\n{'#'*70}")
    print(f"  [{i+1}/{total}] 开始: {name}")
    print(f"  配置: {cfg}")
    print(f"  时间: {time.strftime('%H:%M:%S')}")
    print(f"{'#'*70}\n")

    t0 = time.time()
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    ret = subprocess.run(
        [sys.executable, '-u', 'train_moe.py', cfg],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    elapsed = (time.time() - t0) / 60

    if ret.returncode == 0:
        print(f"\n  ✓ {name} 完成! 用时 {elapsed:.1f}min")
    else:
        print(f"\n  ✗ {name} 失败! (exit code {ret.returncode})")
        print(f"    跳过，继续下一个...")

print(f"\n{'#'*70}")
print(f"  全部实验完成! 运行对比分析...")
print(f"{'#'*70}\n")
subprocess.run([sys.executable, 'compare_experiments.py'])
