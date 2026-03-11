"""
Earthformer Exp1-6 批量运行脚本
=======================================================
用以评估在 Earthformer 上采用相同控制变量 (MoE, RoPE, Flash Attention) 
的公平对比方案。

用法:
    conda activate rtx5070_cu128
    python -u run_all_earthformer.py
"""

import os
import sys
import subprocess
import time

EXPERIMENTS = [
    "baseline",
    "exp1_moe_flash",
    "exp2_swiglu_moe_flash",
    "exp3_balanced_moe_flash",
    "exp4_moe_rope_flash",
    "exp5_swiglu_moe_rope_flash",
    "exp6_balanced_moe_rope_flash",
]

EPOCHS = 10
MAX_RETRIES = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(BASE_DIR, "train_experiment_earthformer.py")

CONFIG = "cfg_sevir_20frame.yaml"
# 使用调起本脚本的 python 解析器
PYTHON = sys.executable

def run_one(exp_name):
    # 为当前实验指定单独的保存目录
    save_dir = f"exp_earthformer_{exp_name}"
    cmd = [
        PYTHON, '-u', SCRIPT,
        '--cfg', CONFIG,
        '--exp', exp_name,
        '--epochs', str(EPOCHS),
        '--save', save_dir
    ]
    print(f"\n{'='*70}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0

def run_with_retry(exp_name):
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"  → {exp_name}: 第 {attempt} 次尝试")

        ok = run_one(exp_name)
        if ok:
            return True
        if not ok:
            print(f"  ⚠ {exp_name} 崩溃, 10秒后重试...")
            time.sleep(10)

    return False

def main():
    print("=" * 70)
    print(f"  Earthformer 20帧 实验对比 (Exp1-6)")
    print(f"  Loss: MAE + MSE | Epochs: {EPOCHS}")
    print("=" * 70)

    results = {}
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> [{i}/{len(EXPERIMENTS)}] {exp}")
        t0 = time.time()
        ok = run_with_retry(exp)
        elapsed = time.time() - t0
        results[exp] = ok
        print(f">>> {exp}: {'✓' if ok else '✗'} ({elapsed/3600:.1f}h)")

    # 汇总
    print(f"\n{'='*70}")
    print("  Earthformer 对比训练汇总")
    print(f"{'='*70}")
    for exp, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {exp}")

    passed = sum(1 for ok in results.values() if ok)
    print(f"\n完成: {passed}/{len(EXPERIMENTS)}")

if __name__ == "__main__":
    main()
