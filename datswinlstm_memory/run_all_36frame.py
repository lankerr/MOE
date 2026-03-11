"""
Exp7-12 训练 36帧 3小时外推版 (L1+MSE, Flash Attention)
=======================================================
与 20帧版区别:
  - seq_len=36 (36帧 × 5min = 3小时)
  - input_frames=12 (12帧 = 1小时观测)
  - output_frames=12 (12帧 = 1小时预测, 与论文一致)
  - long_len=36 (Flash Attention 支持更长 memory)
  - checkpoint 保存到 checkpoints_36frame/ (不覆盖20帧结果)

用法:
    conda activate rtx5070_cu128
    python -u run_all_36frame.py
"""

import os
import sys
import subprocess
import time
import json

EXPERIMENTS = [
    "exp12_balanced_moe_rope_flash",
    "exp11_swiglu_moe_rope_flash",
    "exp10_moe_rope_flash",
    "exp9_balanced_moe_flash",
    "exp8_swiglu_moe_flash",
    "exp7_moe_flash",
]

EPOCHS = 10
MAX_RETRIES = 3

# 36帧参数
SEQ_LEN = 36          # 36帧 = 3小时
INPUT_FRAMES = 12     # 12帧 = 1小时观测 (论文标准)
OUTPUT_FRAMES = 12    # 12帧 = 1小时外推 (论文标准)
CKPT_DIR = './checkpoints_36frame'

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_experiment_fast.py")


def check_done(exp_name, target_epochs):
    log_path = os.path.join(CKPT_DIR, exp_name, 'training_log.json')
    if not os.path.exists(log_path):
        return False, 0
    try:
        with open(log_path) as f:
            log = json.load(f)
        if not log:
            return False, 0
        last = max(e['epoch'] for e in log)
        return last >= target_epochs, last
    except:
        return False, 0


def run_one(exp_name):
    cmd = [
        PYTHON, '-u', SCRIPT,
        '--exp', exp_name,
        '--epochs', str(EPOCHS),
        '--seq_len', str(SEQ_LEN),
        '--input_frames', str(INPUT_FRAMES),
        '--output_frames', str(OUTPUT_FRAMES),
        '--checkpoint_dir', CKPT_DIR,
    ]
    print(f"\n{'='*70}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def run_with_retry(exp_name):
    for attempt in range(1, MAX_RETRIES + 1):
        done, last = check_done(exp_name, EPOCHS)
        if done:
            print(f"  ✓ {exp_name}: 已完成 {last} epochs, 跳过")
            return True

        if attempt > 1:
            _, ep = check_done(exp_name, EPOCHS)
            print(f"  → {exp_name}: 第 {attempt} 次尝试 (从 epoch {ep} 恢复)")

        ok = run_one(exp_name)
        done, _ = check_done(exp_name, EPOCHS)
        if done:
            return True
        if not ok:
            print(f"  ⚠ {exp_name} 崩溃, 5秒后重试...")
            time.sleep(5)

    return False


def main():
    print("=" * 70)
    print(f"  DATSwinLSTM-Memory 36帧 3小时外推训练")
    print(f"  seq_len={SEQ_LEN} | in={INPUT_FRAMES} | out={OUTPUT_FRAMES}")
    print(f"  Loss: L1 + MSE | Epochs: {EPOCHS}")
    print(f"  Checkpoint: {CKPT_DIR}")
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
    print("  36帧训练汇总")
    print(f"{'='*70}")
    for exp, ok in results.items():
        _, last = check_done(exp, EPOCHS)
        log_path = os.path.join(CKPT_DIR, exp, 'training_log.json')
        val_info = ""
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
                if log:
                    best_val = min(e['val_loss'] for e in log)
                    val_info = f" | Best Val: {best_val:.4f} | Epochs: {last}"
            except:
                pass
        print(f"  {'✓' if ok else '✗'} {exp}{val_info}")

    passed = sum(1 for ok in results.values() if ok)
    print(f"\n完成: {passed}/{len(EXPERIMENTS)}")


if __name__ == "__main__":
    main()
