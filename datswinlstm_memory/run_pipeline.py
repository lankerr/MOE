"""
完整流水线: 训练 (补完) → 384基线 → 评估对比
==============================================
1. 检查 exp7-12 是否跑完 10 epochs, 未完成的继续训练
2. 训练 384_opt 基线 (同 L1+MSE loss)
3. 对所有模型执行 CSI/HSS/POD/FAR 评估
4. 生成对比报告

用法:
    conda activate rtx5070_cu128
    python -u run_pipeline.py
"""
import os
import sys
import json
import subprocess
import time

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

EXPERIMENTS = [
    'exp12_balanced_moe_rope_flash',
    'exp11_swiglu_moe_rope_flash',
    'exp10_moe_rope_flash',
    'exp9_balanced_moe_flash',
    'exp8_swiglu_moe_flash',
    'exp7_moe_flash',
]
EPOCHS = 10
MAX_RETRIES = 3


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


def run_cmd(cmd, label=""):
    print(f"\n{'='*70}")
    print(f"  [{label}] {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def stage1_finish_training():
    """阶段1: 补完 exp7-12 训练"""
    print("\n" + "#" * 70)
    print("  阶段1: 检查/补完 exp7-12 训练 (10 epochs)")
    print("#" * 70)

    for exp in EXPERIMENTS:
        done, last = check_done(exp, EPOCHS)
        if done:
            print(f"  ✓ {exp}: 已完成 {last} epochs, 跳过")
            continue

        print(f"  → {exp}: 仅 {last} epochs, 继续训练...")
        for attempt in range(1, MAX_RETRIES + 1):
            ok = run_cmd(
                [PYTHON, '-u', 'train_experiment_fast.py', '--exp', exp, '--epochs', str(EPOCHS)],
                f"{exp} 尝试 {attempt}"
            )
            done, last = check_done(exp, EPOCHS)
            if done:
                print(f"  ✓ {exp}: 完成 {last} epochs")
                break
            if not ok and attempt < MAX_RETRIES:
                print(f"  ⚠ {exp} 崩溃, 5秒后重试...")
                time.sleep(5)

        if not done:
            print(f"  ✗ {exp}: 训练失败!")


def stage2_train_baseline():
    """阶段2: 训练 384_opt 基线 (L1+MSE)"""
    print("\n" + "#" * 70)
    print("  阶段2: 训练 384x384_opt 基线 (L1+MSE)")
    print("#" * 70)

    done, last = check_done('384x384_opt', EPOCHS)
    if done:
        print(f"  ✓ 384x384_opt: 已完成 {last} epochs, 跳过")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        ok = run_cmd(
            [PYTHON, '-u', 'train_384_opt.py', '--epochs', str(EPOCHS)],
            f"384_opt 尝试 {attempt}"
        )
        done, last = check_done('384x384_opt', EPOCHS)
        if done:
            print(f"  ✓ 384x384_opt: 完成 {last} epochs")
            break
        if not ok and attempt < MAX_RETRIES:
            print(f"  ⚠ 384_opt 崩溃, 5秒后重试...")
            time.sleep(5)


def stage3_evaluate():
    """阶段3: 评估所有模型"""
    print("\n" + "#" * 70)
    print("  阶段3: 评估 CSI/HSS/POD/FAR")
    print("#" * 70)

    ok = run_cmd(
        [PYTHON, '-u', 'evaluate.py', '--all'],
        "评估所有模型"
    )
    if ok:
        print("  ✓ 评估完成!")
    else:
        print("  ✗ 评估失败!")


def main():
    t0 = time.time()
    print("=" * 70)
    print("  DATSwinLSTM-Memory 完整流水线")
    print("  阶段1: 补完 exp7-12 训练")
    print("  阶段2: 训练 384_opt 基线")
    print("  阶段3: 评估 CSI/HSS/POD/FAR")
    print("=" * 70)

    stage1_finish_training()
    stage2_train_baseline()
    stage3_evaluate()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  流水线完成! 总耗时: {elapsed/3600:.1f} 小时")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
