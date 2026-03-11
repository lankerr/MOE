"""
Exp12→7 训练 10 epochs (L1+MSE loss, 加速版)
=============================================
特性:
  - 崩溃自动重试 (最多3次)
  - 自动从 latest_model.pt 断点续训
  - 已完成的实验自动跳过

用法:
    conda activate rtx5070_cu128
    python -u run_all.py
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
MAX_RETRIES = 3  # 崩溃最多重试次数

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_experiment_fast.py")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def check_experiment_done(exp_name, target_epochs):
    """检查实验是否已完成所有 epochs"""
    log_path = os.path.join(CKPT_DIR, exp_name, "training_log.json")
    if not os.path.exists(log_path):
        return False, 0
    try:
        with open(log_path) as f:
            log = json.load(f)
        if not log:
            return False, 0
        last_epoch = max(e["epoch"] for e in log)
        return last_epoch >= target_epochs, last_epoch
    except:
        return False, 0


def run_one(exp_name, epochs):
    """运行一个实验, train_experiment_fast.py 会自动检测 latest_model.pt 续训"""
    cmd = [PYTHON, "-u", SCRIPT, "--exp", exp_name, "--epochs", str(epochs)]

    print(f"\n{'='*70}")
    print(f"  运行: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def run_with_retry(exp_name, epochs, max_retries=MAX_RETRIES):
    """带自动重试的实验运行 - 崩溃后从 latest_model.pt 自动恢复"""
    for attempt in range(1, max_retries + 1):
        # 先检查是否已完成
        done, last_epoch = check_experiment_done(exp_name, epochs)
        if done:
            print(f"\n>>> {exp_name}: 已完成 {last_epoch} epochs, 跳过")
            return True

        if attempt > 1:
            # 检查从哪个 epoch 恢复
            _, resumed_epoch = check_experiment_done(exp_name, epochs)
            print(f"\n>>> {exp_name}: 第 {attempt} 次尝试 (从 epoch {resumed_epoch} 恢复)")
        
        ok = run_one(exp_name, epochs)
        
        if ok:
            return True
        
        # 失败了, 检查是否有 latest_model.pt 可以恢复
        latest_path = os.path.join(CKPT_DIR, exp_name, "latest_model.pt")
        if os.path.exists(latest_path):
            _, saved_epoch = check_experiment_done(exp_name, epochs)
            print(f"\n⚠ {exp_name} 崩溃! 已保存到 epoch {saved_epoch}")
            print(f"  自动重试 ({attempt}/{max_retries})...")
            time.sleep(5)  # 等5秒让GPU释放
        else:
            print(f"\n⚠ {exp_name} 崩溃且无 checkpoint, 重试 ({attempt}/{max_retries})...")
            time.sleep(5)
    
    print(f"\n✗ {exp_name}: {max_retries} 次重试后仍失败!")
    return False


def main():
    print("=" * 70)
    print(f"  DATSwinLSTM-Memory 训练 (Exp12→7, {EPOCHS} epochs)")
    print(f"  Loss: L1 + MSE (论文公式12)")
    print(f"  自动重试: 最多 {MAX_RETRIES} 次 | 断点续训: latest_model.pt")
    print("=" * 70)

    results = {}
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> 训练 [{i}/{len(EXPERIMENTS)}]: {exp}")
        t0 = time.time()
        ok = run_with_retry(exp, epochs=EPOCHS)
        elapsed = time.time() - t0
        results[exp] = ok
        status = "✓ 完成" if ok else "✗ 失败"
        print(f"\n>>> {exp}: {status}  ({elapsed/3600:.1f}h)")

    # 汇总
    print("\n" + "=" * 70)
    print("  训练汇总")
    print("=" * 70)
    for exp, ok in results.items():
        icon = "✓" if ok else "✗"
        # 读取最终 val_loss
        done, last_epoch = check_experiment_done(exp, EPOCHS)
        log_path = os.path.join(CKPT_DIR, exp, "training_log.json")
        val_info = ""
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
                if log:
                    best_val = min(e["val_loss"] for e in log)
                    val_info = f" | Best Val: {best_val:.4f} | Epochs: {last_epoch}"
            except:
                pass
        print(f"  {icon} {exp}{val_info}")

    passed = sum(1 for ok in results.values() if ok)
    print(f"\n完成: {passed}/{len(EXPERIMENTS)} 个实验训练成功")


if __name__ == "__main__":
    main()
