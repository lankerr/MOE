"""
Earthformer 20帧 + 49帧 全量实验批量运行脚本 (支持断点续训)
=======================================================
先跑完 20 帧的 7 个实验 (baseline + exp1-6)，
再跑 49 帧的 7 个实验 (baseline + exp1-6)。

断点续训逻辑:
  - 自动检测每个实验的 checkpoints/last.ckpt
  - 如果存在, 从 last.ckpt 继续; 如果不存在, 从头开始
  - 如果已有 best model (earthformer_sevir.pt), 表示该实验已完成, 直接跳过

用法:
    conda activate rtx5070_cu128
    python -u run_all_earthformer_full.py
"""

import os
import sys
import subprocess
import time

EXPERIMENTS = [
    "baseline",
    "exp1_moe_flash",
    "exp1_5_moe_balanced_flash",
    "exp2_swiglu_moe_flash",
    "exp3_balanced_moe_flash",
    "exp4_moe_rope_flash",
    "exp5_swiglu_moe_rope_flash",
    "exp6_balanced_moe_rope_flash",
]

PHASES = [
    {"name": "20帧", "config": "cfg_sevir_20frame.yaml", "prefix": "exp_earthformer"},
    {"name": "49帧", "config": "cfg_sevir_49frame.yaml", "prefix": "exp_earthformer_49f"},
]

EPOCHS = 10
MAX_RETRIES = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPS_DIR = os.path.join(BASE_DIR, "experiments")
SCRIPT = os.path.join(BASE_DIR, "train_experiment_earthformer.py")
PYTHON = r"C:\Users\97290\.conda\envs\rtx5070_cu128\python.exe"


def find_resume_ckpt(save_dir):
    """检查是否有可恢复的 checkpoint，返回 ckpt 文件名或 None"""
    ckpt_dir = os.path.join(EXPS_DIR, save_dir, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return "last.ckpt"
    return None


def is_completed(save_dir):
    """检查实验是否已经完成 (有 best model 导出文件)"""
    best_model = os.path.join(EXPS_DIR, save_dir, "checkpoints", "earthformer_sevir.pt")
    return os.path.exists(best_model)


def run_one(exp_name, config, save_dir):
    """运行单个实验，自动断点续训"""
    # 检查是否已完成
    if is_completed(save_dir):
        print(f"  [跳过] {save_dir} 已完成 (找到 earthformer_sevir.pt)")
        return True

    # 检查是否有可恢复的 checkpoint
    ckpt_name = find_resume_ckpt(save_dir)

    cmd = [
        PYTHON, '-u', SCRIPT,
        '--cfg', config,
        '--exp', exp_name,
        '--epochs', str(EPOCHS),
        '--save', save_dir,
    ]
    if ckpt_name is not None:
        cmd += ['--ckpt_name', ckpt_name]
        print(f"  [续训] 从 {ckpt_name} 恢复")
    else:
        print(f"  [新训] 从 epoch 0 开始")

    print(f"\n{'='*70}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode == 0


def run_with_retry(exp_name, config, save_dir):
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"  -> {exp_name}: 第 {attempt} 次尝试")

        ok = run_one(exp_name, config, save_dir)
        if ok:
            return True
        print(f"  ! {exp_name} 崩溃, 10秒后重试...")
        time.sleep(10)
    return False


def main():
    all_results = {}

    for phase in PHASES:
        phase_name = phase["name"]
        config = phase["config"]
        prefix = phase["prefix"]

        print("\n" + "=" * 70)
        print(f"  Earthformer {phase_name} 实验对比 (baseline + exp1-6 + exp1.5)")
        print(f"  Loss: MAE + MSE + MoE_AuxLoss | Epochs: {EPOCHS}")
        print(f"  断点续训: 自动检测 last.ckpt | 已完成实验自动跳过")
        print("=" * 70)

        phase_results = {}
        for i, exp in enumerate(EXPERIMENTS, 1):
            save_dir = f"{prefix}_{exp}"
            print(f"\n>>> [{phase_name}] [{i}/{len(EXPERIMENTS)}] {exp}")
            print(f"    保存目录: experiments/{save_dir}/")

            t0 = time.time()
            ok = run_with_retry(exp, config, save_dir)
            elapsed = time.time() - t0
            phase_results[exp] = ok
            status = "OK" if ok else "FAIL"
            print(f">>> {exp}: {status} ({elapsed/3600:.1f}h)")

        all_results[phase_name] = phase_results

    # ========== 全局汇总 ==========
    print(f"\n{'='*70}")
    print("  Earthformer 全量实验汇总")
    print(f"{'='*70}")
    total_pass = 0
    total_count = 0
    for phase_name, results in all_results.items():
        print(f"\n  [{phase_name}]")
        for exp, ok in results.items():
            status = "OK" if ok else "FAIL"
            print(f"    {status} {exp}")
            total_count += 1
            if ok:
                total_pass += 1

    print(f"\n完成: {total_pass}/{total_count}")


if __name__ == "__main__":
    main()
