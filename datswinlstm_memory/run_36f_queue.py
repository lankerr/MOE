"""
36帧实验队列: 基线 + exp7-12 (Flash Attention)
==============================================

顺序运行:
1. baseline_36f_flash  - 36帧基线 (无MoE, 有Flash Attention)
2. exp7_moe_flash      - MoE + Flash
3. exp8_swiglu_moe_flash - SwiGLU-MoE + Flash
4. exp9_balanced_moe_flash - Balanced-MoE + Flash
5. exp10_moe_rope_flash - MoE + RoPE + Flash
6. exp11_swiglu_moe_rope_flash - SwiGLU-MoE + RoPE + Flash
7. exp12_balanced_moe_rope_flash - Balanced-MoE + RoPE + Flash

用法:
  conda activate rtx5070_CU128
  python -u run_36f_queue.py 2>&1 | Tee-Object -FilePath ./checkpoints/_runlogs/queue_36f.log
"""

import subprocess
import sys
import os
import time
import datetime

# 36帧通用参数
COMMON_ARGS_EXPERIMENT = [
    "--seq_len", "36",
    "--input_frames", "12", 
    "--output_frames", "24",
    "--epochs", "100",
    "--batch_size", "1",
    "--num_workers", "0",
    "--no_amp",           # fp32 更稳定 (8GB 够用)
    "--log_interval", "10",
]

# 实验队列: (名称, 脚本, 额外参数)
QUEUE = [
    # 1) 36帧 Flash Attention 基线 (无 MoE)
    ("baseline_36f_flash", "train_36f_baseline.py", [
        "--epochs", "200",
        "--batch_size", "1",
        "--num_workers", "0",
    ]),
    # 2-7) MoE 实验 7-12 (Flash Attention 版, 36帧)
    ("exp7_moe_flash_36f", "train_experiment.py", 
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp7_moe_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
    ("exp8_swiglu_moe_flash_36f", "train_experiment.py",
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp8_swiglu_moe_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
    ("exp9_balanced_moe_flash_36f", "train_experiment.py",
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp9_balanced_moe_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
    ("exp10_moe_rope_flash_36f", "train_experiment.py",
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp10_moe_rope_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
    ("exp11_swiglu_moe_rope_flash_36f", "train_experiment.py",
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp11_swiglu_moe_rope_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
    ("exp12_balanced_moe_rope_flash_36f", "train_experiment.py",
     COMMON_ARGS_EXPERIMENT + ["--exp", "exp12_balanced_moe_rope_flash",
                                "--checkpoint_dir", "./checkpoints_36f"]),
]


def run_experiment(name, script, extra_args):
    """运行一个实验"""
    print(f"\n{'#'*70}")
    print(f"# [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 启动: {name}")
    print(f"# 脚本: {script}")
    print(f"# 参数: {' '.join(extra_args)}")
    print(f"{'#'*70}\n")
    sys.stdout.flush()
    
    cmd = [sys.executable, "-u", script] + extra_args
    t0 = time.time()
    
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=sys.stdout,
        stderr=sys.stdout,  # 合并 stderr 到 stdout
    )
    
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n[FAIL] {name} 失败! 返回码: {result.returncode}, 耗时: {elapsed:.0f}s")
        return False
    else:
        print(f"\n[OK] {name} 完成! 耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return True


def main():
    print("=" * 70)
    print("36帧实验队列 - Flash Attention 基线 + exp7-12")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总实验数: {len(QUEUE)}")
    print("=" * 70)
    
    # 创建必要目录
    os.makedirs("./checkpoints/_runlogs", exist_ok=True)
    os.makedirs("./checkpoints_36f", exist_ok=True)
    
    results = {}
    total_t0 = time.time()
    
    for i, (name, script, extra_args) in enumerate(QUEUE):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(QUEUE)}] {name}")
        print(f"{'='*70}")
        
        success = run_experiment(name, script, extra_args)
        results[name] = "OK" if success else "FAIL"
        
        if not success:
            print(f"\n[!] 实验 {name} 失败, 停止后续实验")
            break
    
    # 汇总
    total_time = time.time() - total_t0
    print(f"\n\n{'='*70}")
    print(f"实验队列完成! 总耗时: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {name:40s} [{status}]")
    
    remaining = [name for name, _, _ in QUEUE if name not in results]
    if remaining:
        print(f"\n未运行的实验:")
        for name in remaining:
            print(f"  {name}")
    
    print(f"\n结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
