import sys
import os
import subprocess

def run_36frame_pipeline_now():
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*60)
    print("▶️ [1/2] 开始运行 36帧 Baseline ... (将自动寻找断点接力)")
    print("="*60)
    # train_384_opt.py 会自动读取 save_dir 下的 latest_model.pt 进行续训
    baseline_cmd = [
        sys.executable, "-u", "train_384_opt.py", 
        "--epochs", "10", 
        "--seq_len", "36", 
        "--in_len", "12", 
        "--out_len", "24", 
        "--save_dir", "./checkpoints_36frame/baseline_36f"
    ]
    subprocess.run(baseline_cmd, cwd=cwd)
    
    print("\n" + "="*60)
    print("▶️ [2/2] 开始运行 36帧 Exp7-12 (MoE/Flash) ... (同样自动接力)")
    print("="*60)
    exp_cmd = [sys.executable, "-u", "run_all_36frame.py"]
    subprocess.run(exp_cmd, cwd=cwd)
    
    print("\n✅ 所有 36帧 训练任务已全部完成！")

if __name__ == "__main__":
    try:
        run_36frame_pipeline_now()
    except KeyboardInterrupt:
        print("\n\n🛑 任务已被用户取消。")
