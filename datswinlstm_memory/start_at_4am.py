import time
import datetime
import subprocess
import sys
import os

def schedule_run(target_hour=6, target_minute=0):
    now = datetime.datetime.now()
    
    # 计算目标时间 (今天或者明天的 6:00 AM)
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target_time < now:
        # 如果当前时间已经过了今天的 6:00 AM，就设定为明天的 6:00 AM
        target_time += datetime.timedelta(days=1)
        
    print("=" * 60)
    print(f"🕒 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🚀 计划启动: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 倒计时循环
    while True:
        current = datetime.datetime.now()
        remaining = (target_time - current).total_seconds()
        
        if remaining <= 0:
            print("\n\n⏰ 时间到！开始执行 36帧 训练任务...")
            break
            
        # 格式化倒计时显示
        hours, remainder = divmod(int(remaining), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        sys.stdout.write(f"\r⏳ 倒计时: {hours:02d}小时 {minutes:02d}分钟 {seconds:02d}秒 ... (按 Ctrl+C 取消)")
        sys.stdout.flush()
        time.sleep(1)

def run_36frame_pipeline():
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*60)
    print("▶️ [1/2] 开始运行 36帧 Baseline ...")
    print("="*60)
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
    print("▶️ [2/2] 开始运行 36帧 Exp7-12 (MoE/Flash) ...")
    print("="*60)
    exp_cmd = [sys.executable, "-u", "run_all_36frame.py"]
    subprocess.run(exp_cmd, cwd=cwd)
    
    print("\n✅ 所有 36帧 训练任务已全部完成！")

if __name__ == "__main__":
    try:
        schedule_run(target_hour=6, target_minute=0)
        run_36frame_pipeline()
    except KeyboardInterrupt:
        print("\n\n🛑 倒计时或任务已被用户取消。")
