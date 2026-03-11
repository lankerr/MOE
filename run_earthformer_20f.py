import os
import sys
import subprocess
from pathlib import Path

def main():
    # Setup Earthformer paths
    cwd = os.path.dirname(os.path.abspath(__file__))
    earthformer_dir = os.path.join(cwd, "earth-forecasting-transformer")
    script_dir = os.path.join(earthformer_dir, "scripts", "cuboid_transformer", "sevir")
    
    # Target Config
    config_path = "cfg_sevir_20frame.yaml"
    
    # Save Path
    save_dir = "exp_earthformer_20frame_mae_mse"
    
    print("\n" + "="*70)
    print("🚀 启动纯大模型对照组 (Pure Earthformer Transformer) 🚀")
    print("="*70)
    print("配置档案:")
    print(" - 架构: Hierarchical Cuboid Transformer (无 LSTM，纯注意力机制)")
    print(" - 时序: 20 帧总长 (8帧历史 -> 12帧未来)")
    print(" - 损失函数: MAE (L1) + MSE (与 DATSwinLSTM 完全一致)")
    print(f" - 工作目录: {script_dir}")
    print(f" - 配置文件:  {config_path}")
    print(f" - 保存路径: {save_dir}")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "-u", "train_cuboid_sevir.py",
        "--cfg", config_path,
        "--save", save_dir
    ]
    
    try:
        subprocess.run(cmd, cwd=script_dir)
        print("\n✅ Earthformer 20帧训练流程已结束！")
    except KeyboardInterrupt:
        print("\n\n🛑 Earthformer 训练被用户手动中断。")

if __name__ == "__main__":
    main()
