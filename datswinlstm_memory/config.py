"""
全局配置文件
定义数据集路径和其他全局配置
"""

import os


class cfg:
    """全局配置"""
    
    # ============= 数据集路径配置 =============
    # 数据集路径配置 - 自动识别 Windows / WSL
    if os.name == 'nt':
        datasets_dir = r"c:\Users\97290\Desktop\datasets"   # Windows 路径
    else:
        # WSL 路径映射
        datasets_dir = "/mnt/c/Users/97290/Desktop/datasets"  # WSL 路径
    
    # 智算平台路径 (使用时取消注释)
    # datasets_dir = "/data/datasets"
    
    # ============= 模型保存路径 =============
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # ============= 训练配置 =============
    # 默认使用的设备
    device = "cuda"
    
    # 随机种子
    seed = 42
    
    # ============= SEVIR 数据集配置 =============
    sevir_config = {
        "img_type": "vil",
        "raw_seq_len": 49,
        "seq_len": 36,
        "stride": 12,
        "img_height": 384,
        "img_width": 384,
        "interval_real_time": 5,  # 分钟
    }
    
    @classmethod
    def get_sevir_paths(cls):
        """获取 SEVIR 数据集路径"""
        sevir_root = os.path.join(cls.datasets_dir, "sevir")
        return {
            "root_dir": sevir_root,
            "catalog_path": os.path.join(sevir_root, "CATALOG.csv"),
            "data_dir": os.path.join(sevir_root, "data"),
        }


# 确保 checkpoint 目录存在
os.makedirs(cfg.checkpoint_dir, exist_ok=True)
