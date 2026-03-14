import os
from yacs.config import CfgNode as CN

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))

cfg = CN()
cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..", "..")))
# cfg.datasets_dir = os.path.join(cfg.root_dir, "datasets")
if os.name == 'nt':
    cfg.datasets_dir = r"X:\datasets"
else:
    cfg.datasets_dir = "/mnt/x/datasets"
cfg.pretrained_checkpoints_dir = os.path.join(cfg.root_dir, "pretrained_checkpoints")
cfg.exps_dir = os.path.join(cfg.root_dir, "experiments")

if not os.path.exists(cfg.exps_dir):
    os.makedirs(cfg.exps_dir, exist_ok=True)
