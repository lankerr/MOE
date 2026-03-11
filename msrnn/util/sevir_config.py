import os
from omegaconf import OmegaConf

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


datacfg = OmegaConf.create()
# cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..", "..")))
datacfg.root_dir = '/data_8t/WSG'
# print(datacfg.root_dir)
datacfg.datasets_dir = os.path.join(datacfg.root_dir, "data")  # default directory for loading datasets
datacfg.pretrained_checkpoints_dir = os.path.join(datacfg.root_dir, "pretrained_checkpoints")  # default directory for saving and loading pretrained checkpoints
datacfg.exps_dir = os.path.join(datacfg.root_dir, "experiments")  # default directory for saving experiment results
os.makedirs(datacfg.exps_dir, exist_ok=True)
