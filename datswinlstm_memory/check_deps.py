
import sys
import os
sys.path.insert(0, os.getcwd())

print("Checking dependencies...")

try:
    import torch
    print(f"torch: {torch.__version__}")
except ImportError:
    print("torch: MISSING")

try:
    import pytorch_lightning
    print(f"pytorch_lightning: {pytorch_lightning.__version__}")
except ImportError:
    print("pytorch_lightning: MISSING")

try:
    import timm
    print("timm: OK")
except ImportError:
    print("timm: MISSING")

try:
    import einops
    print("einops: OK")
except ImportError:
    print("einops: MISSING")

try:
    import models.DATSwinLSTM_D_Memory
    print("models package: OK")
except ImportError as e:
    print(f"models package: MISSING - {e}")

try:
    from models.DATSwinLSTM_D_Memory import Memory
    print("Memory class: OK")
except ImportError as e:
    print(f"Memory class: MISSING - {e}")
