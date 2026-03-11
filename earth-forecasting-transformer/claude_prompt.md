# Role: Expert AI Developer specializing in PyTorch & Deep Learning Infrastructure

**Objective**: Complete the code migration of the "Earthformer" project from PyTorch Lightning 1.x to PyTorch Lightning 2.6+, removing the dependency on NVIDIA Apex.

**Context**:
The user is running on an RTX 5070 (Blackwell) which requires CUDA 12.8. The old code (based on PL 1.9) is incompatible. We have already upgraded the environment to PyTorch 2.6.0 and Lightning 2.6.1. We have also mocked the missing `apex` library to prevent immediate import errors, but the core training loop logic is now broken due to API changes in Lightning 2.x.

**Reference Plan**:
Please read the file `20260131_modernization_plan.md` for the full technical breakdown.

**Your Tasks**:
1.  **Refactor `train_cuboid_sevir.py`**:
    -   Update `pl.Trainer` initialization: replace `gpus` with `devices`, handle the removal of `resume_from_checkpoint` (move to `.fit()`), and fix other deprecated arguments.
    -   **CRITICAL**: Refactor `*_epoch_end` methods. PL 2.0 replaced `validation_epoch_end` with `on_validation_epoch_end`. You must rewrite the logic to aggregate outputs manually if `outputs` argument is no longer passed the same way.
2.  **Clean up `apex_ddp.py`**:
    -   Ensure the Mock implementation is robust or switch the code to use `lightning.pytorch.strategies.DDPStrategy` directly.
3.  **Verify & Fix**:
    -   Address any `AttributeError` or `TypeError` arising from the API upgrade.

**Output**:
Provide the fully refactored code for `scripts/cuboid_transformer/sevir/train_cuboid_sevir.py` and any other modified files.
