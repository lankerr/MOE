"""
Bug 修复补丁 - train_49f_gmr.py

修复三个关键问题：
1. GMR 替换范围错误 - 只应替换 initial_encoder + final_decoder
2. 测试集数据泄露 - test_dataloader 返回 val_dataset
3. 学习率偏高 - 1e-3 应改为 3e-4

应用方法：
    1. 备份原文件: cp train_49f_gmr.py train_49f_gmr.py.backup
    2. 应用此修复
"""

# ============================================================
# 修复 1: GMR 替换范围错误 (第 154-157 行)
# ============================================================
# 旧代码:
# patch_model_with_gmr(self.torch_nn_module)
# stats = count_gmr_layers(self.torch_nn_module)
# print(f"[GMR-Conv] 层统计: {stats}")

# 新代码 - 只替换 initial_encoder + final_decoder:
from gmr_patch_embed import patch_model_with_gmr_embed, count_gmr_embed_layers
patch_model_with_gmr_embed(
    self.torch_nn_module,
    in_chans=model_cfg["input_shape"][-1],
    embed_dim=model_cfg["base_units"],
)
stats = count_gmr_embed_layers(self.torch_nn_module)
print(f"[GMR-Conv] 已替换 initial_encoder + final_decoder: {stats}")


# ============================================================
# 修复 2: 测试集数据泄露
# ============================================================
# 找到 SEVIRLightningDataModule 中的 test_dataloader 方法
# 需要先创建 test_dataset，然后修改 test_dataloader

# 在 __init__ 方法的数据集定义部分（约第 320-345 行）添加:
# 在 self.val_dataset 定义之后添加:

# 新增 test_dataset 定义
start_date_test = datetime.datetime(*dataset_oc['train_test_split_date'])
end_date_test = datetime.datetime(*dataset_oc['end_date'])

self.test_dataset = SEVIRTorchDataset(
    sevir_catalog=sevir_catalog,
    sevir_data_dir=sevir_data_dir,
    seq_len=dataset_oc['in_len'] + dataset_oc['out_len'],
    batch_size=micro_batch_size,
    start_date=start_date_test,  # 使用 test 划分日期
    end_date=end_date_test,
    shuffle=False,
    verbose=True,
    layout="NTHWC"
)

# 修改 test_dataloader 方法 (约第 509-512 行):
# 旧代码:
# def test_dataloader(self):
#     return DataLoader(self.val_dataset, ...)

# 新代码:
def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.micro_batch_size,
                      shuffle=False, drop_last=False, collate_fn=self.collate_fn,
                      num_workers=self.num_workers)


# ============================================================
# 修复 3: 学习率偏高
# ============================================================
# 在配置文件中修改 lr 参数
# cfg_sevir_49frame_gmr.yaml 中:
# optim:
#   lr: 0.001  # 改为 3e-4

# 或在代码中修改 (约第 172 行):
# self.lr = oc.optim.lr  # 如果是 0.001，需要改为 0.0003


# ============================================================
# 完整的 test_dataloader 修复代码
# ============================================================

def test_dataloader(self):
    """
    返回测试集 DataLoader (修复: 之前返回 val_dataset)
    """
    return DataLoader(self.test_dataset, batch_size=self.micro_batch_size,
                      shuffle=False, drop_last=False, collate_fn=self.collate_fn,
                      num_workers=self.num_workers)


# ============================================================
# 导入语句更新 (文件顶部)
# ============================================================
# 旧导入:
# from gmr_layers import patch_model_with_gmr, count_gmr_layers

# 新导入 (删除旧的，添加新的):
from gmr_patch_embed import patch_model_with_gmr_embed, count_gmr_embed_layers
