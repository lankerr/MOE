"""
WSD (Warmup-Stable-Decay) 学习率调度器实现

基于论文: Universal Dynamics of Warmup Stable Decay (arxiv 2024)
https://arxiv.org/abs/2401.11079

三阶段设计:
1. Warmup: 线性增长到峰值 LR
2. Stable: 保持峰值 LR (主要学习阶段)
3. Decay: Cosine 衰减到最小 LR

用法:
    from earthformer.utils.lr_schedulers import WSDScheduler

    scheduler = WSDScheduler(
        optimizer,
        warmup_epochs=2,     # 前 2 epoch warmup
        stable_epochs=30,    # 中间 30 epoch 稳定
        decay_epochs=18,     # 最后 18 epoch 衰减
    )
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay 学习率调度器

    Args:
        optimizer: PyTorch 优化器
        warmup_epochs: Warmup 阶段 epoch 数
        stable_epochs: 稳定阶段 epoch 数
        decay_epochs: 衰减阶段 epoch 数
        min_lr_factor: 最小学习率相对于峰值 LR 的比例
        last_epoch: 当前 epoch (用于恢复训练)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        stable_epochs: int,
        decay_epochs: int,
        min_lr_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.stable_epochs = stable_epochs
        self.decay_epochs = decay_epochs
        self.min_lr_factor = min_lr_factor

        super(WSDScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 阶段 1: Warmup - 线性增长
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_epochs + self.stable_epochs:
            # 阶段 2: Stable - 保持峰值
            return self.base_lrs

        else:
            # 阶段 3: Decay - Cosine 衰减
            progress = (self.last_epoch - self.warmup_epochs - self.stable_epochs) / self.decay_epochs
            progress = min(progress, 1.0)

            # Cosine 衰减: 从 1.0 衰减到 min_lr_factor
            decay_factor = self.min_lr_factor + (1 - self.min_lr_factor) * \
                          (1 + math.cos(progress * math.pi)) / 2
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def create_wsd_scheduler(
    optimizer,
    max_epochs: int = 50,
    warmup_ratio: float = 0.04,
    stable_ratio: float = 0.60,
    min_lr_factor: float = 0.01,
):
    """
    创建 WSD 调度器的便捷函数

    Args:
        optimizer: PyTorch 优化器
        max_epochs: 总训练 epoch 数
        warmup_ratio: Warmup 占总训练的比例 (默认 4%)
        stable_ratio: Stable 占总训练的比例 (默认 60%)
        min_lr_factor: 最小 LR 比例 (默认 0.01)

    Returns:
        WSDScheduler 实例
    """
    warmup_epochs = int(max_epochs * warmup_ratio)
    stable_epochs = int(max_epochs * stable_ratio)
    decay_epochs = max_epochs - warmup_epochs - stable_epochs

    return WSDScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        stable_epochs=stable_epochs,
        decay_epochs=decay_epochs,
        min_lr_factor=min_lr_factor,
    )


# 使用示例
if __name__ == "__main__":
    import torch.nn as nn

    # 创建模型和优化器
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 创建 WSD 调度器
    scheduler = WSDScheduler(
        optimizer,
        warmup_epochs=2,
        stable_epochs=30,
        decay_epochs=18,
    )

    # 模拟训练
    print("Epoch | LR")
    print("------|---------")
    for epoch in range(50):
        lr = scheduler.get_last_lr()[0]
        if epoch % 10 == 0 or epoch < 5:
            print(f"{epoch:5d} | {lr:.6f}")
        scheduler.step()

    print("\nWSD Scheduler 测试完成!")
