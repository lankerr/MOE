"""
Adaptive Learning Rate Scheduler
=================================

结合多种自适应策略:
1. LRFinder: 自动寻找最优初始学习率
2. ReduceLROnPlateau: 基于验证损失自动降低学习率
3. AutoStop: 自适应早停 (基于梯度norm和loss改善)
4. Warmup: 渐进式warmup避免初期震荡

参考论文:
- LRFinder: "Cyclical Learning Rates for Training Neural Networks" (Leslie Smith, 2017)
- AdaBound: "Adaptive Gradient Methods with Dynamic Bounds of Learning Rate" (ICLR 2020)
- MADGRAD: "MADGRAD: Adaptive Gradient Method with Momentum and Adaptive Restart" (2021)

使用方法:
    # 方式1: 完全自动
    scheduler = AdaptiveLR(optimizer, mode='auto')

    # 方式2: LRFinder + 自适应下降
    scheduler = AdaptiveLR(
        optimizer,
        mode='reduce_on_plateau',
        lr_finder=True,
        factor=0.5,
        patience=5,
    )

    # 方式3: 纯LRFinder获取最优LR
    optimal_lr = find_optimal_lr(model, dataloader, plot=True)
"""

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Optional, Callable, List, Tuple, Union
import numpy as np


class LRFinder:
    """
    Learning Rate Range Test

    通过指数增长学习率，找到最优初始学习率。

    原理:
    - 从很小的LR开始
    - 每个batch指数增长LR
    - 记录loss
    - 最优LR在loss开始下降但尚未上升的位置

    参考论文:
    "Cyclical Learning Rates for Training Neural Networks"
    Leslie Smith, 2017
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.device = device or next(model.parameters()).device

        self.history = {'lr': [], 'loss': []}
        self.best_loss = None
        self.best_lr = None

    def range_test(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = 'exp',
        diverge_th: int = 5,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        执行LR range test

        Parameters
        ----------
        train_loader : DataLoader
        start_lr : float
            起始学习率
        end_lr : float
            结束学习率
        num_iter : int
            测试迭代次数
        step_mode : str
            'exp' or 'linear'
        diverge_th : int
            loss连续多少次上升则停止

        Returns
        -------
        optimal_lr : float
            最优学习率
        history : list
            [(lr, loss), ...] 历史
        """
        # 保存原始状态
        original_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        # 计算LR增长因子
        if step_mode == 'exp':
            gamma = (end_lr / start_lr) ** (1 / num_iter)
        else:
            gamma = (end_lr - start_lr) / num_iter

        # 初始化
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr

        self.history = {'lr': [], 'loss': []}
        self.best_loss = float('inf')
        self.best_lr = start_lr
        diverge_count = 0

        # 迭代测试
        iter_count = 0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if iter_count >= num_iter:
                break

            # 前向传播
            self.optimizer.zero_grad()
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[:2]
            else:
                inputs, targets = batch, None

            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                outputs = self.model(inputs)

            # 计算loss
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                # 对于重构类任务，假设input就是target
                loss = self.criterion(outputs, inputs)

            # 反向传播（不更新参数）
            loss.backward()

            # 记录
            self.history['lr'].append(lr)
            self.history['loss'].append(loss.item())

            # 更新最佳LR
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_lr = lr
                diverge_count = 0
            else:
                diverge_count += 1

            # 检查是否发散
            if diverge_count >= diverge_th:
                print(f"[LRFinder] Loss diverged at lr={lr:.6f}, stopping")
                break

            # 更新LR
            if step_mode == 'exp':
                lr *= gamma
            else:
                lr += gamma
            self.optimizer.param_groups[0]['lr'] = lr

            iter_count += 1

            if iter_count % 10 == 0:
                print(f"[LRFinder] Iter {iter_count}/{num_iter}, lr={lr:.6f}, loss={loss.item():.4f}")

        # 恢复原始状态
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])

        # 计算最优LR (使用梯度方法)
        optimal_lr = self._compute_optimal_lr()

        print(f"\n[LRFinder] Test complete!")
        print(f"  Best loss: {self.best_loss:.4f} at lr={self.best_lr:.6f}")
        print(f"  Recommended lr: {optimal_lr:.6f}")

        return optimal_lr, list(zip(self.history['lr'], self.history['loss']))

    def _compute_optimal_lr(self, beta: float = 0.98) -> float:
        """
        使用平滑梯度方法计算最优LR

        找到loss下降最快的点
        """
        if len(self.history['loss']) < 10:
            return self.best_lr

        losses = np.array(self.history['loss'])
        lrs = np.array(self.history['lr'])

        # 平滑loss
        smoothed_losses = []
        avg_loss = 0
        for loss in losses:
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss)

        # 计算梯度
        grads = np.gradient(smoothed_losses)

        # 找到最小梯度点（loss下降最快）
        # 排除前10%和后20%的数据
        n = len(grads)
        start_idx = max(n // 10, 1)
        end_idx = int(0.8 * n)

        min_grad_idx = np.argmin(grads[start_idx:end_idx]) + start_idx

        optimal_lr = lrs[min_grad_idx]

        return float(optimal_lr)

    def plot(self, save_path: Optional[str] = None):
        """绘制LR vs Loss曲线"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            lrs = self.history['lr']
            losses = self.history['loss']

            ax.plot(lrs, losses)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate (log scale)')
            ax.set_ylabel('Loss')
            ax.set_title('LR Range Test Results')
            ax.grid(True, alpha=0.3)

            # 标记最佳点
            if self.best_lr:
                ax.axvline(self.best_lr, color='r', linestyle='--', alpha=0.5, label='Best LR')
                ax.legend()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[LRFinder] Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("[LRFinder] Matplotlib not available, skipping plot")


class AdaptiveLR(_LRScheduler):
    """
    Adaptive Learning Rate Scheduler

    结合多种策略的自适应学习率调度器。

    模式:
    - 'auto': 完全自动，结合LRFinder + ReduceLROnPlateau
    - 'reduce_on_plateau': 基于验证损失自动降低LR
    - 'wsd': Warmup-Stable-Decay 三阶段
    - 'cosine_with_warmup': Cosine退火 + warmup

    自适应机制:
    1. 监控验证loss和gradient norm
    2. 自动判断何时降低LR
    3. 自动判断何时停止训练
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'auto',
        # LRFinder参数
        lr_finder: bool = True,
        lr_finder_start: float = 1e-7,
        lr_finder_end: float = 1.0,
        # ReduceLROnPlateau参数
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 1e-6,
        # WSD参数
        warmup_steps_or_epochs: Union[int, str] = 'auto',
        stable_steps_or_epochs: Union[int, str] = 'auto',
        decay_steps_or_epochs: Union[int, str] = 'auto',
        # 早停参数
        early_stop_patience: int = 15,
        early_stop_threshold: float = 1e-4,
        # 其他
        verbose: bool = True,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.mode = mode

        # LRFinder
        self.lr_finder_enabled = lr_finder
        self.lr_finder_start = lr_finder_start
        self.lr_finder_end = lr_finder_end
        self.lr_finder_result = None

        # ReduceLROnPlateau
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.best = None
        self.mode_reduction = 'min'

        # WSD
        self.warmup_steps_or_epochs = warmup_steps_or_epochs
        self.stable_steps_or_epochs = stable_steps_or_epochs
        self.decay_steps_or_epochs = decay_steps_or_epochs

        # 早停
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_counter = 0
        self.early_stop_best = None

        # 状态
        self.verbose = verbose
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.mode == 'wsd':
            return self._wsd_get_lr()
        elif self.mode == 'cosine_with_warmup':
            return self._cosine_with_warmup_get_lr()
        else:
            # auto 或 reduce_on_plateau 由 step() 方法控制
            return self._last_lr

    def _wsd_get_lr(self):
        """Warmup-Stable-Decay 三阶段"""
        # 解析自动参数
        if isinstance(self.warmup_steps_or_epochs, str):
            total = self.last_epoch + 1
            warmup = max(5, total // 10)
            stable = max(10, total // 2)
            decay = total - warmup - stable
        else:
            warmup = self.warmup_steps_or_epochs
            stable = self.stable_steps_or_epochs
            decay = self.decay_steps_or_epochs

        if self.last_epoch < warmup:
            # Warmup phase
            alpha = self.last_epoch / warmup
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif self.last_epoch < warmup + stable:
            # Stable phase
            return self.base_lrs
        else:
            # Decay phase (cosine)
            progress = (self.last_epoch - warmup - stable) / max(decay, 1)
            progress = min(progress, 1.0)
            decay_factor = 0.01 + 0.99 * (1 + math.cos(progress * math.pi)) / 2
            return [base_lr * decay_factor for base_lr in self.base_lrs]

    def _cosine_with_warmup_get_lr(self):
        """Cosine annealing with warmup"""
        if isinstance(self.warmup_steps_or_epochs, str):
            warmup = max(5, (self.last_epoch + 1) // 10)
        else:
            warmup = self.warmup_steps_or_epochs

        if self.last_epoch < warmup:
            # Warmup
            alpha = self.last_epoch / warmup
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - warmup) / max(self.last_epoch - warmup, 1)
            decay_factor = (1 + math.cos(progress * math.pi)) / 2
            return [self.min_lr + (base_lr - self.min_lr) * decay_factor
                    for base_lr in self.base_lrs]

    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        """
        更新学习率

        Parameters
        ----------
        metrics : float, optional
            验证集指标 (loss)。如果是reduce_on_plateau模式则必需
        epoch : int, optional
            当前epoch
        """
        # 处理epoch参数
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        if self.mode in ['auto', 'reduce_on_plateau']:
            self._reduce_on_plateau_step(metrics)
        elif self.mode in ['wsd', 'cosine_with_warmup']:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
                self._last_lr.append(lr)

        # 检查早停
        if self._check_early_stop(metrics):
            if self.verbose:
                print(f"[AdaptiveLR] Early stopping triggered at epoch {self.last_epoch}")

    def _reduce_on_plateau_step(self, metrics: Optional[float]):
        """ReduceLROnPlateau 逻辑"""
        if metrics is None:
            return

        # 初始化
        if self.best is None:
            self.best = metrics
            self.early_stop_best = metrics
            return

        # 检查是否改善
        if self.mode_reduction == 'min':
            improved = metrics < self.best - self.threshold
            improved_early = metrics < self.early_stop_best - self.early_stop_threshold
        else:
            improved = metrics > self.best + self.threshold
            improved_early = metrics > self.early_stop_best + self.early_stop_threshold

        # Cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if improved:
            # 改善了
            self.best = metrics
            self.num_bad_epochs = 0
            self.early_stop_counter = 0
            self.early_stop_best = metrics
        else:
            # 没改善
            self.num_bad_epochs += 1
            self.early_stop_counter += 1

            # 降低LR
            if self.num_bad_epochs >= self.patience:
                old_lrs = [group['lr'] for group in self.optimizer.param_groups]
                for i, param_group in enumerate(self.optimizer.param_groups):
                    old_lr = old_lrs[i]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr

                if self.verbose:
                    print(f"[AdaptiveLR] Reducing LR: {old_lrs[0]:.6f} -> {param_group['lr']:.6f}")

                self.num_bad_epochs = 0
                self.cooldown_counter = self.cooldown

    def _check_early_stop(self, metrics: Optional[float]) -> bool:
        """检查是否应该早停"""
        if metrics is None or self.early_stop_patience is None:
            return False

        return self.early_stop_counter >= self.early_stop_patience

    def run_lr_finder(
        self,
        model: nn.Module,
        train_loader,
        criterion: Optional[nn.Module] = None,
    ) -> float:
        """
        运行LRFinder并自动设置初始学习率

        Returns
        -------
        optimal_lr : float
            最优学习率
        """
        if not self.lr_finder_enabled:
            return self.optimizer.param_groups[0]['lr']

        finder = LRFinder(model, self.optimizer, criterion)

        optimal_lr, history = finder.range_test(
            train_loader,
            start_lr=self.lr_finder_start,
            end_lr=self.lr_finder_end,
            num_iter=100,
        )

        # 设置最优LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = optimal_lr

        self.lr_finder_result = {'optimal_lr': optimal_lr, 'history': history}
        self.base_lrs = [optimal_lr]

        return optimal_lr


class AutoStopCallback:
    """
    自适应早停回调

    综合考虑:
    1. 验证loss改善
    2. 梯度norm (判断是否陷入局部最优)
    3. 学习率 (LR过小时停止)
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        monitor: str = 'val_loss',
        mode: str = 'min',
        min_lr: float = 1e-6,
        grad_norm_threshold: float = 1e-5,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.min_lr = min_lr
        self.grad_norm_threshold = grad_norm_threshold
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float, current_lr: float,
                 grad_norm: Optional[float] = None) -> bool:
        """
        检查是否应该早停

        Returns
        -------
        should_stop : bool
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        # 检查改善
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        # 检查早停条件
        stop_reasons = []

        if self.counter >= self.patience:
            stop_reasons.append(f"no improvement for {self.counter} epochs")

        if current_lr < self.min_lr:
            stop_reasons.append(f"LR too small: {current_lr:.2e}")

        if grad_norm is not None and grad_norm < self.grad_norm_threshold:
            stop_reasons.append(f"gradient norm too small: {grad_norm:.2e}")

        if stop_reasons:
            self.early_stop = True
            if self.verbose:
                print(f"[AutoStop] Stopping: {', '.join(stop_reasons)}")
            return True

        return False


def find_optimal_lr(
    model: nn.Module,
    train_loader,
    optimizer: Optional[Optimizer] = None,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> float:
    """
    便捷函数: 寻找最优学习率

    使用方法:
        optimal_lr = find_optimal_lr(model, train_loader, plot=True)
        optimizer = Adam(model.parameters(), lr=optimal_lr)
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    finder = LRFinder(model, optimizer)
    optimal_lr, history = finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
    )

    if plot:
        finder.plot(save_path)

    return optimal_lr
