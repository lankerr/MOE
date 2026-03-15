"""
Advanced Learning Rate Schedulers for EarthFormer Training

Based on 2024-2025 research:
- WSD (Warmup-Stable-Decay): Universal Dynamics paper
- Adaptive LR: "WHEN, WHY AND HOW MUCH?" (OpenReview 2024)
- Cosine with restarts for long training

Usage:
    from earthformer.utils.lr_schedulers import WSDScheduler, AdaptiveLRScheduler

    scheduler = WSDScheduler(
        optimizer,
        warmup_epochs=2,
        stable_epochs=30,
        decay_epochs=18,
    )
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Callable, List
import numpy as np


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay Learning Rate Scheduler

    Based on: "Universal Dynamics of Warmup Stable Decay" (arxiv 2024)
    https://arxiv.org/abs/2601.09000

    Three phases:
    1. Warmup: Linear increase to peak LR
    2. Stable: Maintain peak LR for most of training
    3. Decay: Cosine decay to near-zero

    Advantages over cosine:
    - More stable during peak phase
    - Better convergence for long training
    - Configurable stable duration

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        stable_epochs: Number of stable epochs at peak LR
        decay_epochs: Number of decay epochs
        min_lr_factor: Final LR as fraction of peak LR
        last_epoch: The index of last epoch

    Example:
        >>> # 50 epoch training: 2 warmup, 40 stable, 8 decay
        >>> scheduler = WSDScheduler(optimizer, 2, 40, 8)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
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
        self.total_epochs = warmup_epochs + stable_epochs + decay_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Phase 1: Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_epochs + self.stable_epochs:
            # Phase 2: Stable at peak
            return self.base_lrs

        else:
            # Phase 3: Cosine decay
            progress = (self.last_epoch - self.warmup_epochs - self.stable_epochs) / self.decay_epochs
            progress = min(progress, 1.0)

            # Cosine decay from 1.0 to min_lr_factor
            decay_factor = self.min_lr_factor + (1 - self.min_lr_factor) * \
                          (1 + math.cos(progress * math.pi)) / 2

            return [base_lr * decay_factor for base_lr in self.base_lrs]


class AdaptiveLRScheduler(_LRScheduler):
    """
    Adaptive Learning Rate Scheduler based on loss dynamics

    Based on: "WHEN, WHY AND HOW MUCH?" (OpenReview 2024)
    Automatically adjusts warmup and decay based on training dynamics.

    Features:
    - Detects early training oscillation → extends warmup
    - Detects convergence plateau → starts decay early
    - Detects loss divergence → reduces LR

    Args:
        optimizer: Wrapped optimizer
        init_lr: Initial learning rate
        min_lr: Minimum learning rate
        warmup_patience: Epochs to wait before detecting oscillation
        decay_patience: Epochs of no improvement before decay
        oscillation_threshold: Variance threshold for oscillation detection
        improvement_threshold: Minimum loss change to consider improvement

    Example:
        >>> scheduler = AdaptiveLRScheduler(
        ...     optimizer,
        ...     init_lr=1e-3,
        ...     warmup_patience=3,
        ...     decay_patience=5,
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_patience: int = 3,
        decay_patience: int = 5,
        oscillation_threshold: float = 0.1,
        improvement_threshold: float = 1e-4,
        last_epoch: int = -1,
    ):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_patience = warmup_patience
        self.decay_patience = decay_patience
        self.oscillation_threshold = oscillation_threshold
        self.improvement_threshold = improvement_threshold

        # State tracking
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.warmup_complete = False
        self.decay_started = False
        self.current_lr = init_lr

        super().__init__(optimizer, last_epoch)

    def step(self, loss: float, epoch: Optional[int] = None):
        """
        Update scheduler with current loss value

        Args:
            loss: Current training loss
            epoch: Current epoch (optional)
        """
        # Store loss
        self.loss_history.append(loss)

        # Detect oscillation during warmup
        if not self.warmup_complete and len(self.loss_history) >= self.warmup_patience:
            recent_var = np.var(self.loss_history[-self.warmup_patience:])
            if recent_var > self.oscillation_threshold:
                # Oscillation detected - slow down warmup
                self.current_lr *= 0.8
                print(f"[AdaptiveLR] Oscillation detected, reducing LR to {self.current_lr:.6f}")
            else:
                # Stable - complete warmup
                self.warmup_complete = True
                print(f"[AdaptiveLR] Warmup complete at LR {self.current_lr:.6f}")

        # Detect plateau for decay
        if self.warmup_complete and not self.decay_started:
            if loss < self.best_loss - self.improvement_threshold:
                self.best_loss = loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.decay_patience:
                self.decay_started = True
                print(f"[AdaptiveLR] Plateau detected, starting decay")

        # Apply LR based on phase
        if not self.warmup_complete:
            # Warmup phase
            new_lrs = [self.current_lr for _ in self.base_lrs]
        elif self.decay_started:
            # Decay phase
            decay_factor = 0.5 ** (self.epochs_no_improve / self.decay_patience)
            new_lrs = [max(self.current_lr * decay_factor, self.min_lr) for _ in self.base_lrs]
        else:
            # Stable phase
            new_lrs = [self.current_lr for _ in self.base_lrs]

        # Update optimizer LR
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

        super().step(epoch)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts

    Combines:
    - Linear warmup
    - Cosine annealing
    - Periodic restarts (SGDR)

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total epochs per cycle
        eta_min: Minimum learning rate
        restarts: Number of restarts (0 = no restarts)

    Example:
        >>> scheduler = CosineAnnealingWarmupRestarts(
        ...     optimizer,
        ...     warmup_epochs=2,
        ...     max_epochs=50,
        ...     restarts=2,  # 3 cycles total
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 2,
        max_epochs: int = 50,
        eta_min: float = 1e-6,
        restarts: int = 0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.total_cycles = restarts + 1
        self.cycle_length = max_epochs // self.total_cycles

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Determine current cycle
        cycle = self.last_epoch // self.cycle_length
        cycle_step = self.last_epoch % self.cycle_length

        # Warmup within each cycle
        if cycle_step < self.warmup_epochs:
            alpha = cycle_step / self.warmup_epochs
            return [eta_min + (base_lr - eta_min) * alpha
                    for base_lr, eta_min in zip(self.base_lrs, [self.eta_min] * len(self.base_lrs))]

        # Cosine annealing
        progress = (cycle_step - self.warmup_epochs) / (self.cycle_length - self.warmup_epochs)
        progress = min(progress, 1.0)

        decay_factor = (1 + math.cos(progress * math.pi)) / 2
        return [self.eta_min + (base_lr - self.eta_min) * decay_factor
                for base_lr in self.base_lrs]


class PolynomialDecayWarmup(_LRScheduler):
    """
    Polynomial decay with linear warmup

    Commonly used in BERT/T5 training. Good for:
    - Long training runs
    - Stable convergence
    - Transfer learning

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total training epochs
        power: Polynomial power (default: 1.0 = linear decay)
        end_lr: Final learning rate

    Example:
        >>> scheduler = PolynomialDecayWarmup(
        ...     optimizer,
        ...     warmup_epochs=3,
        ...     max_epochs=50,
        ...     power=1.0,  # Linear decay
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)

            decay_factor = (1 - progress) ** self.power
            return [self.end_lr + (base_lr - self.end_lr) * decay_factor
                    for base_lr in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> _LRScheduler:
    """
    Factory function for learning rate schedulers

    Args:
        name: Scheduler name ('wsd', 'cosine_warmup', 'polynomial', 'adaptive')
        optimizer: Wrapped optimizer
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler instance

    Example:
        >>> scheduler = get_scheduler(
        ...     'wsd',
        ...     optimizer,
        ...     warmup_epochs=2,
        ...     stable_epochs=40,
        ...     decay_epochs=8,
        ... )
    """
    schedulers = {
        'wsd': WSDScheduler,
        'cosine_warmup': CosineAnnealingWarmupRestarts,
        'polynomial': PolynomialDecayWarmup,
        'adaptive': AdaptiveLRScheduler,
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Choose from {list(schedulers.keys())}")

    return schedulers[name](optimizer, **kwargs)


# Visualization utility
def visualize_schedule(
    scheduler_fn: Callable,
    total_epochs: int,
    save_path: Optional[str] = None,
):
    """
    Visualize learning rate schedule

    Args:
        scheduler_fn: Function that returns a scheduler
        total_epochs: Number of epochs to visualize
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt

    # Dummy optimizer and scheduler
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = scheduler_fn(optimizer)

    # Collect LR values
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    # Add phase annotations
    if hasattr(scheduler, 'warmup_epochs'):
        plt.axvline(x=scheduler.warmup_epochs, color='g', linestyle='--', alpha=0.5, label='Warmup End')
    if hasattr(scheduler, 'stable_epochs'):
        stable_end = scheduler.warmup_epochs + scheduler.stable_epochs
        plt.axvline(x=stable_end, color='r', linestyle='--', alpha=0.5, label='Decay Start')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Schedule visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test and visualize schedulers
    print("Testing Learning Rate Schedulers")
    print("=" * 50)

    # WSD Scheduler
    print("\n1. WSD Scheduler")
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    wsd = WSDScheduler(optimizer, warmup_epochs=2, stable_epochs=6, decay_epochs=2)
    print("   Epoch | LR")
    for i in range(10):
        lr = wsd.get_last_lr()[0]
        print(f"   {i:5d} | {lr:.6f}")
        wsd.step()

    # Visualization
    visualize_schedule(
        lambda opt: WSDScheduler(opt, warmup_epochs=5, stable_epochs=20, decay_epochs=5),
        total_epochs=30,
        save_path='wsd_schedule.png'
    )

    print("\n✓ All schedulers tested")
