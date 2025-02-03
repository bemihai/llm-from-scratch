"""Learning rate scheduler implementations."""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Linearly increases the learning rate from `initial_lr` to the peak lr
    over the initial `warmup_steps` training steps. Then decreases the
    learning rate from the peak lr to `min_lr` over the remaining step
    following a cosine curve.
    """

    def __init__(
            self, optimizer: torch.optim.Optimizer,
            warmup_steps: int, min_lr: float, start_lr: float, total_steps: int, last_epoch: int = -1
        ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.peak_lr = optimizer.param_groups[0]["lr"]
        self.initial_lr = start_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps if self.warmup_steps > 0 else 0
        # linear warmup
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lr + lr_increment * self.last_epoch]
        # cosine decay
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))]