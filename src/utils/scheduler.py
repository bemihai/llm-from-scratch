"""Learning rate scheduler implementations."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Linearly increases the learning rate from `initial_lr` to the peak lr
    over the initial `warmup_steps` training steps. Then decreases the
    learning rate from the peak lr to `min_lr` over the remaining step
    following a cosine curve.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int, min_lr: float, initial_lr: float,
                 total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.peak_lr = optimizer.param_groups[0]["lr"]
        self.initial_lr = initial_lr
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps if self.warmup_steps > 0 else 0
        # linear warmup
        if step < self.warmup_steps:
            return self.initial_lr + lr_increment * step
        # cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))