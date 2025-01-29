"""LoRA layer implementation."""

import math
import torch
from torch import nn


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.

    Args:
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.
        rank (int): The rank of the LoRA layer.
        alpha (int): The scaling parameter of the LoRA layer.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int):
        super().__init__()
        # define A, B, and alpha
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        # initialize A with kaiming uniform (as a liner layer)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LinerWithLoRA(nn.Module):
    """Linear layer with LoRA."""

    def __init__(self, linear: nn.Module, rank: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: int):
    """Replace all the linear layers in a model with linear LoRA layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinerWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)