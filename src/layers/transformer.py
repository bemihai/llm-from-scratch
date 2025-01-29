"""Implementation of the Transformer block used in GPT-like models."""
import torch
from torch import nn

from src.layers.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """
    The layer normalization architecture (see also torch.nn.LayerNorm).

    Layer normalization improves the stability and efficiency of the model training by speeding up the convergence
    to effective weights.
    In modern LLMs, layer normalization is typically applied before and after the multi-head attention module.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function (see also torch.nn.GELU).

    GELU(x) = x * Phi(x), where Phi(x) is the cumulative distribution function of the standard normal distribution.
    GELU is a smooth, non-linear approximation of the ReLU function and is used in many modern LLMs. It does have
    a non-zero gradient for almost all negative values.

    The implementation is based on the approximation of the GELU function found by curve fitting.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """
    Feed forward neural network module.
    The feed forward module plays a crucial role in enhancing the model’s ability to learn
    from and generalize the data.
    """
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        # the default hidden dimension is 4 times the embedding dimension
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """The transformer block architecture (see also torch.nn.Transformer)."""

    def __init__(self, embed_dim: int, context_len: int, n_heads: int, dropout: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        # the multi-head attention module
        self.att = MultiHeadAttention(
            embed_dim, embed_dim, context_len, n_heads, dropout, qkv_bias=qkv_bias
        )
        # the feed forward module
        self.ff = FeedForward(embed_dim)
        # layer normalization modules
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        # dropout layer
        self.drop_shortcut = nn.Dropout(dropout)


    def forward(self, x):
        """The forward pass of the transformer block."""
        # Skip connections help gradients flow through the network during training and improves the
        # learning (avoiding vanishing gradients)
        skip = x
        x = self.norm1(x)  # apply layer normalization
        x = self.att(x)  # apply multi-head attention
        x = self.drop_shortcut(x)  # apply dropout
        x = x + skip  # add the skip connection

        skip = x
        x = self.norm2(x)  # apply layer normalization
        x = self.ff(x)  # apply feed forward
        x = self.drop_shortcut(x)  # apply dropout
        x = x + skip  # add the skip connection

        return x