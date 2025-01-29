"""Multi-head causal attention layer."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal attention with weight splits (see also torch.nn.MultiheadAttention).
    """

    def __init__(self, input_dim: int, output_dim: int, context_len: int, num_heads: int,
                 dropout: float = 0.2, qkv_bias: bool = False):
        super().__init__()
        assert output_dim % num_heads == 0, "Output dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.d_out = output_dim
        self.d_in = input_dim
        # reduces the projection dimension to match the number of heads
        self.head_dim = output_dim // num_heads

        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)

        # linear layer to combine the head outputs
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        """The forward pass of the multi-head attention layer."""
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.d_in, f"Input dimension is incorrect: {input_dim} != {self.d_in}"

        # Project the input to the query, key, and value vectors
        # shape: (batch_size, seq_len, output_dim)
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # Split the query, key, and value vectors into multiple heads: add num_heads dim and split the output_dim
        # shape: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose the dimensions to perform the attention operation
        # shape: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute the unscaled attention scores
        # shape: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        mask_bool = self.mask[:seq_len, :seq_len].bool()
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # Normalize the attention scores and apply dropout
        # shape: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores / (k.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the context vectors and switch back the dimensions
        # shape: (batch_size, seq_len, num_heads, head_dim)
        context_vecs = torch.matmul(attn_weights, v).transpose(1, 2)

        # Combine the heads and project the output
        # shape: (batch_size, seq_len, output_dim)
        context_vecs = context_vecs.contiguous().view(batch_size, seq_len, self.d_out)
        return self.out_proj(context_vecs)

