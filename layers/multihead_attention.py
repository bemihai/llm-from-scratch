"""Multi-head causal attention layer."""

import torch
from torch import nn

from layers.self_attention import CausalAttention

torch.manual_seed(123)


class MultiHeadAttentionV1(nn.Module):
    """Sequential multi-head causal attention layer."""

    def __init__(self, input_dim: int, output_dim: int, context_len: int, num_heads: int,
                 dropout: float = 0.2, qkv_bias: bool = False):
        super().__init__()
        # Create a list of causal attention heads
        self.heads = nn.ModuleList([
            CausalAttention(input_dim, output_dim, context_len, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # concatenate the output of each head along the last dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head causal attention witt weight splits"""

    def __init__(self, input_dim: int, output_dim: int, context_len: int, num_heads: int,
                 dropout: float = 0.2, qkv_bias: bool = False):
        super().__init__()
        assert output_dim % num_heads == 0, "Output dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.input_dim = input_dim
        # reduces the projection dimension to match the number of heads
        self.head_dim = output_dim // num_heads

        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)

        # linear layer to combine the head outputs
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        """The forward pass of the multi-head attention layer."""
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.input_dim, f"Input dimension is incorrect: {input_dim} != {self.input_dim}"

        # Project the input to the query, key, and value vectors
        # shape: (batch_size, seq_len, output_dim)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

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
        context_vecs = context_vecs.contiguous().view(batch_size, seq_len, self.output_dim)
        return self.output_proj(context_vecs)


if __name__ == '__main__':

    # Assume we have 6 input tokens embedded as 3-dim vectors
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    ##################################################################################
    # Multi-head causal attention V1
    ##################################################################################

    print("\nMulti-head causal attention V1")
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    print(f"Batch shape: {batch.shape}")

    context_len = batch.shape[1]
    multihead_attn = MultiHeadAttentionV1(input_dim=3, output_dim=2, context_len=context_len, num_heads=2)

    context_vectors = multihead_attn(batch)
    print(f"Context vectors shape: {context_vectors.shape}")
    print(f"Context vectors: {context_vectors}")


    ##################################################################################
    # Multi-head causal attention with weight splits
    ##################################################################################

    print("\nMulti-head causal attention with weight splits")
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    print(f"Batch shape: {batch.shape}")

    context_len = batch.shape[1]
    multihead_attn = MultiHeadAttention(input_dim=3, output_dim=2, context_len=context_len, num_heads=2)

    context_vectors = multihead_attn(batch)
    print(f"Context vectors shape: {context_vectors.shape}")
    print(f"Context vectors: {context_vectors}")