"""Multi-head causal attention layer."""

import torch
from torch import nn

from playground.self_attention import CausalAttention

from layers.attention import MultiHeadAttention

torch.manual_seed(123)


class MultiHeadAttentionSequential(nn.Module):
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
    multihead_attn = MultiHeadAttentionSequential(input_dim=3, output_dim=2, context_len=context_len, num_heads=2)

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