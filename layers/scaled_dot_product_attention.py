"""Self-attention with trainable weights, i.e., scaled dot-product attention."""

import torch
import torch.nn as nn

torch.manual_seed(123)


class SelfAttentionV1(nn.Module):
    """Scaled dot-product self-attention with trainable weights."""

    def __init__(self, input_dim: int, output_dim: int, qkv_bias: bool = False):
        super().__init__()
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the self-attention layer."""
        # Compute key, value, and query vectors
        q = self.W_q(x)
        k = self.W_k(x)
        v =self.W_v(x)
        # Compute the unscaled attention scores
        attn_scores = torch.matmul(q, k.T)
        # Normalize the attention scores
        attn_scores = attn_scores / (k.shape[-1] ** 0.5)
        # Apply softmax to get the attention weights
        attention_weights = torch.softmax(attn_scores, dim=-1)
        # Compute the context vectors
        context_vecs = torch.matmul(attention_weights, v)
        return context_vecs


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

    # Create a self-attention layer
    self_attn = SelfAttentionV1(input_dim=3, output_dim=2)

    # Compute the context vectors
    context_vecs = self_attn(inputs)
    print(context_vecs.shape)
    print(context_vecs)