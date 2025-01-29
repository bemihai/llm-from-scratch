"""Self-attention with trainable weights (scaled dot-product attention) with causal attention mask and dropout."""

import torch
import torch.nn as nn

torch.manual_seed(123)


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention with trainable weights."""

    def __init__(self, input_dim: int, output_dim: int, qkv_bias: bool = False):
        super().__init__()
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """The forward pass of the self-attention layer."""
        # x has shape (seq_len, input_dim), no batch size
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


class CausalAttention(nn.Module):
    """
    Scaled dot-product self-attention with causal attention mask and dropout.
    See also: torch.nn.functional.scaled_dot_product_attention.
    """

    def __init__(self, input_dim: int, output_dim: int, context_len: int,
                 dropout: float = 0.2, qkv_bias: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Create a buffer to store the causal attention mask
        # buffers are saved in the state dict, but not optimized during training
        # buffers are not returned by model.parameters(), but they are always on the same device as the model
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        """The forward pass of the causal self-attention layer."""
        # x has shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.input_dim, f"Input dimension is incorrect: {input_dim} != {self.input_dim}"

        # Compute key, value, and query vectors
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Compute the unscaled attention scores
        attn_scores = torch.matmul(q, k.transpose(1, 2))  # batch dim (0) is not transposed
        # Mask out the upper triangular part of the attention scores
        # future tokens are masked out by setting them to -inf (and softmax will make them 0)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        # Normalize the masked attention scores
        attn_weights = torch.softmax(attn_scores / (k.shape[-1] ** 0.5), dim=-1)
        # Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # Compute the context vectors
        context_vecs = torch.matmul(attn_weights, v)

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

    ##################################################################################
    # Simplified self-attention mechanism with no trainable weights
    ##################################################################################

    print("Simplified self-attention")
    # Compute the attention scores, i.e., the dot product between the input tokens
    attention_scores = torch.matmul(inputs, inputs.T)

    # Normalize the attention scores, i.e., apply softmax along the last dimension
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Compute context vectors by taking the weighted sum of the input tokens, i.e.,
    # multiply the attention weights by the inputs
    context_vecs = torch.matmul(attention_weights, inputs)

    print(f"Context vectors shape: {context_vecs.shape}")
    print(f"Context vectors: {context_vecs}")

    ##################################################################################
    # Scaled dot product self-attention
    ##################################################################################

    print("\nScaled dot-product self-attention")
    self_attn = SelfAttention(input_dim=3, output_dim=2)
    context_vectors = self_attn(inputs)
    print(f"Context vectors shape: {context_vectors.shape}")
    print(f"Context vectors: {context_vectors}")

    ##################################################################################
    # Causal self-attention
    ##################################################################################

    print("\nCausal self-attention")
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    print(f"Batch shape: {batch.shape}")

    context_len = batch.shape[1]
    causal_attn = CausalAttention(input_dim=3, output_dim=2, context_len=context_len)

    context_vectors = causal_attn(batch)
    print(f"Context vectors shape: {context_vectors.shape}")
    print(f"Context vectors: {context_vectors}")