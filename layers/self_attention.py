"""Simplified self-attention mechanism with no trainable weights."""

import torch

# Assume we have 6 input tokens embedded as 3-dim vectors
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]]
)

# Compute the attention scores, i.e., the dot product between the input tokens
attention_scores = torch.matmul(inputs, inputs.T)

# Normalize the attention scores, i.e., apply softmax along the last dimension
attention_weights = torch.softmax(attention_scores, dim=-1)

# Compute context vectors by taking the weighted sum of the input tokens, i.e.,
# multiply the attention weights by the inputs
context_vecs = torch.matmul(attention_weights, inputs)

print(context_vecs)
assert inputs.shape == context_vecs.shape
