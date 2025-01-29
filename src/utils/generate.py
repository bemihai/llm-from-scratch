"""Text generation utilities for GPT-2."""

import torch
from torch import nn


def get_next_tokens(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int,
                    temperature: float = 0.0, top_k: int | None = None, eos_token: int | None = None):
    """
    Generate the next tokens in the input sequence by selecting the token with respect to the decoding strategy.

    Args:
        model: The trained model.
        idx: The indices of the input sequence.
        max_new_tokens: The maximum number of new tokens to generate.
        context_size: The context size.
        temperature: The temperature to use for temperature scaling.
            - if temperature = 0, select the token with the highest probability at each iteration (greedy decoding).
            - if temperature > 0, sample the token from the temperature-scaled probabilities (temp > 1 leads to
                more diverse samples, temp < 1 leads to more confident samples).
        top_k: The number of top-k tokens to sample from when using temp scaling.
        eos_token: The end-of-sequence token.
    """
    for _ in range(max_new_tokens):
        # crop current context to the context size
        idx_cond = idx[:, -context_size:]
        # generate the next token
        with torch.no_grad():
            logits = model(idx_cond)
        # keep only the last token logits
        logits = logits[:, -1, :]
        # transform logits to probabilities and sample the next token
        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        if temperature > 0:
            scaled_logits = logits / temperature
            probas = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # stop early if the end-of-sequence token is encountered
        if idx_next == eos_token:
            break

        # concatenate the new token to the input sequence
        idx = torch.cat([idx, idx_next], dim=-1)

    return idx
