"""Generate a sequence of tokens from a trained model."""
import tiktoken
import torch
from torch import nn

from layers import GPTModel, Config

torch.manual_seed(123)


def get_next_tokens(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    """
    Generate the next tokens in the input sequence by selecting the token with the highest probability
    at each iteration (greedy decoding).

    Args:
        model: The trained model.
        idx: The indices of the input sequence.
        max_new_tokens: The maximum number of new tokens to generate.
        context_size: The context size.
    """
    for _ in range(max_new_tokens):
        # crop current context to the context size
        idx_cond = idx[:, -context_size:]
        # generate the next token
        with torch.no_grad():
            logits = model(idx_cond)
        # keep only the last token logits
        logits = logits[:, -1, :]
        # transform logits to probabilities
        # softmax is not needed, it suffices to do argmax on the logits
        probas = torch.softmax(logits, dim=-1)
        # get the index of the most probable token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # concatenate the new token to the input sequence
        idx = torch.cat([idx, idx_next], dim=-1)

    return idx


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    print(f"Input: {start_context}")
    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded).unsqueeze(0)  # shape: (batch_size=1, sequence_length)

    # load the trained GPT-2 model
    cfg = Config()
    cfg.context_len = 256
    model = GPTModel(cfg)
    model.eval()

    # generate the next tokens
    generated = get_next_tokens(model, encoded, max_new_tokens=10, context_size=cfg.context_len)
    generated = tokenizer.decode(generated.squeeze(0).tolist())
    print(f"Generated output: {generated}")