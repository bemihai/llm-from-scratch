"""Generate a sequence of tokens from a trained model."""
import tiktoken
import torch
from torch import nn

from layers import GPTModel, Config
from openai import download_and_load_gpt2, load_weights_into_gpt

torch.manual_seed(123)


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


if __name__ == "__main__":

    # load the locally trained GPT-2 model
    # cfg = Config()
    # cfg.context_len = 256
    # model = GPTModel(cfg)
    # model.load_state_dict(torch.load("gpt2/gpt_small.pth", map_location="cpu"))
    # model.eval()

    # load the trained GPT-2 124M model from Open AI
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print(f"GPT-2 124M config: {settings}")

    cfg = Config()
    cfg.context_len = 1024
    cfg.qkv_bias = True

    model = GPTModel(cfg)
    load_weights_into_gpt(model, params)

    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Having two cups of coffee a day can help you"
    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded).unsqueeze(0)  # shape: (batch_size=1, sequence_length)

    # generate the next tokens
    generated = get_next_tokens(
        model, encoded, max_new_tokens=25, context_size=cfg.context_len, temperature=2.5, top_k=50
    )
    generated = tokenizer.decode(generated.squeeze(0).tolist())
    print(f"Input: {start_context}")
    print(f"Generated: {generated.replace("\n", " ")}")