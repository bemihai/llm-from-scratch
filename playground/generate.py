"""Generate a sequence of tokens from a trained model."""
import tiktoken
import torch

from layers import GPTModel, GPTConfig
from utils.api.openai import download_and_load_gpt2, load_weights_into_gpt
from utils.generate import get_next_tokens

torch.manual_seed(123)


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

    cfg = GPTConfig()
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
    generated = tokenizer.decode(generated.squeeze(0).tolist()).replace("\n", " ")
    print(f"Input: {start_context}")
    print(f"Generated: {generated}")