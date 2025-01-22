"""Instruction model inference."""
import json
import random

import tiktoken
import torch
from tqdm import tqdm

from layers import Config, GPTModel
from sampler.instruction_dataset import format_input
from training import get_next_tokens

torch.manual_seed(123)


if __name__ == "__main__":

    device = torch.device("mps")
    tokenizer = tiktoken.get_encoding("gpt2")

    # create a test dataset
    with open("../../data/instruction-data.json", "rb") as f:
        data = json.load(f)

    test_data = random.sample(data, 100)

    # load the instruction model
    config = Config(
        vocab_size=50_257,
        context_len=1024,
        embed_dim=1024,
        n_heads=16,
        n_layers=24,
        dropout=0.0,
        qkv_bias=True,
    )
    model = GPTModel(config)
    model.load_state_dict(
        torch.load("../../pretrained_models/instruction_gpt_355M.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # generate responses for the test set
    for i, sample in tqdm(enumerate(test_data)):
        input_text, _ = format_input(sample)
        input_tensor = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
        tokens = get_next_tokens(
            model, input_tensor, max_new_tokens=256, context_size=model.config.context_len, eos_token=50256
        )
        generated = tokenizer.decode(tokens.squeeze(0).tolist())
        response = generated[len(input_text):].replace("#### Response:", "").strip()

        test_data[i]["model_response"] = response

    with open("../../data/instruction-data-generated.json", "w") as file:
        json.dump(test_data, file, indent=4)