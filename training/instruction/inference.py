"""Instruction model inference."""
import json
import random

import tiktoken
import torch
from tqdm import tqdm

from layers import GPTConfig, GPTModel
from datasets.instruction_dataset import format_input
from training import get_next_tokens
from utils.api import query_ollama

torch.manual_seed(123)


def evaluate_instructions(json_data, json_key, model_name="phi"):
    """
    Compute the scores for the instruction dataset using an ollama model.
    
    Args:
        json_data: The instruction data as json.
        json_key: The key for the model response.
        model_name: The ollama model name.

    Returns a list of scores from 0 to 100.
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_ollama(prompt, model_name)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score to int: {score}")
            continue

    return scores


if __name__ == "__main__":

    device = torch.device("mps")
    tokenizer = tiktoken.get_encoding("gpt2")

    # create a test dataset
    with open("../../data/instruction-data.json", "rb") as f:
        data = json.load(f)

    test_data = random.sample(data, 100)

    # load the instruction model
    config = GPTConfig(
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

    scores = evaluate_instructions(test_data[:10], "model_response", model_name="llama3")
    print(f"Average score: {sum(scores) / len(scores)}")