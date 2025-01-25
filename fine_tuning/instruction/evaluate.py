"""Evaluate the GPT-2 instruction model using a large model from ollama."""
import json

import requests
from tqdm import tqdm

from sampler.instruction_dataset import format_input


def query_ollama(prompt: str, model_name: str = "phi") -> str:
    """Query an ollama model API."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048}
    }
    data = json.dumps(data).encode("utf-8")

    url = "http://localhost:11434/api/chat"
    res = requests.post(url, data=data, headers={"Content-Type": "application/json"})
    res.raise_for_status()

    response = ""
    content = res.content.decode("utf-8").splitlines()
    for line in content:
        response += json.loads(line)["message"]["content"]

    return response

def evaluate_test_data(json_data, json_key, model="phi"):
    """Generate scores for the model responses."""
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_ollama(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score to int: {score}")
            continue

    return scores


if __name__ == "__main__":

    with open("../../data/instruction-data-generated.json", "r") as file:
        test_data = json.load(file)

    scores = evaluate_test_data(test_data[:10], "model_response", model="llama3")
    print(f"Average score: {sum(scores) / len(scores)}")



