"""Ollama client utils."""
import json

import requests


def query_ollama(
        prompt: str, model_name: str = "phi",
        temp: int = 0, context_len: int = 2048, seed: int = 123
) -> str:
    """Query an ollama model API."""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": seed, "temperature": temp, "num_ctx": context_len}
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