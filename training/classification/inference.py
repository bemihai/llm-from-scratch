"""Inference for the GPT-2 binary classifier."""
import tiktoken
import torch
from torch import nn

from layers import GPTClassifier, GPTConfig


def classify_text(
        text: str, model: nn.Module, tokenizer, device: str = "cpu",
        max_length: int | None = None, pad_token_id: int =50256
) -> str:
    """Classify the input text as spam or not spam."""
    # encode the input text
    inputs = tokenizer.encode(text)
    context_len = model.config.context_len
    # truncate the input if it exceeds the max length
    max_length = min(max_length, context_len) if max_length else context_len
    inputs = inputs[:max_length]
    # pad the input if it is shorter than the max length
    if len(inputs) < max_length:
        inputs += [pad_token_id] * (max_length - len(inputs))
    # add the batch dimension
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)
    # get the prediction
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]  # last token
    predicted = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted == 1 else "not spam"


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")

    # load the fine-tuned GPT-2 binary classifier
    cfg = GPTConfig()
    cfg.context_len = 1024
    cfg.qkv_bias = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTClassifier(cfg, num_classes=2)
    model.load_state_dict(torch.load("../../pretrained_models/gpt_spam_classifier.pth", map_location=device))
    model.eval()

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_text(text_1, model, tokenizer, device, max_length=120))
    print(classify_text(text_2, model, tokenizer, device, max_length=120))


