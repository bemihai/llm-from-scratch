"""UI for the spam classifier model."""
from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from layers import Config, GPTClassifier
from fine_tuning.classification import classify_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """Load the model and tokenizer for the spam classifier."""

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path("") / "training/gpt2/gpt_spam_classifier.pth"
    if not model_path.exists():
        print(f"Could not find the {model_path} file.")
        sys.exit()

    # load the fine-tuned GPT-2 binary classifier
    cfg = Config()
    cfg.context_len = 1024
    cfg.qkv_bias = True
    model = GPTClassifier(cfg, num_classes=2)

    # Then load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """The main Chainlit function."""
    user_input = message.content

    label = classify_text(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # spam or not spam
    ).send()