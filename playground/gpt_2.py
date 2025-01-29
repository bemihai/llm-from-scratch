"""GPT-2 model playground."""
import tiktoken
import torch
from torchsummary import summary

from src.layers import GPTConfig, GPTModel

torch.manual_seed(123)


if __name__ == "__main__":

    # tokenize the input text
    tokenizer = tiktoken.get_encoding("gpt2")
    text_1 = "Every effort moves you"
    text_2 = "Every day holds a"
    # inputs shape: (batch_size=2, sequence_length=4)
    inputs = torch.stack([
        torch.tensor(tokenizer.encode(text_1)),
        torch.tensor(tokenizer.encode(text_2))
    ], dim=0)
    print(f"Input shape: {inputs.shape}")
    print(f"Input tokens: {inputs}")

    # create a GPT-2 model with default configuration
    model = GPTModel(GPTConfig())
    logits = model(inputs)
    # output shape: (batch_size=2, sequence_length=4, vocab_size=50257)
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits}")

    # model summary
    print(summary(model, [[4]]))



