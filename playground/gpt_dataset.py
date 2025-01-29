import torch
from torch.utils.data import DataLoader
import tiktoken

from src.data import GPTDataset


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    output_size = 256

    # Create an embedding layer of dimension: vocab_size x output_size
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_size)

    with open("/Users/bemihai/projects/llm-from-scratch/data/the-verdict.txt", "r") as f:
        raw_text = f.read()

    ds = GPTDataset(raw_text, tokenizer, 128, 128)
    dataloader = DataLoader(
        dataset=ds,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Get the first batch of data
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    # Create input embeddings
    input_embeddings = token_embedding_layer(inputs)
    print(f"Input embeddings shape: {input_embeddings.shape}")

    # Create a positional embedding layer (same shape as the input embeddings)
    context_length = 128  # same as max_length in the dataset
    pos_embedding_layer = torch.nn.Embedding(context_length, output_size)
    pos_embedding_layer = pos_embedding_layer(torch.arange(context_length))
    print(f"Positional embeddings shape: {pos_embedding_layer.shape}")

    # Add the positional embeddings to the input embeddings
    input_embeddings = input_embeddings + pos_embedding_layer
    print(f"Input embeddings shape: {input_embeddings.shape}")
