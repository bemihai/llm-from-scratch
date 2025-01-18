from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """PyTorch dataset for loading the text data and encoding it into integers."""

    def __init__(self, text: str, tokenizer: Any, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_seq = token_ids[i:i + max_length]
            target_seq = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(input_seq)
            self.target_ids.append(target_seq)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])


def get_dataloader_v1(
        text: str, tokenizer: Any, batch_size: int = 8, max_length: int = 128, stride: int = 128,
        shuffle: bool = True, drop_last: bool = True, num_workers: int = 0, **kwargs
) -> DataLoader:
    """Returns a PyTorch DataLoader for the GPTDatasetV1 dataset."""
    ds = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    output_size = 256

    # Create an embedding layer of dimension: vocab_size x output_size
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_size)

    with open("/Users/bemihai/projects/llm-from-scratch/data/the-verdict.txt", "r") as f:
        raw_text = f.read()

    dataloader = get_dataloader_v1(raw_text, tokenizer)

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
