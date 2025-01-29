from typing import Any

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
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






