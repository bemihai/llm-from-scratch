"""Create Pytorch dataset and dataloaders for the spam dataset."""
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    """
    Pytorch dataset for the spam dataset.

    Args:
        spam_df: The spam dataset as a pandas DataFrame with columns ["label", "text"].
        tokenizer: The tokenizer to use for encoding the text.
        max_length: The maximum length of the input sequence.
        pad_token_id: The padding token id (all encoded will have the same length).
    """

    def __init__(
            self, spam_df: pd.DataFrame, tokenizer: tiktoken.Encoding,
            max_length: int | None = None, pad_token_id: int = 50256
    ):
        self.data = spam_df
        self.encoded = tokenizer.encode_batch(self.data["text"])
        # crop or pad the input sequences to the max length
        self.max_length = max_length if max_length else max(map(len, self.encoded))
        self.encoded = [enc[:self.max_length] for enc in self.encoded]
        self.encoded = [enc + [pad_token_id] * (self.max_length - len(enc)) for enc in self.encoded]

    def __getitem__(self, index):
        text = torch.tensor(self.encoded[index], dtype=torch.long)
        label = torch.tensor(self.data["label"].iloc[index], dtype=torch.long)
        return text, label

    def __len__(self):
        return len(self.data)