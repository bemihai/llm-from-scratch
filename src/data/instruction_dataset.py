"""Instruction fine-tuning dataset."""
import tiktoken
import torch
from torch.utils.data import Dataset


def format_input(text: dict[str, str]) -> tuple[str, str]:
    """Format the input text using the Alpaca-style formatting."""
    instruction = (f"Below is an instruction that describes a task. "
                   f"Write a response that appropriately completes the request."
                   f"\n\n### Instruction:\n{text['instruction']}")

    input_text = f"\n\n#### Input:\n{text['input']}" if text["input"] else ""
    response = f"\n\n#### Response:\n{text['output']}"

    return instruction + input_text, response


class InstructionDataset(Dataset):
    """Instruction dataset for fine-tuning GPT-2 models."""

    def __init__(self, data: list[dict], tokenizer: tiktoken.Encoding):
        self.data = data
        self.encoded = []
        # format each data point in the Alpaca-style format and tokenize it
        for text in data:
            form_text, response = format_input(text)
            full_text = form_text + response
            self.encoded.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded[idx]

    def __len__(self):
        return len(self.encoded)


def collate_fn(batch: list, pad_token_id: int = 50256, ignore_index: int = -100, max_length: int | None = None,
               device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the instruction dataset.

    Args:
        batch: The list of encoded text data.
        pad_token_id: The token ID for padding.
        ignore_index: The token ID to ignore during training.
        max_length: The maximum length of the input sequence.
        device: The device to use for the tensors.
    """
    batch_max_len = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        # add an extra padding at the end of the sequence
        item_copy = item.copy()
        item_copy += [pad_token_id]
        # pad the sequence to the maximum length
        padded = item_copy + [pad_token_id] * (batch_max_len - len(item_copy))
        # create the input and target tensors (target is input shifted by one token to the right)
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        # replace all but the last padding token in targets with the ignore index
        # pytorch cross-entropy loss ignores the -100 tokens
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        # truncate the input and target sequences if they exceed the maximum length
        if max_length:
            inputs = inputs[:max_length]
            targets = targets[:max_length]
        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor
