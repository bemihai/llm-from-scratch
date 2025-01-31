"""Metrics for the fine-tuned models."""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader


def ds_accuracy(data: DataLoader, model: nn.Module, num_batches: int | None = None):
    """
    Computes the binary classifier accuracy for a dataset.

    Args:
        data: The dataset to evaluate.
        model: The model to evaluate.
        num_batches: Optional, the number of batches to evaluate.

    Returns the accuracy of the model on the dataset.
    """
    device = next(model.parameters()).device
    model.eval()
    correct, num_items = 0, 0
    num_batches = min(num_batches, len(data)) if num_batches else len(data)

    for batch_idx, (inputs, labels) in enumerate(data):
        if batch_idx == num_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(inputs)[:, -1, :]  # last token
        predictions = torch.argmax(logits, dim=-1)
        num_items += predictions.shape[0]
        correct += (predictions == labels).sum().item()

    return correct/ num_items


def ds_cross_entropy(data: DataLoader, model: nn.Module, strategy: str = "all", num_batches: int | None = None):
    """
    Computes the average cross-entropy loss on a dataset.

    Args:
        data: The data loader.
        model: The model to evaluate.
        strategy: The strategy to use for computing the loss, i.e. which tokens to use in the computations.
            Should be one of ["first", "last", "all"].
        num_batches: Optional, the number of batches to evaluate.

    Returns the average cross-entropy loss on the dataset.
    """
    device = next(model.parameters()).device
    total_loss = 0
    num_batches = min(num_batches, len(data)) if num_batches else len(data)
    for i, (inputs, targets) in enumerate(data):
        if i > num_batches:
            break
        inputs = inputs.to(device)
        logits = model(inputs)
        match strategy:
            case "first":
                loss = cross_entropy(logits[:, 0, :], targets.to(device))
            case "last":
                loss = cross_entropy(logits[:, -1, :], targets.to(device))
            case "all":
                loss = cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
            case _:
                raise ValueError(f"Invalid strategy: {strategy}")
        total_loss += loss.item()

    return total_loss / num_batches




