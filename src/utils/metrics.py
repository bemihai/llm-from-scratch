"""Metrics for the fine-tuned models."""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

# TODO: refactor this module


def dataset_accuracy(data: DataLoader, model: nn.Module, device: str, num_batches: int | None = None):
    """Compute the binary classifier accuracy for a dataset."""
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


def batch_ce_loss_all(inputs: torch.Tensor, targets:torch.Tensor, model: nn.Module, device: str = "cpu"):
    """Computes the cross-entropy loss on a single batch."""
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    loss = cross_entropy(logits.flatten(0, 1), targets.flatten())

    return loss


def dataset_ce_loss_all(dl: DataLoader, model: nn.Module, device: str = "cpu", num_batches: int | None = None):
    """Computes the average cross-entropy loss on a data loader."""
    total_loss = 0
    num_batches = min(num_batches, len(dl)) if num_batches else len(dl)
    for i, (inputs, targets) in enumerate(dl):
        if i > num_batches:
            break
        loss = batch_ce_loss_all(inputs, targets, model, device)
        total_loss += loss.item()

    return total_loss / len(dl)


def batch_ce_loss_last(inputs, labels, model, device):
    """Computes the batch cross-entropy loss of the last output tokens."""
    inputs = inputs.to(device)
    labels = labels.to(device)
    logits = model(inputs)[:, -1, :]
    return cross_entropy(logits, labels)


def dataset_ce_loss_last(data, model, device, num_batches: int | None = None):
    """Computes the average dataset cross-entropy loss of the last output tokens."""
    loss = 0
    num_batches = min(num_batches, len(data)) if num_batches else len(data)
    for batch_idx, (inputs, labels) in enumerate(data):
        if batch_idx == num_batches:
            break
        b_loss = batch_ce_loss_last(inputs, labels, model, device)
        loss += b_loss

    return loss / num_batches