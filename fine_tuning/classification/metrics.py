"""Metrics for the fine-tuned models."""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader


def dl_accuracy(data: DataLoader, model: nn.Module, device: str, num_batches: int | None = None):
    """Compute accuracy for a dataset."""
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


def batch_loss(inputs, labels, model, device):
    """Computes the batch cross-entropy loss of the last output tokens."""
    inputs = inputs.to(device)
    labels = labels.to(device)
    logits = model(inputs)[:, -1, :]
    return cross_entropy(logits, labels)


def dataset_loss(data, model, device, num_batches: int | None = None):
    """Computes the average dataset cross-entropy loss of the last output tokens."""
    loss = 0
    num_batches = min(num_batches, len(data)) if num_batches else len(data)
    for batch_idx, (inputs, labels) in enumerate(data):
        if batch_idx == num_batches:
            break
        b_loss = batch_loss(inputs, labels, model, device)
        loss += b_loss

    return loss / num_batches