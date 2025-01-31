"""Utility functions for plotting."""
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator


def plot_metrics(
        num_epochs: int, train_metrics: list[float], val_metrics: list[float], metric_name: str,
):
    """
    Plots the training and validation metrics.

    Args:
        num_epochs: The number of epochs.
        train_metrics: The list of training metrics.
        val_metrics: The list of validation metrics.
        metric_name: The name of the metric.
    """
    epochs = torch.linspace(0, num_epochs, len(train_metrics))
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs, train_metrics, label=f"Training {metric_name}")
    ax1.plot(epochs, val_metrics, linestyle="-.", label=f"Validation {metric_name}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(metric_name)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    plt.show()

