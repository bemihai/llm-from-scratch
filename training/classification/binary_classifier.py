"""Fine-tune a pre-trained GPT-2 model on a binary classification task."""

import pandas as pd
import tiktoken
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary

from src.utils.metrics import batch_ce_loss_last, dataset_ce_loss_last, dataset_accuracy
from src.layers import GPTConfig, GPTClassifier, replace_linear_with_lora
from src.utils.api import load_weights_into_gpt, download_and_load_gpt2
from src.data import SpamDataset, balanced_dataset

torch.manual_seed(123)


def train_model(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, optimizer: Optimizer,
                num_epochs: int, eval_freq: int, eval_iter: int, device: str = "cpu"):
    """The training loop of the GPT-2 classifier."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    data_count, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_dl:
            optimizer.zero_grad()
            global_step += 1

            loss = batch_ce_loss_last(inputs, labels, model, device)
            loss.backward()
            optimizer.step()
            data_count += inputs.shape[0]

            if global_step % eval_freq == 0:
                with torch.no_grad():
                    train_loss = dataset_ce_loss_last(train_dl, model, device, num_batches=eval_iter)
                    val_loss = dataset_ce_loss_last(val_dl, model, device, num_batches=eval_iter)
                    train_acc = dataset_accuracy(train_dl, model, device, num_batches=eval_iter)
                    val_acc = dataset_accuracy(val_dl, model, device, num_batches=eval_iter)
                    print(f"Epoch {epoch}, Step {global_step}, "
                          f"Train loss: {train_loss:.2f}, Val loss: {val_loss:.2f}, "
                          f"Train acc: {train_acc * 100:.2f}%, Val acc: {val_acc * 100:.2f}%")
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)

    return train_losses, val_losses, train_acc, val_acc, data_count


def get_model_ready_for_ft(_model: nn.Module, lora: bool = False, rank: int | None = None, alpha: int | None = None):
    """
    Sets the fine-tuning strategy for the GPT-2 binary classifier - either the last layers or LoRA.

    Args:
        _model: The pre-trained GPT-2 model.
        lora: Whether to use the LoRA fine-tuning strategy.
        rank: The rank of the LoRA fine-tuning strategy.
        alpha: The alpha parameter of the LoRA fine-tuning strategy.
    """
    # freeze all the layers of the model
    for param in _model.parameters():
        param.requires_grad = False

    # fine tune the last layers of the model: the last transformer block, the final layer norm, and
    # the original output head
    if not lora:
        for param in _model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in _model.final_norm.parameters():
            param.requires_grad = True
        for param in _model.out_head.parameters():
            param.requires_grad = True
    # fine tune the model with LoRA
    else:
        assert rank and alpha, "Rank and alpha parameters are required for LoRA fine-tuning."
        replace_linear_with_lora(_model, rank, alpha)

    return _model


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")

    # load the spam dataset
    raw_df = pd.read_csv("../../data/spam/spam_collection.tsv", sep="\t", header=None, names=["label", "text"])
    df = balanced_dataset(raw_df)

    # split the dataset into training, validation, and test sets
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # shuffle the dataset
    ds_ratios = [0.7, 0.1, 0.2]  # train, val, test
    train_end_idx = int(ds_ratios[0] * len(df))
    val_end_idx = train_end_idx + int(ds_ratios[1] * len(df))
    train_df = df[:train_end_idx]
    val_df = df[train_end_idx:val_end_idx]
    test_df = df[val_end_idx:]
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # create data and dataloaders
    train_ds = SpamDataset(train_df, tokenizer)
    val_ds = SpamDataset(val_df, tokenizer)
    test_ds = SpamDataset(test_df, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=8, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=8, drop_last=False)

    # instantiate the GPT-2 classifier model
    cfg = GPTConfig()
    cfg.context_len = 1024
    cfg.qkv_bias = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTClassifier(cfg, num_classes=2)

    # load the pre-trained GPT-2 weights
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="../../pretrained_models")
    load_weights_into_gpt(model, params)
    model.to(device)

    # test the model on a sample input
    model.eval()
    inputs = tokenizer.encode("Do you have time for a beer?")
    inputs = torch.tensor(inputs).unsqueeze(0)
    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")

    # we only need to train the last output token as it is the only one attending on all the other tokens
    probs = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probs)
    print(f"Predicted class probs: {probs}")
    print(f"Predicted label: {label.item()}")

    # compute the initial training set loss and accuracy before fine-tuning
    # with torch.no_grad():
    #     train_loss = dataset_loss(train_dl, model, device, num_batches=10)
    # print(f"Initial training loss: {train_loss:.2f}")
    # train_accuracy = dl_accuracy(train_dl, model, device, num_batches=10)
    # print(f"Initial training accuracy: {train_accuracy * 100:.2f}%")

    # fine-tune the model with LoRA layers
    ft_model = get_model_ready_for_ft(model, lora=True, rank=16, alpha=16)

    # train the model on the spam dataset
    num_epochs = 5
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=5e-5, weight_decay=0.1)

    # print model summary
    summary(ft_model, input_size=[[120]])

    print(f"Training the classification model for {num_epochs} epochs...")
    train_losses, val_losses, train_acc, val_acc, data_count = train_model(
        ft_model, train_dl, val_dl, optimizer, num_epochs, eval_freq=50, eval_iter=5, device=device
    )

    # compute the training, validation, and test accuracies
    train_accuracy = dataset_accuracy(train_dl, ft_model, device)
    val_accuracy = dataset_accuracy(val_dl, ft_model, device)
    test_accuracy = dataset_accuracy(test_dl, ft_model, device)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # save the trained model to disk
    torch.save(ft_model.state_dict(), "../../pretrained_models/gpt_spam_classifier_lora.pth")




