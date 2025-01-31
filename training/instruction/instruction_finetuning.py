"""Fine tune a pre-trained GPT-2 model on an instruction dataset."""

import json
from functools import partial

import tiktoken
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.layers import GPTConfig, GPTModel
from src.utils.api import download_and_load_gpt2, load_weights_into_gpt
from src.data import collate_fn, InstructionDataset, format_input
from src.utils.metrics import ds_cross_entropy
from training.foundation.train_gpt2 import train_model
from src.utils import plot_metrics

torch.manual_seed(123)


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("mps")

    with open("../../data/instruction-data.json", "rb") as f:
        data = json.load(f)

    # split the data into training and validation sets
    train_val_ratio = 0.85
    train_data = data[:int(len(data) * train_val_ratio)]
    val_data = data[int(len(data) * train_val_ratio):]

    # create the training and validation dataloaders
    _collate_fn = partial(collate_fn, device=device, max_length=1024)
    num_workers = 0
    batch_size = 8
    train_loader = DataLoader(
        dataset=InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        drop_last=False
    )

    print("Train loader shapes:")
    for inputs, targets in train_loader:
        print(f"Input: {inputs.shape}, Targets: {targets.shape}")
        break

    # load GPT-2 medium
    settings, params = download_and_load_gpt2(model_size="355M", models_dir="../../pretrained_models")
    config = GPTConfig(
        vocab_size=50_257,
        context_len=1024,
        embed_dim=1024,
        n_heads=16,
        n_layers=24,
        dropout=0.0,
        qkv_bias=True,
    )
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.eval()
    # print(summary(model, [[120]]))

    # compute the initial training and validation loss
    with torch.no_grad():
        train_loss = ds_cross_entropy(train_loader, model, device, num_batches=5)
        val_loss = ds_cross_entropy(val_loader, model, device, num_batches=5)
        print(f"Initial train loss: {train_loss:.2f}, Initial val loss: {val_loss:.2f}")

    # fine-tune the model
    num_epochs = 2
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        eval_freq=5,
        start_context=format_input(val_data[0])[0],
        tokenizer=tokenizer,
        warmup_steps=0,
        initial_lr=5e-5,
        min_lr=5e-5,
    )

    # plot the training and validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_metrics(epochs_tensor, tokens_seen, train_losses, val_losses)

    # save the trained model to disk
    torch.save(model.state_dict(), "../../pretrained_models/instruction_gpt_355M.pth")





