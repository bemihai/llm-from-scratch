"""Training loop of the GPT-2 model."""
import math

import tiktoken
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader

from layers import Config, GPTModel
from sampler import get_dataloader_v1
from training import dl_ce_loss, batch_ce_loss, get_next_tokens
from utils import plot_losses

torch.manual_seed(123)


def train_model(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, optimizer: Optimizer,
                num_epochs: int, eval_freq: int, initial_lr: float, min_lr: float, warmup_steps: int,
                start_context: str, tokenizer: tiktoken.Encoding, device: str = "cpu"):
    """The training loop of the GPT-2 model."""
    train_losses, val_losses, track_tokens_seen, track_lr = [], [], [], []
    tokens_seen, global_step = 0, -1

    # lr increment per step during warmup
    total_steps = len(train_dl) * num_epochs
    peak_lr = optimizer.param_groups[0]["lr"]
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(num_epochs):
        # set model to training mode
        model.to(device).train()
        # iterate over the training data
        for inputs, targets in train_dl:
            # reset the gradients and update the lr
            optimizer.zero_grad()
            global_step += 1
            # linear warmup for lr followed by cosine decay
            if global_step < warmup_steps:
                lr = initial_lr + lr_increment * global_step
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
            track_lr.append(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # compute the loss of the current batch
            loss = batch_ce_loss(inputs, targets, model, device)
            # compute the loss gradients
            loss.backward()
            # apply gradient clipping after warmup to avoid exploding gradients
            if global_step > warmup_steps:
                clip_grad_norm_(model.parameters(), 1.0)
            # update model weights
            optimizer.step()
            tokens_seen += inputs.numel()

            # evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_dl, val_dl, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}: "
                      f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {lr:.4f}")

        # generate a sample from the model for visual inspection
        context_size = model.position_embedding.weight.shape[0]
        generate_sample(model, tokenizer, start_context, context_size, device)

    return train_losses, val_losses, track_tokens_seen, track_lr


def evaluate_model(model: nn.Module, train_dl: DataLoader, val_dl:DataLoader, device: str = "cpu"):
    """Evaluate the model on the training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = dl_ce_loss(train_dl, model, device)
        val_loss = dl_ce_loss(val_dl, model, device)

    model.train()
    return train_loss, val_loss


def generate_sample(model: nn.Module, tokenizer: tiktoken.Encoding,
                    start_context: str, context_size: int, device: str = "cpu"):
    """Generate a sample from the model."""
    model.eval()
    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = get_next_tokens(model, encoded, max_new_tokens=20, context_size=context_size)

    generated = tokenizer.decode(generated.squeeze(0).tolist())
    print(generated.replace("\n", " "))
    model.train()


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")
    gpt_cfg = Config()
    gpt_cfg.context_len = 256

    with open("/Users/bemihai/projects/llm-from-scratch/data/the-verdict.txt", "r") as f:
        raw_text = f.read()

    # split the data into training and validation sets
    train_val_ratio = 0.9
    train_size = int(len(raw_text) * train_val_ratio)
    train_data = raw_text[:train_size]
    val_data = raw_text[train_size:]

    # training data loader
    train_dl = get_dataloader_v1(
        text=train_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=gpt_cfg.context_len,
        stride=gpt_cfg.context_len,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # validation data loader
    val_dl = get_dataloader_v1(
        text=val_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=gpt_cfg.context_len,
        stride=gpt_cfg.context_len,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # instantiate the GPT-2 model
    model = GPTModel(gpt_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # compute the initial training and validation losses
    with torch.no_grad():
        train_loss = dl_ce_loss(train_dl, model, device)
        val_loss = dl_ce_loss(val_dl, model, device)

    print(f"Initial training loss: {train_loss:.4f}")
    print(f"Initial validation loss: {val_loss:.4f}")

    # set training parameters
    num_epochs = 10
    initial_lr = 0.004
    min_lr = 0.0004
    total_steps = len(train_dl) * num_epochs
    warmup_steps = int(0.05 * total_steps)  # 20% steps for learning rate warmup

    # instantiate the optimizer
    # AdamW is a variant of Adam that improves the weight decay approach (better regularization)
    optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # train the model
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_freq=5,
        initial_lr=initial_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        device=device,
    )

    # plot the training and validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # save the trained model to disk
    torch.save(model.state_dict(), "gpt2/gpt_small.pth")


