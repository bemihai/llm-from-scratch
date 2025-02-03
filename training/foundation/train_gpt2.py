"""Training loop of the GPT-2 model."""
import importlib
import math

import tiktoken
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler
from torch.utils.data import DataLoader

from src.data import GPTDataset
from src.layers import GPTModel
from src.utils import plot_metrics, get_project_root
from src.utils.generate import get_next_tokens
from src.utils.metrics import ds_cross_entropy

torch.manual_seed(123)


def build_optimizer_and_schedule(train_cfg: DictConfig, **kwargs):
    """Build the optimizer and scheduler based on the training config."""
    _optim = getattr(torch.optim, train_cfg.optimizer.type)
    optimizer = _optim(model.parameters(), **train_cfg.optimizer.params)
    if "lr_scheduler" in train_cfg and train_cfg.lr_scheduler is not None:
        try:
            _scheduler = getattr(torch.optim.lr_scheduler, train_cfg.lr_scheduler.type)
        except AttributeError:
            schedulers_module = importlib.import_module("src.utils.scheduler")
            _scheduler = getattr(schedulers_module, train_cfg.lr_scheduler.type)
        scheduler = _scheduler(optimizer, **train_cfg.lr_scheduler.params, **kwargs)
        return optimizer, scheduler

    return optimizer, None



def train_model(
        model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, optimizer: Optimizer, scheduler: Scheduler | None,
        num_epochs: int, eval_freq: int, start_context: str, tokenizer: tiktoken.Encoding, device: str = "cpu"
):
    """The training loop of the GPT-2 model."""
    train_losses, val_losses, track_tokens_seen, track_lr = [], [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        # set model to training mode
        model.to(device).train()
        # iterate over the training data
        for inputs, targets in train_dl:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # reset the gradients and update the lr
            optimizer.zero_grad()
            global_step += 1

            # compute the cross-entropy loss of the current batch
            logits = model(inputs)
            loss = cross_entropy(logits.flatten(0, 1), targets.flatten())

            # compute the loss gradients
            loss.backward()

            # apply gradient clipping after warmup to avoid exploding gradients
            if epoch > 5:
                clip_grad_norm_(model.parameters(), 1.0)

            # update model weights
            optimizer.step()
            if scheduler:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            track_lr.append(lr)

            tokens_seen += inputs.numel()

            # evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_dl, val_dl, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}: "
                      f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {lr:.6f}")

                # generate a sample from the model for visual inspection
                context_size = model.config.context_len
                generate_sample(model, tokenizer, start_context, context_size, device)

    return train_losses, val_losses, track_tokens_seen, track_lr


def evaluate_model(model: nn.Module, train_dl: DataLoader, val_dl:DataLoader):
    """Evaluate the model on the training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = ds_cross_entropy(train_dl, model)
        val_loss = ds_cross_entropy(val_dl, model)

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
    # load the GPT-2 small config
    cfg = OmegaConf.load(get_project_root() / "src/config/gpt2_small.yaml")
    cfg.model.params.context_len = 256
    cfg.model.params.qkv_bias = False

    tokenizer = tiktoken.get_encoding("gpt2")

    with open("../../data/the-verdict.txt", "r") as f:
        raw_text = f.read()

    # split the data into training and validation sets
    train_size = int(len(raw_text) * cfg.data.train_val_ratio)
    train_data = raw_text[:train_size]
    val_data = raw_text[train_size:]

    # training data loader
    train_dl = DataLoader(
        dataset=GPTDataset(
            text=train_data, tokenizer=tokenizer,
            max_length=cfg.model.params.context_len, stride=cfg.model.params.context_len,
        ),
        **cfg.data.train_dl,
    )

    # validation data loader
    val_dl = DataLoader(
        dataset=GPTDataset(
            text=val_data, tokenizer=tokenizer,
            max_length=cfg.model.params.context_len, stride=cfg.model.params.context_len
        ),
        **cfg.data.val_dl,
    )

    # instantiate the GPT-2 model
    model = GPTModel(cfg.model.params)
    model.to(cfg.device)

    # compute the initial training and validation losses
    with torch.no_grad():
        train_loss = ds_cross_entropy(train_dl, model)
        val_loss = ds_cross_entropy(val_dl, model)

    print(f"Initial training loss: {train_loss:.4f}")
    print(f"Initial validation loss: {val_loss:.4f}")

    # instantiate the optimizer and lr scheduler
    num_training_steps = len(train_dl) * cfg.train.max_epochs
    optimizer, scheduler = build_optimizer_and_schedule(
        cfg.train, warmup_steps = 0.2 * num_training_steps, total_steps=num_training_steps
    )

    # train the model
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.train.max_epochs,
        eval_freq=cfg.train.log_every_n_steps,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        device=cfg.device,
    )

    # plot the training and validation losses
    plot_metrics(cfg.train.max_epochs, train_losses, val_losses, "Cross-Entropy Loss")

    # save the trained model to disk
    torch.save(model.state_dict(), "../../pretrained_models/gpt_small.pth")


