"""Fine-tune a pre-trained GPT-2 model on a binary classification task."""

import pandas as pd
import tiktoken
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchsummary import summary

from fine_tunning.classification.metrics import dl_accuracy, dataset_loss, batch_loss
from layers import Config, GPTClassifier
from openai import load_weights_into_gpt, download_and_load_gpt2
from sampler import SpamDataset

torch.manual_seed(123)


def balanced_dataset(df: pd.DataFrame):
    """Creates a balanced dataset by undersampling the majority class."""
    num_spam = df["label"].value_counts()["spam"]
    spam_sampled = df.query("label == 'spam'")
    ham_sampled = df.query("label == 'ham'").sample(n=num_spam, random_state=123)
    bdf = pd.concat([spam_sampled, ham_sampled])
    bdf["label"] = bdf["label"].map({"spam": 1, "ham": 0})

    return bdf


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

            loss = batch_loss(inputs, labels, model, device)
            loss.backward()
            optimizer.step()
            data_count += inputs.shape[0]

            if global_step % eval_freq == 0:
                with torch.no_grad():
                    train_loss = dataset_loss(train_dl, model, device, num_batches=eval_iter)
                    val_loss = dataset_loss(val_dl, model, device, num_batches=eval_iter)
                    train_acc = dl_accuracy(train_dl, model, device, num_batches=eval_iter)
                    val_acc = dl_accuracy(val_dl, model, device, num_batches=eval_iter)
                    print(f"Epoch {epoch}, Step {global_step}, "
                          f"Train loss: {train_loss:.2f}, Val loss: {val_loss:.2f}, "
                          f"Train acc: {train_acc * 100:.2f}%, Val acc: {val_acc * 100:.2f}%")
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)

    return train_losses, val_losses, train_acc, val_acc, data_count


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

    # create datasets and dataloaders
    train_ds = SpamDataset(train_df, tokenizer)
    val_ds = SpamDataset(val_df, tokenizer)
    test_ds = SpamDataset(test_df, tokenizer)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=8, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=8, drop_last=False)

    # instantiate the GPT-2 classifier model
    cfg = Config()
    cfg.context_len = 1024
    cfg.qkv_bias = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTClassifier(cfg, num_classes=2)

    # load the pre-trained GPT-2 weights
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="../../training/gpt2")
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
    with torch.no_grad():
        train_loss = dataset_loss(train_dl, model, device, num_batches=10)
    print(f"Initial training loss: {train_loss:.2f}")
    train_accuracy = dl_accuracy(train_dl, model, device, num_batches=10)
    print(f"Initial training accuracy: {train_accuracy * 100:.2f}%")

    # freeze all the layers except the last transformer block, the final layer norm, and
    # the original output head (these are fine-tuned for classification)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    # train the model on the spam dataset
    num_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # print model summary
    summary(model, input_size=[[120]])

    print(f"Training the classification model for {num_epochs} epochs...")
    train_losses, val_losses, train_acc, val_acc, data_count = train_model(
        model, train_dl, val_dl, optimizer, num_epochs, eval_freq=50, eval_iter=5, device=device
    )

    # compute the training, validation, and test accuracies
    train_accuracy = dl_accuracy(train_dl, model, device)
    val_accuracy = dl_accuracy(val_dl, model, device)
    test_accuracy = dl_accuracy(test_dl, model, device)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # save the trained model to disk
    torch.save(model.state_dict(), "../../pretrained_models/gpt_spam_classifier.pth")




