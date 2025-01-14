"""Loss functions for training a GPT-2 model."""
import tiktoken
import torch
from torch.nn.functional import cross_entropy

from layers import GPTModel, Config

if __name__ == "__main__":

    text_inputs = [
        "every effort moves",
        "I really like"
    ]

    text_targets = [
        " effort moves you",
        " really like chocolate"
    ]

    tokenizer = tiktoken.get_encoding("gpt2")
    inputs = torch.tensor(tokenizer.encode_batch(text_inputs))
    targets = torch.tensor(tokenizer.encode_batch(text_targets))

    cfg = Config()
    model = GPTModel(cfg)
    model.eval()

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(f"Predictions shape: {probas.shape}")

    # initial softmax probabilities for the target tokens
    target_probas_1 = probas[0, [0, 1, 2], targets[0]]
    target_probas_2 = probas[1, [0, 1, 2], targets[1]]
    print(f"Text 1: {target_probas_1}")
    print(f"Text 2: {target_probas_2}")

    # Cross-entropy measures the difference between two probability distributions,
    # i.e., the predicted and target distributions

    # compute the cross-entropy loss explicitly
    log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
    neg_avg_log_probas = -1 * torch.mean(log_probas)
    print(f"CE Loss: {neg_avg_log_probas}")

    # compute the cross-entropy loss using PyTorch's built-in function
    # flatten the tensors to shape (batch_size * sequence_length, vocab_size)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = cross_entropy(logits_flat, targets_flat)
    print(f"CE Loss: {loss}")

    # Perplexity measures how well the probability distribution predicted by the model matches
    # the actual distribution of the inputs
    # Perplexity = the effective vocabulary size about which the model is uncertain at each step,
    # e.g., a perplexity of 10 means the model is uncertain about the next token among 10 possible tokens
    perplexity = torch.exp(loss)
    print(f"Perplexity: {perplexity}")

