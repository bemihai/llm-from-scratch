"""Implementation of GPT-2 model."""
import tiktoken
import torch
from torch import nn
from torchsummary import summary
from dataclasses import dataclass

from transformer_block import TransformerBlock

torch.manual_seed(123)


@dataclass
class Config:
    """
    The configuration of the GPT-2 model. Default config is GPT-2 small (124M params).
    """
    vocab_size: int = 50_257  # vocabulary size
    context_len: int = 1024  # context length - the max number of input tokens to process
    embed_dim: int = 768  # embedding size of the input tokens
    n_heads: int = 12  # number of attention heads
    n_layers: int = 12  # number of transformer layers
    dropout: float = 0.1  # dropout rate
    qkv_bias: bool = False  # whether to include bias in qkv projection layers (True in the original GPT-2)


class GPTModel(nn.Module):
    """The (modern) GPT-2 model architecture."""

    def __init__(self, cfg: Config):
        super().__init__()
        # input tokens embedding layer
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # input tokens positional embedding layer
        self.position_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # dropout layer
        self.dropout = nn.Dropout(cfg.dropout)
        # transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg.embed_dim, cfg.context_len, cfg.n_heads, cfg.dropout, cfg.qkv_bias)
            for _ in range(cfg.n_layers)]
        )
        # layer normalization
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        # output linear layer
        self.output_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, inputs):
        """The forward pass of the GPT-2 model."""
        # inputs shape: (batch_size, seq_len)
        batch_size, seq_len = inputs.shape
        # generate the inputs token/positional embeddings
        token_embed = self.token_embedding(inputs.long())
        pos_embed = self.position_embedding(torch.arange(seq_len, device=inputs.device))
        # sum up the embeddings and apply dropout
        x = self.dropout(token_embed + pos_embed)
        # apply the transformer blocks and layer normalization
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        # project the output back to the vocabulary size
        logits = self.output_head(x)

        return logits


if __name__ == "__main__":

    # tokenize the input text
    tokenizer = tiktoken.get_encoding("gpt2")
    text_1 = "Every effort moves you"
    text_2 = "Every day holds a"
    # inputs shape: (batch_size=2, sequence_length=4)
    inputs = torch.stack([
        torch.tensor(tokenizer.encode(text_1)),
        torch.tensor(tokenizer.encode(text_2))
    ], dim=0)
    print(f"Input shape: {inputs.shape}")
    print(f"Input tokens: {inputs}")

    # create a GPT-2 model with default configuration
    model = GPTModel(Config())
    logits = model(inputs)
    # output shape: (batch_size=2, sequence_length=4, vocab_size=50257)
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits}")

    # model summary
    print(summary(model, [[4]]))



