"""Implementation of GPT-2 model."""
import torch
from torch import nn
from dataclasses import dataclass

from layers.transformer import TransformerBlock, LayerNorm


@dataclass
class GPTConfig:
    """
    The configuration of the GPT-2 model. Default config is GPT-2 small (124M params) without biases.
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

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        # input tokens embedding layer
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # input tokens positional embedding layer
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)
        # dropout layer
        self.dropout = nn.Dropout(cfg.dropout)
        # transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg.embed_dim, cfg.context_len, cfg.n_heads, cfg.dropout, cfg.qkv_bias)
            for _ in range(cfg.n_layers)]
        )
        # layer normalization
        self.final_norm = LayerNorm(cfg.embed_dim)
        # output linear layer
        self.out_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, inputs):
        """The forward pass of the GPT-2 model."""
        # inputs shape: (batch_size, seq_len)
        try:
            batch_size, seq_len = inputs.shape
        except ValueError:
            breakpoint()
        # generate the inputs token/positional embeddings
        token_embed = self.tok_emb(inputs.long())
        pos_embed = self.pos_emb(torch.arange(seq_len, device=inputs.device))
        # sum up the embeddings and apply dropout
        x = self.dropout(token_embed + pos_embed)
        # apply the transformer blocks and layer normalization
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # project the output back to the vocabulary size
        logits = self.out_head(x)

        return logits


class GPTClassifier(GPTModel):
    """GPT-2 classifier model."""

    def __init__(self, config, num_classes: int):
        super().__init__(config)
        # add a classification head
        self.classifier = nn.Linear(config.vocab_size, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = super().forward(x)
        x = self.classifier(x)
        return x






