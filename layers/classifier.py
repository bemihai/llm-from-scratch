"""Classifier based on the GPT-2 model."""
from torch import nn

from .gpt_2 import GPTModel


class GPTClassifier(GPTModel):
    """GPT-2 classifier model."""

    def __init__(self, config, num_classes: int):
        super().__init__(config)
        # add a classification head
        self.classifier = nn.Linear(config.vocab_size, num_classes)
        # freeze all the layers except the last transformer block, the final layer norm, and
        # the original output head (these are fine-tuned for classification)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.trf_blocks[-1].parameters():
            param.require_grad = True
        for param in self.final_norm.parameters():
            param.require_grad = True
        for param in self.out_head.parameters():
            param.require_grad = True


    def forward(self, x):
        """Forward pass."""
        x = super().forward(x)
        x = self.classifier(x)
        return x