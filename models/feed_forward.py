"""
Feed forward network for transformer blocks.

Simple MLP with activation and dropout.
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Two-layer feed forward network with GELU activation.

    Architecture:
        Linear(embedding_dim -> embedding_dim*4)
        GELU activation
        Linear(embedding_dim*4 -> embedding_dim)
        Dropout
    """

    def __init__(self, embedding_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 4),
            nn.GELU(),
            nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Transformed tensor of same shape
        """
        return self.ffn(x)
