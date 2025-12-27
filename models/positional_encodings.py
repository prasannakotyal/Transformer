"""
Positional encoding implementations for Transformer models.

Implements:
- NoPositionalEncoding: Identity (no position information)
- AbsolutePositionalEncoding: Sinusoidal absolute position encoding
"""

import torch
import torch.nn as nn
import math


class NoPositionalEncoding(nn.Module):
    """
    No positional encoding - returns zeros.
    Used as baseline to understand if position encoding is necessary.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns zero tensor (adds no position information)."""
        return torch.zeros_like(x)


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute sinusoidal positional encoding.
    Adds position information via sinusoidal embeddings at each position.
    """

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns position encoding of shape (batch, seq_len, embedding_dim)."""
        return self.pe[:, : x.size(1), :]
