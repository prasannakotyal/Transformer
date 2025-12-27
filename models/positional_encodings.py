"""
Positional encoding implementations for Transformer models.

Implements:
- NoPositionalEncoding: Identity (no position information)
- AbsolutePositionalEncoding: Sinusoidal absolute position encoding
- RotaryPositionalEncoding: Rotary position encoding (RoPE)
"""

import torch
import torch.nn as nn
import math


class NoPositionalEncoding(nn.Module):
    """
    No positional encoding - identity function.

    Used as baseline to understand if position encoding is necessary.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Same tensor (no modification)
        """
        return x


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute sinusoidal positional encoding.

    Adds position information via learned or sinusoidal embeddings
    at each position.
    """

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term[:, 0::2])
        pe[:, 1::2] = torch.cos(position * div_term[:, 1::2])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            x + position_encoding
        """
        return x + self.pe[:, : x.size(1), :]


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).

    Encodes position information by rotating query and key vectors
    in 2D planes based on token position. Naturally incorporates
    relative position dependencies.

    Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        assert embedding_dim % 2 == 0, "Embedding dim must be even for RoPE"

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )
        freqs = torch.einsum("i,j->ij", torch.arange(max_seq_len).float(), inv_freq)

        self.register_buffer("cos", freqs.cos().unsqueeze(0))
        self.register_buffer("sin", freqs.sin().unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Apply rotary encoding to queries and keys.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, num_heads, head_dim)

        Returns:
            q_rotated: Rotated queries
            k_rotated: Rotated keys
        """
        seq_len = q.size(1)

        q_rotated = self._apply_rotary(q, seq_len)
        k_rotated = self._apply_rotary(k, seq_len)

        return q_rotated, k_rotated

    def _apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotation to tensor x.

        Splits x into pairs and rotates each pair.
        """
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], 2, -1))

        cos_seq = self.cos[:, :seq_len, :, None, :]
        sin_seq = self.sin[:, :seq_len, :, None, :]

        x_rotated = torch.view_as_real(
            x_complex * cos_seq - torch.complex(x_complex) * sin_seq
        )

        return x_rotated.reshape(x.shape)
