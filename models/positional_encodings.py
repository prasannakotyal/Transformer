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
            Zero tensor of same shape (adds nothing)
        """
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

        # Compute div_term for sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            Position encoding of shape (batch, seq_len, embedding_dim)
        """
        return self.pe[:, : x.size(1), :]


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

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate for pairing: [f1, f1, f2, f2, ...]
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For compatibility with non-RoPE interface, returns zeros.
        RoPE is applied in attention via apply_rotary method.
        """
        return torch.zeros_like(x)

    def apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary encoding to tensor.

        Args:
            x: Tensor of shape (batch, seq_len, num_heads, head_dim)
            seq_len: Sequence length

        Returns:
            Rotated tensor of same shape
        """
        # Get cached values
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]

        # Rotate pairs of dimensions
        x_rotated = self._rotate_half(x)

        return (x * cos) + (x_rotated * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
