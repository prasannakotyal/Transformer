"""
Attention mechanisms for Transformer models.

Implements:
- Head: Single attention head with causal masking
- MultiHeadAttention: Concatenated multiple attention heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Head(nn.Module):
    """
    Single self-attention head.

    Computes scaled dot-product attention with causal masking for
    autoregressive generation.
    """

    def __init__(
        self,
        head_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )
        self.query_layer = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )
        self.value_layer = nn.Linear(
            in_features=embedding_dim, out_features=head_size, bias=False
        )

        self.register_buffer(
            "tril", torch.tril(torch.ones((context_length, context_length)))
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of attention head.

        Args:
            x: Input tensor of shape (batch, time, channels)
            return_attention: Whether to return attention weights

        Returns:
            out: Context vectors of shape (batch, time, head_size)
            attention_weights: Optional, shape (batch, time, time)
        """
        B, T, C = x.shape

        assert T <= self.context_length, (
            f"Sequence length {T} exceeds max {self.context_length}"
        )
        assert C == self.embedding_dim, (
            f"Channel dim {C} doesn't match {self.embedding_dim}"
        )

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        weights = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout_layer(weights)

        out = weights @ v

        if return_attention:
            return out, weights
        return out, None


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel and concatenates their outputs.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    embedding_dim=embedding_dim,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        self.projection_layer = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=True
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, time, embedding_dim)
            return_attention: Whether to return attention weights

        Returns:
            out: Concatenated head outputs, shape (batch, time, embedding_dim)
            attention_weights: Optional, shape (batch, num_heads, time, time)
        """
        head_outputs = []
        attention_weights_list = []

        for head in self.heads:
            out, weights = head(x, return_attention=return_attention)
            head_outputs.append(out)
            if return_attention and weights is not None:
                attention_weights_list.append(weights)

        out = torch.cat(head_outputs, dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)

        if return_attention and attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)
            return out, attention_weights
        return out, None
