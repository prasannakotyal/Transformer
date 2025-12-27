"""
Main Transformer Language Model.

GPT-style decoder-only transformer with configurable architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork
from .positional_encodings import NoPositionalEncoding, AbsolutePositionalEncoding


class TransformerBlock(nn.Module):
    """
    Transformer block: multi-head attention + feed forward.
    Supports both pre-norm and post-norm variants.
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.2,
        norm_type: str = "pre",
    ):
        super().__init__()
        self.norm_type = norm_type
        head_size = embedding_dim // num_heads

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
            dropout=dropout,
        )
        self.ffn = FeedForwardNetwork(embedding_dim=embedding_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.norm_type == "pre":
            attn_out, attn_weights = self.attention(
                self.norm1(x), return_attention=return_attention
            )
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
        else:  # post-norm
            attn_out, attn_weights = self.attention(
                x, return_attention=return_attention
            )
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class TransformerLanguageModel(nn.Module):
    """
    GPT-style decoder-only transformer for language modeling.

    Configurable architecture:
    - pos_enc_type: 'none' or 'absolute'
    - norm_type: 'pre' or 'post'
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        context_length: int = 256,
        dropout: float = 0.2,
        pos_enc_type: str = "absolute",
        norm_type: str = "pre",
        return_attention: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.return_attention = return_attention

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding
        if pos_enc_type == "none":
            self.pos_encoding = NoPositionalEncoding(embedding_dim)
        elif pos_enc_type == "absolute":
            self.pos_encoding = AbsolutePositionalEncoding(
                embedding_dim, context_length
            )
        else:
            raise ValueError(
                f"Unknown pos_enc_type: {pos_enc_type}. Use 'none' or 'absolute'."
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads, embedding_dim, context_length, dropout, norm_type
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token indices (batch, seq_len)
            targets: Optional target indices for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        # Embeddings + positional encoding
        x = self.token_embedding(idx)
        x = x + self.pos_encoding(x)

        # Transformer blocks
        attention_cache = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=self.return_attention)
            if self.return_attention and attn_weights is not None:
                attention_cache.append(attn_weights.detach().cpu())

        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            idx: Context tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Sample from top k tokens only

        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.context_length :]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
