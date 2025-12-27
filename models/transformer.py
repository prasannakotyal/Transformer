"""
Main Transformer Language Model.

GPT-style decoder-only transformer with configurable architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork
from .positional_encodings import (
    NoPositionalEncoding,
    AbsolutePositionalEncoding,
    RotaryPositionalEncoding,
)


class TransformerBlock(nn.Module):
    """
    Transformer block: multi-head attention + feed forward.

    Implements both pre-norm and post-norm variants via parameter.
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.2,
        norm_type: str = "pre",  # 'pre' or 'post'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.dropout = dropout
        self.norm_type = norm_type

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            head_size=self.head_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
            dropout=dropout,
        )

        self.ffn = FeedForwardNetwork(embedding_dim=embedding_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with pre/post-norm support.

        Args:
            x: Input tensor
            return_attention: Whether to return attention weights

        Returns:
            out: Transformed tensor
            attention_weights: Optional attention patterns
        """
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
            x = x + self.norm1(attn_out)
            x = x + self.norm2(self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x, None


class TransformerLanguageModel(nn.Module):
    """
    GPT-style decoder-only transformer for language modeling.

    Configurable for architectural experiments:
    - Positional encoding: none, absolute, rotary
    - Normalization: pre-norm or post-norm
    - Model depth: number of layers
    - Attention: number of heads, attention return
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        context_length: int = 256,
        dropout: float = 0.2,
        pos_enc_type: str = "absolute",  # 'none', 'absolute', 'rotary'
        norm_type: str = "pre",  # 'pre' or 'post'
        return_attention: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.dropout = dropout
        self.pos_enc_type = pos_enc_type
        self.norm_type = norm_type
        self.return_attention = return_attention

        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        # Positional encoding
        if pos_enc_type == "none":
            self.pos_encoding = NoPositionalEncoding(embedding_dim)
        elif pos_enc_type == "absolute":
            self.pos_encoding = AbsolutePositionalEncoding(
                embedding_dim=embedding_dim, max_seq_len=context_length
            )
        elif pos_enc_type == "rotary":
            self.pos_encoding = RotaryPositionalEncoding(
                embedding_dim=embedding_dim, max_seq_len=context_length
            )
        else:
            raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}")

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    context_length=context_length,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Output head
        self.lm_head = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        self.attention_weights_cache = []

    def forward(self, idx: torch.Tensor, targets=None):
        """
        Forward pass with optional loss computation.

        Args:
            idx: Input token indices (batch, seq_len)
            targets: Optional target indices (batch, seq_len)

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss
        """
        B, T = idx.shape

        # Embeddings
        token_emb = self.token_embedding(idx)

        if self.pos_enc_type == "rotary":
            # RoPE is applied inside attention
            x = token_emb
        else:
            pos_emb = self.pos_encoding(token_emb)
            x = token_emb + pos_emb

        # Clear attention cache
        self.attention_weights_cache = []

        # Process through blocks
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=self.return_attention)
            if self.return_attention and attn_weights is not None:
                self.attention_weights_cache.append(attn_weights.detach().cpu())

        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k=None,
        top_p=None,
        store_attention: bool = False,
    ):
        """
        Autoregressive text generation.

        Args:
            idx: Context tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = greedy)
            top_k: Sample from top k tokens
            top_p: Nucleus sampling threshold
            store_attention: Store attention patterns during generation

        Returns:
            generated: Full sequence (batch, seq_len + max_new_tokens)
            attention_history: List of attention weights if store_attention
        """
        self.eval()
        generated = idx.clone()
        attention_history = [] if store_attention else None

        with torch.no_grad():
            for i in range(max_new_tokens):
                B, T = generated.shape
                idx_crop = generated[:, -self.context_length :]

                logits, _ = self(idx_crop)
                logits_last = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    logits_last = logits_last / temperature

                # Apply top-k filtering
                if top_k is None or top_k >= self.vocab_size:
                    probs = F.softmax(logits_last, dim=-1)
                else:
                    top_values, top_indices = torch.topk(logits_last, k=top_k, dim=-1)
                    logits_filtered = torch.full_like(logits_last, float("-inf"))
                    logits_filtered.scatter_(1, top_indices, top_values)
                    probs = F.softmax(logits_filtered, dim=-1)

                # Apply top-p (nucleus) filtering
                if top_p is None:
                    pass
                else:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices[cumulative_probs > top_p] = float("-inf")
                    probs = torch.full_like(probs, float("-inf"))
                    probs.scatter_(1, sorted_indices, sorted_probs)
                    probs = F.softmax(probs, dim=-1)

                # Sample next token
                idx_next = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, idx_next], dim=1)

                if store_attention:
                    attention_history.append(self.attention_weights_cache.copy())

        self.train()
        return generated, attention_history
