"""
Experiment configuration dataclass.

Central configuration system for all transformer experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for transformer training experiments.

    Model parameters:
    - vocab_size: Size of vocabulary
    - embedding_dim: Token embedding dimension
    - num_layers: Number of transformer blocks
    - num_heads: Number of attention heads
    - context_length: Maximum sequence length
    - dropout: Dropout rate

    Architecture variants:
    - pos_enc_type: 'none', 'absolute', or 'rotary'
    - norm_type: 'pre' or 'post-norm'
    - return_attention: Whether to return attention weights (for visualization)

    Training parameters:
    - batch_size: Batch size
    - learning_rate: Learning rate
    - max_iters: Maximum training iterations
    - eval_interval: Steps between evaluations
    - eval_iters: Number of batches for evaluation
    - warmup_iters: Learning rate warmup steps
    - use_mixed_precision: Enable AMP training
    - gradient_accumulation: Gradient accumulation steps

    Data parameters:
    - dataset: Dataset name ('shakespeare' or 'gutenberg')
    - train_split: Training data ratio
    - seed: Random seed for reproducibility
    """

    # Model architecture
    vocab_size: int = 100
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    context_length: int = 256
    dropout: float = 0.2

    # Architecture variants
    pos_enc_type: str = "absolute"
    norm_type: str = "pre"
    return_attention: bool = False

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 200
    warmup_iters: int = 100
    use_mixed_precision: bool = True
    gradient_accumulation: int = 2

    # Data
    dataset: str = "gutenberg"
    train_split: float = 0.9
    seed: int = 1337

    # Paths
    data_path: str = "data/gutenberg_subset/"
    output_dir: str = "outputs/"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.pos_enc_type in ["none", "absolute", "rotary"], (
            f"Invalid pos_enc_type: {self.pos_enc_type}"
        )
        assert self.norm_type in ["pre", "post"], f"Invalid norm_type: {self.norm_type}"
        assert self.embedding_dim % self.num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
