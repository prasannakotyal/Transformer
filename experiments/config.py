"""
Experiment configuration dataclass.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for transformer training experiments."""

    # Model architecture
    vocab_size: int = 100
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    context_length: int = 256
    dropout: float = 0.2

    # Architecture variants
    pos_enc_type: str = "absolute"  # 'none' or 'absolute'
    norm_type: str = "pre"  # 'pre' or 'post'
    return_attention: bool = False

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 50
    warmup_iters: int = 100
    use_mixed_precision: bool = True
    gradient_accumulation: int = 1

    # Data
    train_split: float = 0.9
    seed: int = 1337

    # Paths
    data_path: str = "data/gutenberg_subset/all_books.txt"
    output_dir: str = "outputs/"

    # Experiment name
    name: str = "experiment"

    def __post_init__(self):
        assert self.pos_enc_type in ["none", "absolute"], (
            f"Invalid pos_enc_type: {self.pos_enc_type}. Use 'none' or 'absolute'."
        )
        assert self.norm_type in ["pre", "post"], (
            f"Invalid norm_type: {self.norm_type}. Use 'pre' or 'post'."
        )
        assert self.embedding_dim % self.num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
