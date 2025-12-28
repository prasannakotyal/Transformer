"""
Visualization tools for Transformer training and analysis.

Creates publication-quality plots:
1. Training curves (train/val loss)
2. Attention heatmaps
3. Token embedding visualization (t-SNE/PCA)
4. Generation samples at different training stages
5. KV-cache speedup comparison

Usage:
    python visualize.py --all
    python visualize.py --plot training_curves
    python visualize.py --plot attention_heatmap
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import Transformer
from tokenizer import BPETokenizer

# Paths
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
LOG_FILE = OUTPUT_DIR / "training_log.json"

# Dark theme colors
COLORS = {
    "background": "#1a1a2e",
    "text": "#eaeaea",
    "grid": "#2d2d44",
    "train": "#00d4ff",
    "val": "#ff6b6b",
    "accent1": "#4ecdc4",
    "accent2": "#ffe66d",
    "accent3": "#95e1d3",
}


def setup_dark_theme():
    """Configure matplotlib for dark theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "legend.facecolor": COLORS["background"],
            "legend.edgecolor": COLORS["grid"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def plot_training_curves(log_path: Path = LOG_FILE, save_path: Optional[Path] = None):
    """
    Plot training and validation loss curves.

    Shows model convergence and potential overfitting.
    """
    setup_dark_theme()

    with open(log_path, "r") as f:
        log = json.load(f)

    iterations = log["iterations"]
    train_losses = log["train_losses"]
    val_losses = log["val_losses"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        iterations, train_losses, color=COLORS["train"], linewidth=2, label="Train Loss"
    )
    ax.plot(
        iterations,
        val_losses,
        color=COLORS["val"],
        linewidth=2,
        label="Validation Loss",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add final loss annotation
    ax.annotate(
        f"Final: {val_losses[-1]:.3f}",
        xy=(iterations[-1], val_losses[-1]),
        xytext=(iterations[-1] - len(iterations) * 0.15, val_losses[-1] + 0.1),
        color=COLORS["val"],
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLORS["val"], alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_heatmap(
    model: Transformer,
    tokenizer: BPETokenizer,
    text: str,
    layer: int = 0,
    head: int = 0,
    save_path: Optional[Path] = None,
    device: torch.device = None,
):
    """
    Visualize attention patterns for a given text.

    Shows which tokens attend to which other tokens.
    """
    setup_dark_theme()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Tokenize
    token_ids = tokenizer.encode(text)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get attention weights
    with torch.inference_mode():
        _, _, attentions, _ = model(x, return_attention=True)

    if not attentions:
        print("No attention weights returned. Model may not support return_attention.")
        return

    # Get attention for specified layer and head
    # attentions[layer] shape: (batch, num_heads, seq_len, seq_len)
    attn = attentions[layer][0, head].cpu().numpy()  # (seq_len, seq_len)

    # Create token labels
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    # Truncate long tokens
    tokens = [t[:10] + "..." if len(t) > 10 else t for t in tokens]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    # Labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"Attention Heatmap (Layer {layer}, Head {head})")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_embeddings(
    model: Transformer,
    tokenizer: BPETokenizer,
    num_tokens: int = 500,
    method: str = "tsne",
    save_path: Optional[Path] = None,
):
    """
    Visualize token embeddings using dimensionality reduction.

    Shows how tokens cluster in embedding space.
    """
    setup_dark_theme()

    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()

    # Select subset of tokens
    num_tokens = min(num_tokens, embeddings.shape[0])
    embeddings = embeddings[:num_tokens]

    # Dimensionality reduction
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            reducer = TSNE(
                n_components=2, random_state=42, perplexity=min(30, num_tokens - 1)
            )
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            print("sklearn not available, falling back to PCA")
            method = "pca"

    if method == "pca":
        # Simple PCA without sklearn
        embeddings_centered = embeddings - embeddings.mean(axis=0)
        _, _, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
        reduced = embeddings_centered @ Vt[:2].T

    # Get token labels for annotation
    labels = [tokenizer.decode([i])[:10] for i in range(num_tokens)]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by token ID (shows structure)
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=range(num_tokens),
        cmap="viridis",
        alpha=0.7,
        s=30,
    )

    # Annotate some tokens
    annotate_every = max(1, num_tokens // 20)  # Annotate ~20 tokens
    for i in range(0, num_tokens, annotate_every):
        ax.annotate(
            labels[i],
            (reduced[i, 0], reduced[i, 1]),
            fontsize=7,
            alpha=0.8,
            color=COLORS["text"],
        )

    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_title(f"Token Embeddings ({method.upper()})")

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Token ID")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved embeddings plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_generation_samples(
    log_path: Path = LOG_FILE, save_path: Optional[Path] = None
):
    """
    Display text samples generated at different training stages.

    Shows how generation quality improves during training.
    """
    setup_dark_theme()

    with open(log_path, "r") as f:
        log = json.load(f)

    samples = log.get("samples", [])
    if not samples:
        print("No samples found in training log.")
        return

    # Select samples to display
    if len(samples) > 6:
        indices = np.linspace(0, len(samples) - 1, 6, dtype=int)
        samples = [samples[i] for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, sample in enumerate(samples):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.axis("off")

        # Wrap text
        text = sample["text"][:300]  # Limit length
        wrapped = "\n".join([text[j : j + 50] for j in range(0, len(text), 50)])

        ax.text(
            0.05,
            0.95,
            f"Iteration {sample['iter']}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=COLORS["accent1"],
            verticalalignment="top",
        )

        ax.text(
            0.05,
            0.85,
            wrapped,
            transform=ax.transAxes,
            fontsize=8,
            color=COLORS["text"],
            verticalalignment="top",
            family="monospace",
            wrap=True,
        )

        # Border
        ax.patch.set_edgecolor(COLORS["grid"])
        ax.patch.set_linewidth(2)

    # Hide unused axes
    for i in range(len(samples), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Generation Samples During Training", fontsize=16, color=COLORS["text"]
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved generation samples to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_kv_cache_speedup(
    model: Transformer,
    tokenizer: BPETokenizer,
    max_tokens: int = 100,
    num_trials: int = 5,
    save_path: Optional[Path] = None,
    device: torch.device = None,
):
    """
    Benchmark and visualize KV-cache speedup.

    Shows the efficiency gain from caching key-value pairs.
    """
    import time

    setup_dark_theme()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Warmup
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        _ = model.generate(context, max_new_tokens=10, use_cache=True)
        _ = model.generate(context, max_new_tokens=10, use_cache=False)

    # Benchmark different sequence lengths
    token_counts = [25, 50, 75, 100]
    token_counts = [t for t in token_counts if t <= max_tokens]

    cache_times = []
    no_cache_times = []

    for num_tokens in token_counts:
        # With cache
        times = []
        for _ in range(num_trials):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            start = time.time()
            with torch.inference_mode():
                _ = model.generate(context, max_new_tokens=num_tokens, use_cache=True)
            times.append(time.time() - start)
        cache_times.append(np.mean(times))

        # Without cache
        times = []
        for _ in range(num_trials):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            start = time.time()
            with torch.inference_mode():
                _ = model.generate(context, max_new_tokens=num_tokens, use_cache=False)
            times.append(time.time() - start)
        no_cache_times.append(np.mean(times))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    x = np.arange(len(token_counts))
    width = 0.35

    ax1.bar(
        x - width / 2,
        cache_times,
        width,
        label="With KV Cache",
        color=COLORS["accent1"],
    )
    ax1.bar(
        x + width / 2,
        no_cache_times,
        width,
        label="Without KV Cache",
        color=COLORS["val"],
    )

    ax1.set_xlabel("Tokens Generated")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Generation Time Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(token_counts)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Speedup line chart
    speedups = [
        no_cache / cache if cache > 0 else 0
        for cache, no_cache in zip(cache_times, no_cache_times)
    ]

    ax2.plot(
        token_counts,
        speedups,
        marker="o",
        linewidth=2,
        markersize=8,
        color=COLORS["accent2"],
    )
    ax2.axhline(y=1, color=COLORS["grid"], linestyle="--", alpha=0.5)

    ax2.set_xlabel("Tokens Generated")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("KV-Cache Speedup Factor")
    ax2.grid(True, alpha=0.3)

    # Annotate speedups
    for i, (tc, s) in enumerate(zip(token_counts, speedups)):
        ax2.annotate(
            f"{s:.1f}x",
            (tc, s),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=COLORS["accent2"],
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved KV-cache speedup plot to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print summary
    print("\nKV-Cache Speedup Summary:")
    for tc, cache, no_cache, speedup in zip(
        token_counts, cache_times, no_cache_times, speedups
    ):
        print(
            f"  {tc} tokens: {cache:.3f}s (cache) vs {no_cache:.3f}s (no cache) = {speedup:.1f}x speedup"
        )


def create_all_plots(checkpoint_path: str, tokenizer_path: str):
    """Create all visualization plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = Transformer(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        context_length=config["context_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Creating all visualizations...")

    # 1. Training curves
    if LOG_FILE.exists():
        plot_training_curves(save_path=PLOTS_DIR / "training_curves.png")

    # 2. Attention heatmap
    sample_text = "The quick brown fox jumps over the lazy dog. This is a test."
    plot_attention_heatmap(
        model,
        tokenizer,
        sample_text,
        layer=0,
        head=0,
        save_path=PLOTS_DIR / "attention_heatmap.png",
        device=device,
    )

    # 3. Token embeddings
    plot_embeddings(
        model,
        tokenizer,
        num_tokens=300,
        method="pca",  # Use PCA (doesn't require sklearn)
        save_path=PLOTS_DIR / "embeddings.png",
    )

    # 4. Generation samples
    if LOG_FILE.exists():
        plot_generation_samples(save_path=PLOTS_DIR / "generation_samples.png")

    # 5. KV-cache speedup
    plot_kv_cache_speedup(
        model,
        tokenizer,
        max_tokens=100,
        save_path=PLOTS_DIR / "kv_cache_speedup.png",
        device=device,
    )

    print(f"\nAll plots saved to {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Create visualizations for transformer training"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all plots",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=[
            "training_curves",
            "attention_heatmap",
            "embeddings",
            "generation_samples",
            "kv_cache_speedup",
        ],
        help="Create specific plot",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/checkpoint_final.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer.json",
        help="Path to tokenizer",
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        create_all_plots(args.checkpoint, args.tokenizer)
    elif args.plot == "training_curves":
        plot_training_curves(save_path=PLOTS_DIR / "training_curves.png")
    elif args.plot == "generation_samples":
        plot_generation_samples(save_path=PLOTS_DIR / "generation_samples.png")
    else:
        # Plots requiring model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = BPETokenizer()
        tokenizer.load(args.tokenizer)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint["config"]
        model = Transformer(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            context_length=config["context_length"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if args.plot == "attention_heatmap":
            text = "The quick brown fox jumps over the lazy dog."
            plot_attention_heatmap(
                model,
                tokenizer,
                text,
                save_path=PLOTS_DIR / "attention_heatmap.png",
                device=device,
            )
        elif args.plot == "embeddings":
            plot_embeddings(model, tokenizer, save_path=PLOTS_DIR / "embeddings.png")
        elif args.plot == "kv_cache_speedup":
            plot_kv_cache_speedup(
                model,
                tokenizer,
                save_path=PLOTS_DIR / "kv_cache_speedup.png",
                device=device,
            )


if __name__ == "__main__":
    main()
