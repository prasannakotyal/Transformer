"""
Training script for Transformer Language Model.

Downloads FineWeb-Edu data, trains BPE tokenizer, and trains the model.
Designed to run on Kaggle T4x2 GPU in ~1-2 hours.

Usage:
    python train.py
"""

import os
import sys
import json
import time
import math
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import Transformer
from tokenizer import BPETokenizer

# =============================================================================
# Hyperparameters (all in one place, no config files)
# =============================================================================

# Model
VOCAB_SIZE = 4096  # BPE vocabulary size
CONTEXT_LENGTH = 256  # Maximum sequence length
EMBEDDING_DIM = 384  # Model dimension
NUM_LAYERS = 6  # Number of transformer blocks
NUM_HEADS = 6  # Number of attention heads
DROPOUT = 0.1  # Dropout probability

# Training
BATCH_SIZE = 64  # Batch size (adjust for GPU memory)
MAX_ITERS = 5000  # Total training iterations
EVAL_INTERVAL = 250  # Evaluate every N iterations
EVAL_ITERS = 100  # Number of batches for evaluation
LEARNING_RATE = 3e-4  # Peak learning rate
WARMUP_ITERS = 500  # Learning rate warmup steps
MIN_LR = 3e-5  # Minimum learning rate
GRAD_CLIP = 1.0  # Gradient clipping threshold
WEIGHT_DECAY = 0.1  # AdamW weight decay

# Data
DATA_SIZE_MB = 50  # Amount of FineWeb-Edu data to download
TRAIN_SPLIT = 0.9  # Train/validation split ratio

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_FILE = OUTPUT_DIR / "training_log.json"

# =============================================================================
# Data Loading
# =============================================================================


def download_fineweb_edu(size_mb: int = 50) -> str:
    """
    Download FineWeb-Edu sample data.

    Returns:
        Text content as a single string
    """
    cache_file = DATA_DIR / "fineweb_edu.txt"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        return cache_file.read_text(encoding="utf-8")

    print(f"Downloading FineWeb-Edu sample (~{size_mb}MB)...")

    try:
        from datasets import load_dataset

        # Load streaming dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        # Collect text until we have enough
        texts = []
        total_chars = 0
        target_chars = size_mb * 1_000_000  # Approximate chars from MB

        for sample in tqdm(dataset, desc="Downloading"):
            text = sample["text"]
            texts.append(text)
            total_chars += len(text)

            if total_chars >= target_chars:
                break

        full_text = "\n\n".join(texts)

        # Cache to disk
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(full_text, encoding="utf-8")
        print(f"Saved {len(full_text):,} characters to {cache_file}")

        return full_text

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Falling back to TinyShakespeare...")
        return download_tiny_shakespeare()


def download_tiny_shakespeare() -> str:
    """Fallback dataset: TinyShakespeare."""
    import requests

    cache_file = DATA_DIR / "tinyshakespeare.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")

    return text


class DataLoader:
    """Simple data loader for language modeling."""

    def __init__(
        self,
        token_ids: list,
        batch_size: int,
        context_length: int,
        split: str = "train",
        train_split: float = 0.9,
    ):
        self.batch_size = batch_size
        self.context_length = context_length

        # Split data
        n = len(token_ids)
        split_idx = int(n * train_split)

        if split == "train":
            self.data = torch.tensor(token_ids[:split_idx], dtype=torch.long)
        else:  # validation
            self.data = torch.tensor(token_ids[split_idx:], dtype=torch.long)

        print(f"{split.capitalize()} data: {len(self.data):,} tokens")

    def get_batch(self, device: torch.device) -> tuple:
        """Get a random batch of data."""
        ix = torch.randint(len(self.data) - self.context_length - 1, (self.batch_size,))
        x = torch.stack([self.data[i : i + self.context_length] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.context_length + 1] for i in ix])
        return x.to(device), y.to(device)


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def get_lr(iter_num: int) -> float:
    """Cosine learning rate schedule with warmup."""
    # Warmup
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * (iter_num + 1) / WARMUP_ITERS

    # Cosine decay
    decay_ratio = (iter_num - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# =============================================================================
# Training Loop
# =============================================================================


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Estimate loss on train and validation sets."""
    model.eval()
    losses = {"train": 0.0, "val": 0.0}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for _ in range(EVAL_ITERS):
            x, y = loader.get_batch(device)
            with autocast(dtype=torch.float16):
                _, loss, _, _ = model(x, targets=y)
            total_loss += loss.item()
        losses[split] = total_loss / EVAL_ITERS

    model.train()
    return losses


def train():
    """Main training function."""
    print("=" * 60)
    print("Transformer Language Model Training")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Download data
    text = download_fineweb_edu(DATA_SIZE_MB)
    print(f"Total text: {len(text):,} characters")

    # Train or load tokenizer
    tokenizer_path = DATA_DIR / "tokenizer.json"
    tokenizer = BPETokenizer()

    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(str(tokenizer_path))
    else:
        print("Training BPE tokenizer...")
        tokenizer.train(text, vocab_size=VOCAB_SIZE, verbose=True)
        tokenizer.save(str(tokenizer_path))

    # Tokenize data
    print("Tokenizing data...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    # Create data loaders
    train_loader = DataLoader(
        token_ids, BATCH_SIZE, CONTEXT_LENGTH, "train", TRAIN_SPLIT
    )
    val_loader = DataLoader(token_ids, BATCH_SIZE, CONTEXT_LENGTH, "val", TRAIN_SPLIT)

    # Create model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # Mixed precision
    scaler = GradScaler()

    # Training log
    log = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "batch_size": BATCH_SIZE,
            "max_iters": MAX_ITERS,
            "learning_rate": LEARNING_RATE,
        },
        "train_losses": [],
        "val_losses": [],
        "iterations": [],
        "samples": [],  # Text samples at different stages
    }

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    model.train()
    for iter_num in tqdm(range(MAX_ITERS), desc="Training"):
        # Update learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        x, y = train_loader.get_batch(device)

        # Forward pass with mixed precision
        with autocast(dtype=torch.float16):
            _, loss, _, _ = model(x, targets=y)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Evaluation
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, train_loader, val_loader, device)

            elapsed = time.time() - start_time
            print(
                f"\nIter {iter_num}: "
                f"train_loss={losses['train']:.4f}, "
                f"val_loss={losses['val']:.4f}, "
                f"lr={lr:.2e}, "
                f"time={elapsed:.1f}s"
            )

            # Log
            log["train_losses"].append(losses["train"])
            log["val_losses"].append(losses["val"])
            log["iterations"].append(iter_num)

            # Generate sample
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_ids = model.generate(context, max_new_tokens=100, temperature=0.8)
            sample_text = tokenizer.decode(generated_ids[0].tolist())
            log["samples"].append({"iter": iter_num, "text": sample_text[:500]})
            print(f"Sample: {sample_text[:200]}...")
            model.train()

            # Save checkpoint
            checkpoint = {
                "iter": iter_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "config": log["config"],
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f"checkpoint_{iter_num:05d}.pt")

    # Save final checkpoint
    final_checkpoint = {
        "iter": MAX_ITERS - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": log["train_losses"][-1] if log["train_losses"] else 0.0,
        "val_loss": log["val_losses"][-1] if log["val_losses"] else 0.0,
        "config": log["config"],
    }
    torch.save(final_checkpoint, CHECKPOINT_DIR / "checkpoint_final.pt")

    # Save training log
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Final train loss: {log['train_losses'][-1]:.4f}")
    print(f"Final val loss: {log['val_losses'][-1]:.4f}")
    print(f"Checkpoint saved to {CHECKPOINT_DIR}")
    print(f"Training log saved to {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    train()
