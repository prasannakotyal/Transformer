"""
Data loader for Gutenberg corpus.

Handles character-level tokenization and batching.
"""

import os
import torch
from typing import Tuple


class GutenbergDataset:
    """
    Character-level dataset for Project Gutenberg books.

    Supports:
    - Character-level tokenization
    - Train/val/test splits
    - Batching with context windows
    """

    def __init__(
        self,
        data_path: str,
        context_length: int = 256,
        train_split: float = 0.9,
        device: str = "cuda",
    ):
        """
        Args:
            data_path: Path to text file or directory
            context_length: Maximum sequence length
            train_split: Fraction of data for training
            device: 'cuda' or 'cpu'
        """
        self.context_length = context_length
        self.device = device

        # Load text
        self.text = self._load_data(data_path)

        # Build vocabulary
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)

        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Tokenize
        self.data = torch.tensor(
            [self.char_to_idx[char] for char in self.text], dtype=torch.long
        )

        # Split
        split_idx = int(len(self.data) * train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

        print(f"Loaded {len(self.text):,} characters")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train: {len(self.train_data):,}, Val: {len(self.val_data):,}")

    def _load_data(self, data_path: str) -> str:
        """
        Load text from file or directory.
        """
        if os.path.isfile(data_path):
            # Single file
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            return text
        elif os.path.isdir(data_path):
            # Directory - concatenate all files
            all_text = []
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    all_text.append(text)
                    print(f"Loaded: {filename}")
            return "".join(all_text)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

    def encode(self, text: str) -> torch.Tensor:
        """Convert string to token indices."""
        return torch.tensor(
            [self.char_to_idx[char] for char in text],
            dtype=torch.long,
            device=self.device,
        )

    def decode(self, indices: torch.Tensor) -> str:
        """Convert token indices to string."""
        chars = [self.idx_to_char[int(idx.item())] for idx in indices.flatten()]
        return "".join(chars)

    def get_batch(
        self, split: str, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.

        Args:
            split: 'train' or 'val'
            batch_size: Number of sequences per batch

        Returns:
            x: Input sequences (batch, context_length)
            y: Target sequences (batch, context_length)
        """
        data = self.train_data if split == "train" else self.val_data

        # Random start positions
        max_start = len(data) - self.context_length - 1
        start_indices = torch.randint(0, max_start, (batch_size,))

        # Create batches
        x_batch = torch.stack(
            [data[start : start + self.context_length] for start in start_indices]
        )
        y_batch = torch.stack(
            [
                data[start + 1 : start + self.context_length + 1]
                for start in start_indices
            ]
        )

        return x_batch.to(self.device), y_batch.to(self.device)


def download_gutenberg_sample(
    output_dir: str = "data/gutenberg_subset/", num_books: int = 10
) -> str:
    """
    Download sample of Project Gutenberg books for experiments.

    Args:
        output_dir: Directory to save books
        num_books: Number of books to download

    Returns:
        Path to combined text file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Selected books with direct Gutenberg mirror URLs
    # Format: (book_id, title) - uses gutenberg.org/cache/epub/{id}/pg{id}.txt
    BOOKS = [
        # Classic Literature
        (1342, "Pride and Prejudice - Jane Austen"),
        (2701, "Moby Dick - Herman Melville"),
        (11, "Alice in Wonderland - Lewis Carroll"),
        # Fantasy/Adventure
        (84, "Frankenstein - Mary Shelley"),
        (35, "The Time Machine - H.G. Wells"),
        # Philosophy
        (1232, "The Prince - Niccolo Machiavelli"),
        # Mystery
        (1661, "The Adventures of Sherlock Holmes - Arthur Conan Doyle"),
        # Science Fiction
        (64, "The Gods of Mars - Edgar Rice Burroughs"),
        # Additional classics
        (1952, "The Yellow Wallpaper - Charlotte Perkins Gilman"),
        (74, "The Adventures of Tom Sawyer - Mark Twain"),
    ][:num_books]

    import requests

    all_text = []

    for book_id, book_title in BOOKS:
        print(f"Downloading: {book_title}")

        try:
            # Use the reliable cache/epub URL format
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            text = response.text

            # Save individual file
            safe_title = "".join(
                c for c in book_title if c.isalnum() or c in " -_"
            ).strip()
            filepath = os.path.join(output_dir, f"{safe_title}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  Saved: {filepath} ({len(text):,} chars)")
            all_text.append(text)

        except Exception as e:
            print(f"  Error downloading {book_title}: {e}")

    if not all_text:
        raise RuntimeError("Failed to download any books. Check internet connection.")

    combined_file = os.path.join(output_dir, "all_books.txt")
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join([t for t in all_text if t.strip()]))

    print(f"\nCombined dataset saved to: {combined_file}")
    print(f"Total characters: {sum(len(t) for t in all_text):,}")

    return combined_file
