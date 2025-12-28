"""
Byte-level BPE Tokenizer from scratch.

Inspired by Karpathy's minbpe. Implements the core BPE algorithm:
1. Start with byte-level tokens (256 base vocabulary)
2. Iteratively merge most frequent adjacent pairs
3. Build vocabulary up to target size

Reference: https://github.com/karpathy/minbpe
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(self):
        # Base vocabulary: 256 bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab_size = 256

    def _get_pair_counts(self, token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequency of adjacent token pairs."""
        counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(
        self, token_ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
        """Replace all occurrences of pair with new_id."""
        result = []
        i = 0
        while i < len(token_ids):
            if (
                i < len(token_ids) - 1
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        Train BPE tokenizer on text.

        Args:
            text: Training text
            vocab_size: Target vocabulary size (must be >= 256)
            verbose: Print progress
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (base byte vocabulary)"

        # Start with byte-level tokens
        token_ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        if verbose:
            print(f"Training BPE: {len(text):,} chars -> {len(token_ids):,} bytes")
            print(f"Target vocab size: {vocab_size} ({num_merges} merges)")

        for i in range(num_merges):
            # Count pairs
            pair_counts = self._get_pair_counts(token_ids)
            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            # Create new token
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Merge in token sequence
            token_ids = self._merge_pair(token_ids, best_pair, new_id)

            if verbose and (i + 1) % 500 == 0:
                print(
                    f"  Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} "
                    f"(count: {best_count}, tokens: {len(token_ids):,})"
                )

        self.vocab_size = 256 + len(self.merges)
        if verbose:
            print(f"Final vocab size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        token_ids = list(text.encode("utf-8"))

        # Apply merges in order learned
        for pair, new_id in self.merges.items():
            token_ids = self._merge_pair(token_ids, pair, new_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        byte_sequence = b"".join(self.vocab[id] for id in token_ids)
        return byte_sequence.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        data = {
            "vocab_size": self.vocab_size,
            # Convert tuple keys to strings for JSON
            "merges": {f"{p[0]},{p[1]}": v for p, v in self.merges.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        path = Path(path)
        data = json.loads(path.read_text())

        self.vocab_size = data["vocab_size"]
        # Reconstruct merges with tuple keys
        self.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }
        # Rebuild vocab from merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]


# Quick test
if __name__ == "__main__":
    tokenizer = BPETokenizer()

    # Test on small text
    text = "hello world! hello hello world" * 100
    tokenizer.train(text, vocab_size=280, verbose=True)

    # Test encode/decode
    test = "hello world!"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest: '{test}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Roundtrip OK: {test == decoded}")
