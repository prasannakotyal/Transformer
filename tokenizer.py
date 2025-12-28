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
from collections import defaultdict


class BPETokenizer:
    """Byte-level BPE tokenizer with optimized training."""

    def __init__(self):
        # Base vocabulary: 256 bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab_size = 256

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        Train BPE tokenizer on text using optimized algorithm.

        Uses a chunked approach for large texts to avoid memory issues.
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (base byte vocabulary)"

        # Start with byte-level tokens
        all_bytes = text.encode("utf-8")
        num_merges = vocab_size - 256

        if verbose:
            print(f"Training BPE: {len(text):,} chars -> {len(all_bytes):,} bytes")
            print(f"Target vocab size: {vocab_size} ({num_merges} merges)")

        # For large texts, use sampling-based approach
        if len(all_bytes) > 1_000_000:
            self._train_sampled(all_bytes, num_merges, verbose)
        else:
            self._train_full(all_bytes, num_merges, verbose)

        self.vocab_size = 256 + len(self.merges)
        if verbose:
            print(f"Final vocab size: {self.vocab_size}")

    def _train_sampled(self, all_bytes: bytes, num_merges: int, verbose: bool) -> None:
        """
        Train on sampled chunks for large texts.

        Strategy: Sample ~1MB of text, train on that, which is fast and gives
        good merges for common patterns.
        """
        # Sample evenly spaced chunks
        sample_size = 1_000_000  # 1MB sample
        chunk_size = 10_000
        num_chunks = sample_size // chunk_size

        total_len = len(all_bytes)
        step = max(1, (total_len - chunk_size) // num_chunks)

        # Collect sampled chunks
        sampled = bytearray()
        for i in range(0, min(total_len - chunk_size, step * num_chunks), step):
            sampled.extend(all_bytes[i : i + chunk_size])

        if verbose:
            print(f"  Sampled {len(sampled):,} bytes from {total_len:,} total")

        # Train on sampled data
        self._train_full(bytes(sampled), num_merges, verbose)

    def _train_full(self, data: bytes, num_merges: int, verbose: bool) -> None:
        """Train on full data using optimized pair counting."""
        token_ids = list(data)

        for i in range(num_merges):
            # Count pairs (optimized with defaultdict)
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for j in range(len(token_ids) - 1):
                pair_counts[(token_ids[j], token_ids[j + 1])] += 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                # No more useful merges
                break

            # Create new token
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Merge in token sequence (in-place for speed)
            token_ids = self._merge_pair_fast(token_ids, best_pair, new_id)

            if verbose and (i + 1) % 100 == 0:
                print(
                    f"  Merge {i + 1}/{num_merges}: "
                    f"count={best_count:,}, tokens={len(token_ids):,}"
                )

    def _merge_pair_fast(
        self, token_ids: List[int], pair: Tuple[int, int], new_id: int
    ) -> List[int]:
        """Replace all occurrences of pair with new_id (optimized)."""
        result = []
        i = 0
        p0, p1 = pair
        n = len(token_ids)
        while i < n:
            if i < n - 1 and token_ids[i] == p0 and token_ids[i + 1] == p1:
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (chunked for large texts)."""
        data = text.encode("utf-8")

        # For large texts, encode in chunks to avoid O(n*m) on full text
        if len(data) > 50_000:  # 50KB threshold
            chunk_size = 10_000  # 10KB chunks
            all_ids = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                all_ids.extend(self._encode_chunk(chunk))
            return all_ids

        return self._encode_chunk(data)

    def _encode_chunk(self, data: bytes) -> List[int]:
        """Encode a small chunk of bytes to token IDs."""
        token_ids = list(data)

        # Apply merges in order learned
        for pair, new_id in self.merges.items():
            token_ids = self._merge_pair_fast(token_ids, pair, new_id)

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
