"""
Text generation script.

Usage:
    python generate.py --checkpoint outputs/checkpoints/exp1_absolute_encoding.pt --max_tokens 500
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models.transformer import TransformerLanguageModel
from data.dataset import GutenbergDataset


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--prompt", type=str, default="", help="Starting prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]

    # Load dataset for vocab
    dataset = GutenbergDataset(
        "data/gutenberg_subset/all_books.txt", cfg["context_length"], device=str(device)
    )

    # Build model
    model = TransformerLanguageModel(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        context_length=cfg["context_length"],
        dropout=cfg["dropout"],
        pos_enc_type=cfg["pos_enc_type"],
        norm_type=cfg["norm_type"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: {cfg['pos_enc_type']} pos_enc, {cfg['norm_type']}-norm")

    # Encode prompt
    if args.prompt:
        idx = dataset.encode(args.prompt).unsqueeze(0)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    print(
        f"\nGenerating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...\n"
    )
    generated = model.generate(idx, args.max_tokens, args.temperature, args.top_k)
    text = dataset.decode(generated[0])

    print(text)
    print(f"\n[{len(text)} characters]")


if __name__ == "__main__":
    main()
