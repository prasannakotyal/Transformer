"""
Text generation script.

Usage:
    python generate.py --checkpoint outputs/checkpoints/exp1_absolute_step2000.pt --max_tokens 500
"""

import argparse
import torch
from models.transformer import TransformerLanguageModel
from data.dataset import GutenbergDataset


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--prompt", type=str, default="", help="Starting prompt")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for generation"
    )
    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["config"]

    # Reconstruct model
    model = TransformerLanguageModel(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        context_length=model_config["context_length"],
        dropout=model_config["dropout"],
        pos_enc_type=model_config["pos_enc_type"],
        norm_type=model_config["norm_type"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(
        f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(
        f"Architecture: {model_config['pos_enc_type']} PE, {model_config['norm_type']}-norm"
    )
    print(f"Final loss: {checkpoint['metrics']['val_loss']:.4f}")

    # Encode prompt
    dataset = GutenbergDataset(
        "data/gutenberg_subset/all_books.txt",
        model_config["context_length"],
        device=str(device),
    )

    if args.prompt:
        idx = dataset.encode(args.prompt).unsqueeze(0).to(device)
    else:
        idx = torch.tensor([[0]], dtype=torch.long).to(device)

    print(f"\nGenerating {args.max_tokens} tokens...")
    print(f"Temperature: {args.temperature}")

    # Generate
    with torch.no_grad():
        generated, attention_history = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            store_attention=False,
        )

    # Decode
    text = dataset.decode(generated[0])
    print(f"\nGenerated text:\n")
    print(text)
    print(f"\nTotal tokens generated: {len(text)}")


if __name__ == "__main__":
    main()
