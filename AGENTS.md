# Transformer from Scratch

A minimal, educational implementation of the Transformer architecture from
"Attention Is All You Need" (Vaswani et al., 2017).

## Project Overview

This is a **GPT-style decoder-only transformer** trained on FineWeb-Edu data.
The codebase is intentionally minimal (~1000 lines) to serve as a learning resource.

### Key Features
- **BPE Tokenizer from scratch** - No external tokenizer libraries
- **Self-attention with causal masking** - Core transformer mechanism
- **Multi-head attention** - Parallel attention heads
- **KV-cache for efficient generation** - O(T) instead of O(T²) per token
- **Sinusoidal positional encoding** - From the original paper
- **Mixed precision training** - AMP for faster training on GPUs

## File Structure

```
Transformer/
├── models/
│   ├── __init__.py              # Model exports
│   ├── self_attention.py        # Single attention head (~60 lines)
│   ├── multi_head_attention.py  # Multi-head wrapper (~50 lines)
│   ├── kv_cache.py              # KV cache for inference (~80 lines)
│   └── transformer.py           # Full GPT model (~250 lines)
├── tokenizer.py                 # Byte-level BPE (~150 lines)
├── train.py                     # Training loop (~300 lines)
├── generate.py                  # Text generation (~200 lines)
├── visualize.py                 # All visualizations (~400 lines)
├── data/                        # Downloaded data (gitignored)
├── outputs/                     # Checkpoints and plots (gitignored)
├── requirements.txt
└── AGENTS.md                    # This file
```

## Architecture Details

### Model Configuration (Default)
- Vocabulary: 4096 BPE tokens
- Context length: 256 tokens
- Embedding dimension: 384
- Layers: 6 transformer blocks
- Heads: 6 attention heads per layer
- Parameters: ~10M

### Key Components

1. **SelfAttention** (`models/self_attention.py`)
   - Scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
   - Causal mask for autoregressive modeling
   - Supports KV caching for efficient generation

2. **MultiHeadAttention** (`models/multi_head_attention.py`)
   - Wraps multiple SelfAttention heads
   - Concatenates and projects outputs

3. **Transformer** (`models/transformer.py`)
   - Token embedding + sinusoidal positional encoding
   - N transformer blocks (attention + FFN + residuals + LayerNorm)
   - Pre-norm architecture (more stable than post-norm)
   - Weight tying between embedding and output projection

4. **BPETokenizer** (`tokenizer.py`)
   - Byte-level BPE (starts with 256 byte tokens)
   - Iteratively merges most frequent pairs
   - No regex pre-splitting (simple implementation)

## Usage

### Training
```bash
# Train on FineWeb-Edu (downloads automatically)
python train.py
```

Training takes ~1-2 hours on a T4 GPU for 5000 iterations.

### Generation
```bash
# Generate text
python generate.py --checkpoint outputs/checkpoints/checkpoint_final.pt

# With custom prompt
python generate.py --checkpoint outputs/checkpoints/checkpoint_final.pt --prompt "Once upon a time"

# Compare KV-cache speedup
python generate.py --checkpoint outputs/checkpoints/checkpoint_final.pt --compare-speed
```

### Visualization
```bash
# Create all plots
python visualize.py --all

# Specific plots
python visualize.py --plot training_curves
python visualize.py --plot attention_heatmap
python visualize.py --plot embeddings
python visualize.py --plot kv_cache_speedup
```

## Hyperparameters

All hyperparameters are at the top of `train.py`:

```python
# Model
VOCAB_SIZE = 4096
CONTEXT_LENGTH = 256
EMBEDDING_DIM = 384
NUM_LAYERS = 6
NUM_HEADS = 6

# Training
BATCH_SIZE = 64
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
```

No config files - just edit the Python directly.

## Development Guidelines

### Code Style
- Keep it simple and readable
- Prefer explicit over implicit
- Document key equations/formulas in comments
- No unnecessary abstractions

### Adding Features
When adding new features:
1. Keep the file structure flat
2. Add hyperparameters to top of relevant file
3. Update this AGENTS.md

### Testing Changes
```bash
# Quick model test
python -c "from models import Transformer; import torch; m = Transformer(1000); print(m.count_parameters())"

# Tokenizer test
python tokenizer.py

# Full model test
python models/transformer.py
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Pre-norm, GELU activation
- [Karpathy's minbpe](https://github.com/karpathy/minbpe) - BPE tokenizer reference
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Training loop reference
