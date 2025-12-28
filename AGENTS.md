# Transformer from Scratch

Educational GPT-style decoder-only transformer implementation from "Attention Is All You Need" (2017). Minimal codebase (~1000 lines) designed for learning.

## Build/Test/Lint

```bash
# Install dependencies
pip install -r requirements.txt

# Run module tests (each has __main__ block)
python models/transformer.py
python models/self_attention.py
python tokenizer.py

# Quick smoke test
python -c "from models import Transformer; import torch; m = Transformer(1000); print(m.count_parameters())"

# Train model
python train.py

# Generate text
python generate.py --checkpoint outputs/checkpoints/checkpoint_final.pt --prompt "Hello"

# Create visualizations
python visualize.py --all
```

**Note:** No formal test framework (pytest) or linter configured. Use manual testing via `__main__` blocks. Add test files to `tests/` if setting up pytest.

**Tokenizer:** Uses tiktoken (OpenAI's BPE) with cl100k_base encoding for fast, production-ready tokenization. Focus on transformer model implementation, not tokenization.

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local
- Alphabetical within groups
- Type hints: `from typing import Optional, Tuple, List, Dict`
- No wildcard imports (`from X import *`)

```python
# Good
import math
import json
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
```

### Type Hints
- Required on all functions and methods
- Use `typing` module: `Optional`, `Tuple`, `List`, `Dict`
- Return types must be annotated

```python
def forward(
    self,
    x: torch.Tensor,
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `SelfAttention`, `TransformerBlock`)
- Functions/methods: `snake_case` (e.g., `train_tokenizer`, `get_batch`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `BATCH_SIZE`, `LEARNING_RATE`)
- Private methods: `_prefix` (e.g., `_train_full`, `_encode_optimized`)
- Variables: `snake_case` (e.g., `token_ids`, `attn_weights`)

### Docstrings
- Required on all classes and public methods
- Use Args/Returns sections
- Document key equations/formulas
- Keep descriptions concise

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through attention layer.

    Args:
        x: Input tensor (batch, seq_len, embedding_dim)

    Returns:
        Output tensor (batch, seq_len, embedding_dim)
    """
```

### Formatting
- 4-space indentation
- Max 100 characters per line
- Section dividers in long files: `#=============` or `# Section Name`
- No unnecessary comments
- Document complex math with equations

### Error Handling
- Use `assert` for validation and invariants
- Minimal try-except (only for external I/O like downloads)
- Clear error messages
- Fallback mechanisms for critical paths (e.g., dataset download failures)

### Architecture Principles
1. **Explicit over implicit**: No magic, clear code flow
2. **Minimal abstractions**: Keep code flat and readable
3. **Educational focus**: Comment equations and design choices
4. **Production-grade tokenization**: Use tiktoken for fast BPE tokenization (100k+ vocab, handles large datasets)
5. **Pre-norm architecture**: LayerNorm before attention/FFN for stability

### Hyperparameters
- Place at top of files (no config files)
- Group by category (Model, Training, Data)
- Comment values and rationale

```python
# Model
VOCAB_SIZE = 4096  # BPE vocabulary size
CONTEXT_LENGTH = 256  # Maximum sequence length
EMBEDDING_DIM = 384  # Model dimension

# Training
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0
```

## Adding Features

1. Keep file structure flat (avoid deep directories)
2. Add hyperparameters to top of relevant file
3. Update this AGENTS.md with new patterns
4. Add `__main__` test block to new modules
5. Follow existing docstring and type hint conventions

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Pre-norm, GELU activation
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's fast BPE tokenizer
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Training loop reference
