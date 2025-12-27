# Transformer Experiments - Agent Guidelines

This repository contains transformer architectural experiments with ablation studies. This document helps AI agents understand the codebase and follow conventions.

## Project Overview

**Goal**: Demonstrate deep transformer understanding through systematic architectural experiments (positional encodings, normalization, attention analysis).

**Architecture**: GPT-style decoder-only transformer, character-level tokenization on Project Gutenberg corpus.

**Model Size**: ~3.5M parameters (6-layer, 384-dim, 6-head, 256-context)

## Build/Test Commands

**Installation**:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Data Download**:
```bash
# Download 10 curated books from Project Gutenberg (~5M characters)
python data/dataset.py
```

**Training**:
```bash
# Run single experiment
python train.py --config configs/exp1_pos_enc.yaml --variant absolute_encoding

# Run all experiments (7 variants total)
python experiments/ablation.py --all

# For Kaggle (T4x2 recommended), use GPU T4 x2 accelerator
```

**Generation**:
```bash
# Generate text from trained checkpoint
python generate.py --checkpoint outputs/checkpoints/exp1_absolute_step2000.pt --max_tokens 500
```

**Visualization**:
```bash
# Generate all visualizations (loss GIFs, attention heatmaps, dashboard)
python experiments/visualize.py --create-all
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import torch
import torch.nn as nn
import torch.nn.functional as F

# Third-party imports
import yaml
from dataclasses import dataclass
from typing import Optional

# Local imports (use absolute paths within project)
from models.transformer import TransformerLanguageModel
from data.dataset import GutenbergDataset
from experiments.config import ExperimentConfig
```

### Type Hints
- Use inline type hints for function parameters
- Use `Optional[torch.Tensor]` for optional returns
- Keep simple - no complex generics
- Example:
```python
def forward(self, x: torch.Tensor, return_attention: bool = False):
    # Type hints for parameters
```

### Naming Conventions
- **Classes**: PascalCase (`TransformerBlock`, `MultiHeadAttention`)
- **Functions/Methods**: snake_case (`forward`, `get_batch`, `train`)
- **Variables**: snake_case (`embedding_dim`, `context_length`, `learning_rate`)
- **Constants**: UPPER_SNAKE_CASE (`COLOR_SCHEME`, `MAX_SEQ_LEN`)
- **Private methods**: prefix with underscore (`_apply_rotary`, `_load_data`)

### File Organization
- `models/`: Model architecture (modular components)
- `data/`: Dataset loading and preprocessing
- `experiments/`: Training, ablation, visualization
- `configs/`: Experiment configurations in YAML
- `outputs/`: Generated checkpoints, logs, visualizations

### Class Structure
```python
class TransformerLanguageModel(nn.Module):
    def __init__(self, ...):
        """Clear docstring explaining purpose."""
        super().__init__()
        # Initialize components
        # Register buffers
        
    def forward(self, ...):
        """Forward pass docstring."""
        # Logic
        return output
```

### Error Handling
- Use try-except for external operations (file I/O, network requests)
- Use assertions for internal logic checks (tensor shapes, parameter ranges)
- Log errors with descriptive messages
- Example:
```python
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except Exception as e:
    print(f"Error: {e}")
```

### PyTorch Best Practices
- Move model to device after initialization: `model.to(device)`
- Use `torch.no_grad()` context for inference/evaluation
- Use `model.eval()` for evaluation, `model.train()` for training
- Enable mixed precision with `torch.cuda.amp.autocast()` for faster training
- Use `torch.manual_seed()` for reproducibility

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    # Forward pass
    loss = criterion(output, targets)
    
# Backward pass
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Documentation
- Use triple double quotes for docstrings: `"""Description"""`
- Include Args, Returns sections
- One-liner comments: `# Positional encoding```
- Multi-line comments: Use `#` for each line

### Configuration System
```python
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    vocab_size: int = 100
    embedding_dim: int = 384
    num_layers: int = 6
    # ... more config fields
```

**Usage**:
```python
# Load from YAML
with open('configs/exp1.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig(**config_dict)
```

## Model Architecture Details

### Components

1. **Attention** (`models/attention.py`):
   - `Head`: Single self-attention head with causal masking
   - `MultiHeadAttention`: Concatenates multiple heads
   - Supports `return_attention=True` for visualization

2. **Positional Encodings** (`models/positional_encodings.py`):
   - `NoPositionalEncoding`: Identity (baseline)
   - `AbsolutePositionalEncoding`: Sinusoidal absolute encoding
   - `RotaryPositionalEncoding`: Rotary embeddings (RoPE)

3. **Feed Forward** (`models/feed_forward.py`):
   - Two-layer MLP with GELU activation
   - Hidden dimension: 4x embedding_dim

4. **Transformer Block** (`models/transformer.py`):
   - Pre-norm or post-norm (configurable)
   - Attention + FFN pattern
   - Residual connections

5. **Full Model** (`models/transformer.py`):
   - Configurable GPT-style decoder
   - Token + position embeddings
   - Stack of transformer blocks
   - Final LM head

### Configuration Variants

**Positional Encoding Types**:
- `none`: No position information (identity)
- `absolute`: Sinusoidal absolute positions
- `rotary`: Rotary position encoding (RoPE)

**Normalization Types**:
- `pre`: Pre-layer normalization (modern, more stable)
- `post`: Post-layer normalization (original)

## Training Details

### Training Loop Pattern

```python
for step in range(max_iters):
    # Evaluation
    if step % eval_interval == 0:
        metrics = evaluate()
        log_metrics(step, metrics)
    
    # Training step
    x_batch, y_batch = get_batch('train', batch_size)
    
    if use_mixed_precision:
        with autocast():
            logits, loss = model(x_batch, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        logits, loss = model(x_batch, y_batch)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
```

### Metrics Tracked

- `train_loss`: Training cross-entropy loss
- `val_loss`: Validation cross-entropy loss  
- `learning_rate`: Current learning rate
- `gradient_norm`: L2 norm of all gradients (every 100 steps)

### Checkpointing

```python
checkpoint = {
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'config': config_dict
}
torch.save(checkpoint, f'outputs/checkpoints/exp_name_step{step}.pt')
```

## Experiment Protocol

### Experiment 1: Positional Encoding
Run 3 variants to understand position encoding impact:
1. No position encoding (identity)
2. Absolute sinusoidal encoding
3. Rotary encoding (RoPE)

**Metrics**: Loss convergence speed, final validation loss, generation quality

### Experiment 2: Normalization
Compare pre-norm vs post-norm for training stability:
1. Pre-layer normalization (modern standard)
2. Post-layer normalization (original)

**Metrics**: Gradient norm stability, loss oscillations, maximum achievable depth

### Experiment 3: Attention Analysis
Train model with `return_attention=True` and analyze:
1. Head diversity (correlations between heads)
2. Position bias (local vs global attention)
3. Specialization (what each head learns)

**Metrics**: Attention pattern visualizations, head correlation matrix

## Dataset Information

**Source**: Project Gutenberg corpus
**Books**: 10 curated classics (literature, philosophy, sci-fi, mystery)
**Total Characters**: ~5M
**Tokenization**: Character-level (vocabulary ~100 chars)
**Split**: 90% train, 10% validation

## Visualization Guidelines

### Plot Styling
- Dark theme: Background `#1e1e2e`, grid `#2d2d2d`
- Colors: Train `#4a90e2`, Val `#f39c12`
- Font: Arial or sans-serif
- DPI: 300 for publication quality
- Use seaborn for clean styling

### Visualizations Generated

1. **Loss Comparison GIF**: Animated loss curves comparing experiments
2. **Generation GIFs**: Progressive text generation with blinking cursor
3. **Attention Heatmaps**: 6x2 grid showing different head patterns
4. **Gradient Norm Plots**: Training stability comparison
5. **Training Dashboard**: Publication-ready summary figure

## Output Locations

- **Checkpoints**: `outputs/checkpoints/` - Model weights and optimizer states
- **Logs**: `outputs/logs/` - JSON files with training history
- **Visualizations**: `outputs/visualizations/` - PNGs, GIFs

## Environment Setup

### Kaggle T4x2 (Recommended)
- GPU: 2x NVIDIA T4 (16GB each = 32GB total)
- Accelerator: "GPU T4 x2"
- Batch size: 64
- Training time per experiment: ~40-60 minutes

### Local RTX 3050 (4GB VRAM)
- GPU: NVIDIA RTX 3050 Laptop (4GB VRAM)
- Batch size: 32 (use gradient accumulation for effective 64)
- Training time per experiment: ~1.5-2 hours
- Use mixed precision to reduce memory

## Common Patterns

### Reading Config
```python
import yaml

with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

from experiments.config import ExperimentConfig
config = ExperimentConfig(**config_dict)
```

### Model Instantiation
```python
from models.transformer import TransformerLanguageModel

model = TransformerLanguageModel(
    vocab_size=config.vocab_size,
    embedding_dim=config.embedding_dim,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    context_length=config.context_length,
    pos_enc_type=config.pos_enc_type,
    norm_type=config.norm_type,
    return_attention=config.return_attention
)
```

### Running Experiments

```bash
# All experiments
python experiments/ablation.py --all

# Single experiment group
python experiments/ablation.py --exp exp1

# With visualization
python experiments/ablation.py --all --visualize
```

## Testing

No formal test suite currently exists. To validate code:

1. **Smoke Tests**: Import all modules and instantiate
```bash
python3 -c "from models.transformer import TransformerLanguageModel; print('Import OK')"
```

2. **Forward Pass Test**: Run forward pass with dummy data
```python
import torch
from models.transformer import TransformerLanguageModel

model = TransformerLanguageModel(vocab_size=100, num_layers=2)
x = torch.randint(0, 100, (1, 10))
logits, loss = model(x)
assert logits.shape == (1, 10, 100)
```

3. **Generation Test**: Test autoregressive generation
```bash
python generate.py --checkpoint <path> --max_tokens 10 --seed 42
```

## Git Workflow

This project uses standard Git workflow:
- Main branch contains stable code
- Experiments tracked via commit messages
- Outputs/ directory gitignored
- Large files (checkpoints, .pt files) not committed

## Notes for Agents

- This is a research/experimental project, not production code
- Code is organized for easy modification and experimentation
- Use config-driven experiments rather than hardcoded values
- All model architecture variants supported via parameters
- Visualizations are polished and publication-ready
- Dataset downloads automatically via script

## Troubleshooting

### Common Issues

**CUDA OOM**:
- Reduce batch_size or gradient_accumulation
- Use smaller context_length

**Import Errors**:
- Ensure running from repository root
- Check Python path includes project directory

**Training Slow**:
- Verify GPU is being used (not CPU)
- Check batch_size is appropriate for hardware
- Enable mixed precision

### Key Commands to Check

```bash
# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check file structure
tree -L 2 -I '__pycache__|*.pyc|.git'

# Run quick test
python3 experiments/ablation.py --exp exp1 --variant absolute_encoding 2>&1 | head -20
```

## Future Improvements

- Add formal unit tests (pytest)
- Implement gradient checkpointing for even larger models
- Add WandB/MLflow integration for experiment tracking
- Add data augmentation (text dropout, token masking)
- Implement beam search decoding
- Add token-level tokenizer (BPE) for better performance

---

**Last Updated**: December 2025
**Purpose**: Guide for AI agents working on Transformer experiments
