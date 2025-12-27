"""
Training loop with experiment tracking.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from data.dataset import GutenbergDataset
from experiments.config import ExperimentConfig


class Trainer:
    """Trainer for transformer experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Directories
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{config.output_dir}/logs", exist_ok=True)

        # Dataset
        self.dataset = GutenbergDataset(
            data_path=config.data_path,
            context_length=config.context_length,
            train_split=config.train_split,
            device=str(self.device),
        )

        # Model
        self.model = TransformerLanguageModel(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            context_length=config.context_length,
            dropout=config.dropout,
            pos_enc_type=config.pos_enc_type,
            norm_type=config.norm_type,
            return_attention=config.return_attention,
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {params:,}")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, config.warmup_iters)),
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "gradient_norm": [],
            "step": [],
        }

        # Seed
        torch.manual_seed(config.seed)

    def train(self):
        """Run training loop."""
        print(f"\nTraining for {self.config.max_iters} steps...")

        for step in tqdm(range(self.config.max_iters), desc="Training"):
            # Evaluate periodically
            if step % self.config.eval_interval == 0:
                metrics = self._evaluate()
                self.history["train_loss"].append(metrics["train"])
                self.history["val_loss"].append(metrics["val"])
                self.history["step"].append(step)
                tqdm.write(
                    f"Step {step}: train={metrics['train']:.4f}, val={metrics['val']:.4f}"
                )

            # Training step
            self.model.train()
            x, y = self.dataset.get_batch("train", self.config.batch_size)

            if self.scaler:
                with autocast():
                    _, loss = self.model(x, y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self.model(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Track gradient norm
            if step % 100 == 0:
                total_norm = (
                    sum(
                        p.grad.norm().item() ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                self.history["gradient_norm"].append(total_norm)

        # Final save
        self._save_checkpoint()
        self._save_logs()
        print(f"\nTraining complete! Saved to {self.config.output_dir}")

    def _evaluate(self):
        """Evaluate on train and val sets."""
        self.model.eval()
        results = {}

        with torch.no_grad():
            for split in ["train", "val"]:
                losses = []
                for _ in range(self.config.eval_iters):
                    x, y = self.dataset.get_batch(split, self.config.batch_size)
                    if self.scaler:
                        with autocast():
                            _, loss = self.model(x, y)
                    else:
                        _, loss = self.model(x, y)
                    losses.append(loss.item())
                results[split] = sum(losses) / len(losses)

        return results

    def _save_checkpoint(self):
        """Save model checkpoint."""
        path = f"{self.config.output_dir}/checkpoints/{self.config.name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "vocab_size": self.dataset.vocab_size,
                    "embedding_dim": self.config.embedding_dim,
                    "num_layers": self.config.num_layers,
                    "num_heads": self.config.num_heads,
                    "context_length": self.config.context_length,
                    "dropout": self.config.dropout,
                    "pos_enc_type": self.config.pos_enc_type,
                    "norm_type": self.config.norm_type,
                },
                "final_loss": self.history["val_loss"][-1]
                if self.history["val_loss"]
                else None,
            },
            path,
        )
        print(f"Checkpoint: {path}")

    def _save_logs(self):
        """Save training history."""
        path = f"{self.config.output_dir}/logs/{self.config.name}_history.json"
        with open(path, "w") as f:
            json.dump(
                {"history": self.history, "config": self.config.__dict__},
                f,
                indent=2,
                default=str,
            )
        print(f"Logs: {path}")
