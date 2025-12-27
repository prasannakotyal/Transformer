"""
Training loop with experiment tracking.

Supports:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate warmup
- Loss and gradient norm logging
- Checkpoint saving
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.transformer import TransformerLanguageModel
from experiments.config import ExperimentConfig


class Trainer:
    """
    Trainer for transformer experiments.

    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: Experiment configuration
        """
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Setup directories
        self.checkpoint_dir = os.path.join(config.output_dir, "checkpoints/")
        self.log_dir = os.path.join(config.output_dir, "logs/")
        self.vis_dir = os.path.join(config.output_dir, "visualizations/")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # Load data
        from data.dataset import GutenbergDataset

        self.dataset = GutenbergDataset(
            data_path=config.data_path,
            context_length=config.context_length,
            train_split=config.train_split,
            device=str(self.device),
        )

        # Initialize model
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

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

        # Learning rate scheduler with warmup
        self.warmup_iters = config.warmup_iters
        self.total_iters = config.max_iters
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.warmup_iters)
            if step < self.warmup_iters
            else 1.0,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "step": [],
        }

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def train(self):
        """Run full training loop."""
        print(f"\nStarting training for {self.config.max_iters} iterations...")
        print(f"Config: {self.config}")

        for step in tqdm(range(self.config.max_iters), desc="Training"):
            # Evaluation
            if step % self.config.eval_interval == 0:
                metrics = self.evaluate()
                print(
                    f"Step {step}: Train Loss: {metrics['train']:.4f}, "
                    f"Val Loss: {metrics['val']:.4f}"
                )

                self.history["train_loss"].append(metrics["train"])
                self.history["val_loss"].append(metrics["val"])
                self.history["step"].append(step)

            # Training step
            if self.config.use_mixed_precision:
                with autocast():
                    x_batch, y_batch = self.dataset.get_batch(
                        "train", self.config.batch_size
                    )
                    logits, loss = self.model(x_batch, y_batch)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Track gradient norm
                    if step % 100 == 0:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2).item()
                                total_norm += param_norm**2
                        self.history["gradient_norm"].append(total_norm**0.5)
                        self.history["learning_rate"].append(
                            self.scheduler.get_last_lr()[0]
                        )
            else:
                x_batch, y_batch = self.dataset.get_batch(
                    "train", self.config.batch_size
                )
                logits, loss = self.model(x_batch, y_batch)

                loss.backward()

                if (step + 1) % self.config.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if step % 100 == 0:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm += param_norm**2
                    self.history["gradient_norm"].append(total_norm**0.5)
                    self.history["learning_rate"].append(
                        self.scheduler.get_last_lr()[0]
                    )

            self.scheduler.step()

        # Final evaluation
        final_metrics = self.evaluate()
        print(
            f"\nFinal - Train Loss: {final_metrics['train']:.4f}, "
            f"Val Loss: {final_metrics['val']:.4f}"
        )

        # Save final model
        self.save_checkpoint(final_metrics, step=self.config.max_iters)

        # Save training history
        self.save_logs()

        print(f"\nTraining complete! Checkpoint saved to: {self.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")

    def evaluate(self):
        """
        Evaluate on train and validation sets.

        Returns:
            dict: train_loss and val_loss
        """
        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for split in ["train", "val"]:
                losses = torch.zeros(self.config.eval_iters)

                for k in range(self.config.eval_iters):
                    x_batch, y_batch = self.dataset.get_batch(
                        split, self.config.batch_size
                    )

                    if self.config.use_mixed_precision:
                        with autocast():
                            logits, loss = self.model(x_batch, y_batch)
                            losses[k] = self.scaler.scale(loss).item()
                    else:
                        logits, loss = self.model(x_batch, y_batch)
                        losses[k] = loss.item()

                metrics[split] = losses.mean().item()

        self.model.train()
        return metrics

    def save_checkpoint(self, metrics, step):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "vocab_size": self.config.vocab_size,
                "embedding_dim": self.config.embedding_dim,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "context_length": self.config.context_length,
                "pos_enc_type": self.config.pos_enc_type,
                "norm_type": self.config.norm_type,
                "dropout": self.config.dropout,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
            },
        }

        exp_name = getattr(self.config, "name", "experiment")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{exp_name}_step{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def save_logs(self):
        """Save training history to JSON."""
        exp_name = getattr(self.config, "name", "experiment")
        log_path = os.path.join(self.log_dir, f"{exp_name}_history.json")

        with open(log_path, "w") as f:
            json.dump(
                {
                    "history": self.history,
                    "config": {
                        "vocab_size": self.config.vocab_size,
                        "embedding_dim": self.config.embedding_dim,
                        "num_layers": self.config.num_layers,
                        "num_heads": self.config.num_heads,
                        "context_length": self.config.context_length,
                        "pos_enc_type": self.config.pos_enc_type,
                        "norm_type": self.config.norm_type,
                    },
                },
                f,
                indent=2,
            )

        print(f"Logs saved: {log_path}")


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)
