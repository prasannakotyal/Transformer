"""
Main training script entry point.

Usage:
    python train.py --config configs/exp1_pos_enc.yaml --variant absolute_encoding
"""

import argparse
import yaml
from experiments.trainer import Trainer
from experiments.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Experiment name (overrides config)"
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Variant name for this run"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Override name if provided
    if args.name:
        config_dict["name"] = args.name
    if args.variant:
        config_dict["name"] = f"{config_dict.get('name', 'exp')}_{args.variant}"

    config = ExperimentConfig(**config_dict)

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
