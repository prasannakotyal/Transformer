"""
Ablation experiment runner.
Runs all architectural experiments and compares results.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiments.trainer import Trainer
from experiments.config import ExperimentConfig


def run_experiment(config_path: str, name: str, overrides: dict):
    """Run single experiment variant."""
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict.update(overrides)
    config_dict["name"] = name

    config = ExperimentConfig(**config_dict)
    trainer = Trainer(config)
    trainer.train()

    print(f"Completed: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run transformer ablation experiments")
    parser.add_argument(
        "--exp", type=str, help="Run specific experiment (exp1, exp2, exp3)"
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if args.all:
        print("Running all experiments (5 variants total)...")

        # Experiment 1: Positional Encodings (2 variants)
        run_experiment(
            "configs/exp1_pos_enc.yaml",
            "exp1_none_encoding",
            {"pos_enc_type": "none", "norm_type": "pre"},
        )
        run_experiment(
            "configs/exp1_pos_enc.yaml",
            "exp1_absolute_encoding",
            {"pos_enc_type": "absolute", "norm_type": "pre"},
        )

        # Experiment 2: Normalization (2 variants)
        run_experiment(
            "configs/exp2_norm.yaml",
            "exp2_pre_norm",
            {"pos_enc_type": "absolute", "norm_type": "pre"},
        )
        run_experiment(
            "configs/exp2_norm.yaml",
            "exp2_post_norm",
            {"pos_enc_type": "absolute", "norm_type": "post"},
        )

        # Experiment 3: Attention Analysis (1 variant)
        run_experiment(
            "configs/exp3_attention.yaml",
            "exp3_attention_analysis",
            {"pos_enc_type": "absolute", "norm_type": "pre", "return_attention": True},
        )

        print("\n" + "=" * 60)
        print("All experiments complete!")
        print("=" * 60)

    elif args.exp:
        experiments = {
            "exp1": [
                (
                    "configs/exp1_pos_enc.yaml",
                    "exp1_none_encoding",
                    {"pos_enc_type": "none"},
                ),
                (
                    "configs/exp1_pos_enc.yaml",
                    "exp1_absolute_encoding",
                    {"pos_enc_type": "absolute"},
                ),
            ],
            "exp2": [
                ("configs/exp2_norm.yaml", "exp2_pre_norm", {"norm_type": "pre"}),
                ("configs/exp2_norm.yaml", "exp2_post_norm", {"norm_type": "post"}),
            ],
            "exp3": [
                (
                    "configs/exp3_attention.yaml",
                    "exp3_attention_analysis",
                    {"return_attention": True},
                ),
            ],
        }

        if args.exp in experiments:
            for config_path, name, overrides in experiments[args.exp]:
                run_experiment(config_path, name, overrides)
        else:
            print(f"Unknown experiment: {args.exp}")
            print(f"Available: {list(experiments.keys())}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
