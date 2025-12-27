"""
Ablation experiment runner.

Runs all architectural experiments and compares results.
"""

import os
import argparse
import yaml
from experiments.trainer import Trainer
from experiments.config import ExperimentConfig
from experiments.visualize import create_all_visualizations


def run_variant(config_path: str, variant_name: str, variant_config: dict):
    """
    Run single variant of an experiment.

    Args:
        config_path: Path to base config file
        variant_name: Name of this variant
        variant_config: Dict of config overrides
    """
    print(f"\n{'=' * 60}")
    print(f"Running variant: {variant_name}")
    print(f"{'=' * 60}")

    # Load base config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply variant overrides
    config_dict.update(variant_config)
    config_dict["name"] = f"{config_dict.get('name', 'exp')}_{variant_name}"

    # Create config object
    config = ExperimentConfig(**config_dict)

    # Run training
    trainer = Trainer(config)
    trainer.train()

    print(f"Variant {variant_name} complete!")
    print(f"{'=' * 60}")


def run_experiment_group(config_path: str, group_name: str, variants: dict):
    """
    Run all variants for an experiment group.

    Args:
        config_path: Path to base config file
        group_name: Name of experiment group
        variants: Dict of {variant_name: variant_config}
    """
    print(f"\n{'#' * 60}")
    print(f"# Experiment Group: {group_name}")
    print(f"{'#' * 60}\n")

    for variant_name, variant_config in variants.items():
        try:
            run_variant(config_path, variant_name, variant_config)
        except Exception as e:
            print(f"Error in {variant_name}: {e}")
            continue

    print(f"\n{'#' * 60}")
    print(f"# All variants for {group_name} complete!")
    print(f"{'#' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run transformer ablation experiments")
    parser.add_argument(
        "--config", type=str, required=False, help="Path to config YAML file"
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=False,
        help="Experiment group to run (exp1, exp2, exp3)",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after training",
    )
    args = parser.parse_args()

    if args.all:
        # Run all experiment groups
        print("Running all experiments...")

        # Experiment 1: Positional Encodings
        run_experiment_group(
            "configs/exp1_pos_enc.yaml",
            "Positional Encoding",
            {
                "none": {"pos_enc_type": "none", "name": "none_encoding"},
                "absolute": {"pos_enc_type": "absolute", "name": "absolute_encoding"},
                "rotary": {"pos_enc_type": "rotary", "name": "rotary_encoding"},
            },
        )

        # Experiment 2: Normalization
        run_experiment_group(
            "configs/exp2_norm.yaml",
            "Normalization",
            {
                "pre": {"norm_type": "pre", "name": "pre_norm"},
                "post": {"norm_type": "post", "name": "post_norm"},
            },
        )

        # Experiment 3: Attention Analysis
        run_experiment_group(
            "configs/exp3_attention.yaml",
            "Attention Analysis",
            {
                "baseline": {"return_attention": True, "name": "baseline_attention"},
            },
        )

        print("\nAll experiments complete!")

        if args.visualize:
            print("\nGenerating visualizations...")
            create_all_visualizations()
            print("Visualizations complete!")

    elif args.exp:
        # Run specific experiment group
        exp_map = {
            "exp1": ("Positional Encoding", "configs/exp1_pos_enc.yaml"),
            "exp2": ("Normalization", "configs/exp2_norm.yaml"),
            "exp3": ("Attention Analysis", "configs/exp3_attention.yaml"),
        }

        if args.exp in exp_map:
            group_name, config_path = exp_map[args.exp]

            # Load config to get variants
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            variants = config_dict.get("variants", {})

            run_experiment_group(config_path, group_name, variants)
        else:
            print(f"Unknown experiment: {args.exp}")
            print(f"Available: {list(exp_map.keys())}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
