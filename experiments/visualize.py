"""
Visualizations for transformer experiments.
Creates loss comparison plots and training dashboards.
"""

import os
import json
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


# Dark theme colors
COLORS = {
    "background": "#1e1e2e",
    "text": "#e0e0e0",
    "grid": "#2d2d2d",
    "train": "#4a90e2",
    "val": "#f39c12",
    "accent": "#00b894",
}


def smooth(data: List[float], window: int = 20) -> List[float]:
    """Smooth data with moving average."""
    if len(data) < window:
        return data
    return [np.mean(data[max(0, i - window) : i + 1]) for i in range(len(data))]


def plot_loss_comparison(
    logs_dir: str = "outputs/logs/",
    save_path: str = "outputs/visualizations/loss_comparison.png",
):
    """Create comparison plot of loss curves across experiments."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load all experiment logs
    experiments = {}
    for f in os.listdir(logs_dir):
        if f.endswith("_history.json"):
            with open(os.path.join(logs_dir, f)) as file:
                data = json.load(file)
                name = f.replace("_history.json", "")
                experiments[name] = data["history"]

    if not experiments:
        print("No experiment logs found.")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    for name, history in experiments.items():
        steps = history.get("step", list(range(len(history["val_loss"]))))
        val_loss = smooth(history["val_loss"])
        ax.plot(steps[: len(val_loss)], val_loss, label=name, linewidth=2)

    ax.set_xlabel("Step", color=COLORS["text"])
    ax.set_ylabel("Validation Loss", color=COLORS["text"])
    ax.set_title(
        "Experiment Comparison", color=COLORS["text"], fontsize=14, fontweight="bold"
    )
    ax.legend(facecolor=COLORS["grid"], edgecolor=COLORS["text"])
    ax.grid(True, color=COLORS["grid"], alpha=0.3)
    ax.tick_params(colors=COLORS["text"])

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradient_norms(
    logs_dir: str = "outputs/logs/",
    save_path: str = "outputs/visualizations/gradient_norms.png",
):
    """Plot gradient norms to show training stability."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    experiments = {}
    for f in os.listdir(logs_dir):
        if f.endswith("_history.json"):
            with open(os.path.join(logs_dir, f)) as file:
                data = json.load(file)
                name = f.replace("_history.json", "")
                if data["history"].get("gradient_norm"):
                    experiments[name] = data["history"]["gradient_norm"]

    if not experiments:
        print("No gradient norm data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    for name, grad_norms in experiments.items():
        ax.plot(smooth(grad_norms), label=name, linewidth=2)

    ax.set_xlabel("Step (x100)", color=COLORS["text"])
    ax.set_ylabel("Gradient Norm", color=COLORS["text"])
    ax.set_title(
        "Training Stability (Gradient Norms)",
        color=COLORS["text"],
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yscale("log")
    ax.legend(facecolor=COLORS["grid"], edgecolor=COLORS["text"])
    ax.grid(True, color=COLORS["grid"], alpha=0.3)
    ax.tick_params(colors=COLORS["text"])

    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"Saved: {save_path}")


def create_summary(
    logs_dir: str = "outputs/logs/",
    save_path: str = "outputs/visualizations/summary.txt",
):
    """Create text summary of experiment results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results = []
    for f in os.listdir(logs_dir):
        if f.endswith("_history.json"):
            with open(os.path.join(logs_dir, f)) as file:
                data = json.load(file)
                name = f.replace("_history.json", "")
                final_val = (
                    data["history"]["val_loss"][-1]
                    if data["history"]["val_loss"]
                    else None
                )
                final_train = (
                    data["history"]["train_loss"][-1]
                    if data["history"]["train_loss"]
                    else None
                )
                results.append((name, final_train, final_val))

    results.sort(key=lambda x: x[2] if x[2] else float("inf"))

    with open(save_path, "w") as f:
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Experiment':<35} {'Train Loss':>10} {'Val Loss':>10}\n")
        f.write("-" * 55 + "\n")
        for name, train, val in results:
            train_str = f"{train:.4f}" if train else "N/A"
            val_str = f"{val:.4f}" if val else "N/A"
            f.write(f"{name:<35} {train_str:>10} {val_str:>10}\n")

    print(f"Saved: {save_path}")


def create_all_visualizations():
    """Generate all visualizations from experiment logs."""
    print("\nGenerating visualizations...")
    plot_loss_comparison()
    plot_gradient_norms()
    create_summary()
    print("\nAll visualizations complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment visualizations")
    parser.add_argument(
        "--create-all", action="store_true", help="Generate all visualizations"
    )
    args = parser.parse_args()

    if args.create_all:
        create_all_visualizations()
    else:
        parser.print_help()
