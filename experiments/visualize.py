"""
Professional visualizations for transformer experiments.

Creates:
- Loss comparison GIFs with smooth curves
- Progressive text generation GIFs with cursor animation
- Attention heatmaps with dark theme styling
- Training dashboard with publication-ready styling
"""

import os
import json
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import imageio
from PIL import Image, ImageDraw, ImageFont


COLOR_SCHEME = {
    'background': '#1e1e2e',
    'grid': '#2d2d2d',
    'text': '#e0e0e0',
    'train_loss': '#4a90e2',
    'val_loss': '#f39c12',
    'accent1': '#00b894',
    'accent2': '#00ff88',
    'accent3': '#ff5722',
}


def smooth_data(data: List[float], window: int = 50) -> List[float]:
    """
    Smooth data using moving average.
    
    Args:
        data: List of values
        window: Window size for smoothing
        
    Returns:
        Smoothed list
    """
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def setup_seaborn_style():
    """Configure seaborn for dark theme styling."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def plot_loss_comparison(experiment_logs: Dict[str, List[dict]], save_path: str):
    """
    Create comparison plot of loss curves across experiments.
    
    Args:
        experiment_logs: Dict of {variant_name: [history dicts]}
        save_path: Where to save the plot
    """
    setup_seaborn_style()
    
    fig, axes = plt.subplots(1, len(experiment_logs), 
                                figsize=(6 * len(experiment_logs), 5),
                                facecolor=COLOR_SCHEME['background'])
    fig.suptitle('Positional Encoding Comparison', fontsize=16, 
                  color=COLOR_SCHEME['text'], weight='bold')
    
    for idx, (exp_name, histories) in enumerate(experiment_logs.items()):
        # Combine all histories for this experiment
        all_train_loss = []
        all_val_loss = []
        all_steps = []
        
        for history in histories:
            all_train_loss.extend(history['history']['train_loss'])
            all_val_loss.extend(history['history']['val_loss'])
            all_steps.extend(history['history']['step'])
        
        # Sort by step
        combined = list(zip(all_steps, all_train_loss, all_val_loss))
        combined.sort(key=lambda x: x[0])
        all_steps, all_train_loss, all_val_loss = zip(*combined)
        
        # Smooth curves
        train_smooth = smooth_data(all_train_loss)
        val_smooth = smooth_data(all_val_loss)
        
        axes[idx].plot(all_steps, train_smooth, 
                     color=COLOR_SCHEME['train_loss'], 
                     linewidth=2.5, alpha=0.8, label='Train')
        axes[idx].plot(all_steps, val_smooth, 
                     color=COLOR_SCHEME['val_loss'], 
                     linewidth=2.5, alpha=0.8, label='Validation')
        
        axes[idx].set_title(exp_name.replace('_', ' ').title(), 
                          fontsize=12, color=COLOR_SCHEME['text'])
        axes[idx].set_xlabel('Training Step', color=COLOR_SCHEME['text'])
        axes[idx].set_ylabel('Loss', color=COLOR_SCHEME['text'])
        axes[idx].legend(loc='upper right', facecolor=COLOR_SCHEME['grid'])
        axes[idx].grid(True, color=COLOR_SCHEME['grid'], alpha=0.3)
        axes[idx].set_facecolor(COLOR_SCHEME['background'])
        axes[idx].tick_params(colors=COLOR_SCHEME['text'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
    plt.close()
    print(f"Loss comparison saved: {save_path}")


def plot_gradient_norms(experiment_logs: Dict[str, List[dict]], save_path: str):
    """
    Create comparison plot of gradient norms.
    
    Shows training stability between pre-norm and post-norm.
    """
    setup_seaborn_style()
    
    fig, axes = plt.subplots(1, len(experiment_logs), 
                                figsize=(6 * len(experiment_logs), 5),
                                facecolor=COLOR_SCHEME['background'])
    fig.suptitle('Gradient Norm Comparison', fontsize=16, 
                  color=COLOR_SCHEME['text'], weight='bold')
    
    for idx, (exp_name, histories) in enumerate(experiment_logs.items()):
        # Combine gradient norms
        all_grad_norms = []
        all_steps = []
        
        for history in histories:
            all_grad_norms.extend(history['history']['gradient_norm'])
            all_steps.extend(history['history']['step'])
        
        combined = list(zip(all_steps, all_grad_norms))
        combined.sort(key=lambda x: x[0])
        all_steps, all_grad_norms = zip(*combined)
        
        # Smooth
        grad_smooth = smooth_data(all_grad_norms)
        
        # Plot
        axes[idx].plot(all_steps, grad_smooth, 
                     color=COLOR_SCHEME['accent2'], 
                     linewidth=2.5, alpha=0.8)
        
        axes[idx].set_title(exp_name.replace('_', ' ').title(), 
                          fontsize=12, color=COLOR_SCHEME['text'])
        axes[idx].set_xlabel('Training Step', color=COLOR_SCHEME['text'])
        axes[idx].set_ylabel('Gradient Norm', color=COLOR_SCHEME['text'])
        axes[idx].grid(True, color=COLOR_SCHEME['grid'], alpha=0.3)
        axes[idx].set_facecolor(COLOR_SCHEME['background'])
        axes[idx].tick_params(colors=COLOR_SCHEME['text'])
        axes[idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
    plt.close()
    print(f"Gradient norm comparison saved: {save_path}")


def create_generation_gif(text: str, max_tokens: int, 
                       fps: int = 30, save_path: str):
    """
    Create progressive text generation GIF with blinking cursor.
    
    Shows tokens appearing one-by-one with smooth animation.
    """
    from data.dataset import GutenbergDataset
    dataset = GutenbergDataset('data/gutenberg_subset/all_books.txt', 256, device='cpu')
    
    width = 1200
    height = 600
    frames = []
    
    for i in range(min(max_tokens, len(text))):
        current_text = text[:i + 1]
        
        # Create frame
        img = Image.new('RGB', (width, height), color=COLOR_SCHEME['background'])
        draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((50, 30), f"Generated Text (token {i + 1}/{max_tokens})",
                fill=COLOR_SCHEME['accent1'], 
                font=ImageFont.truetype("arial.ttf", 20))
        
        # Text with wrapping
        max_chars_per_line = 80
        lines = []
        for j in range(0, len(current_text), max_chars_per_line):
            lines.append(current_text[j:j + max_chars_per_line])
        
        y_offset = 100
        for line in lines:
            draw.text((50, y_offset), line,
                    fill=COLOR_SCHEME['text'],
                    font=ImageFont.truetype("arial.ttf", 24))
            y_offset += 40
        
        # Blinking cursor
        cursor_x = 50 + (len(current_text) % max_chars_per_line) * 10
        cursor_y = y_offset
        if (i % 20) < 10:  # Blink effect
            draw.text((cursor_x, cursor_y), '▌',
                    fill=COLOR_SCHEME['accent2'],
                    font=ImageFont.truetype("arial.ttf", 24))
        
        # Progress bar
        progress = (i + 1) / max_tokens
        draw.rectangle([50, height - 50, width * progress, 10],
                   outline=COLOR_SCHEME['accent3'], width=2)
        
        frames.append(np.array(img))
    
    # Save as GIF
    imageio.mimsave(save_path, frames, fps=fps, duration=1000 // fps)
    print(f"Generation GIF saved: {save_path}")


def visualize_attention_heatmaps(attention_weights: np.ndarray, 
                            tokens: List[str],
                            save_path: str,
                            num_heads: int = 6):
    """
    Create attention heatmap visualization.
    
    Args:
        attention_weights: (num_heads, seq_len, seq_len) attention matrices
        tokens: List of token strings
        save_path: Where to save
        num_heads: Number of heads to visualize
    """
    setup_seaborn_style()
    
    # Calculate grid size
    cols = min(3, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, 
                                figsize=(6 * cols, 4 * rows),
                                facecolor=COLOR_SCHEME['background'])
    fig.suptitle('Multi-Head Attention Patterns', fontsize=16, 
                  color=COLOR_SCHEME['text'], weight='bold')
    
    for head_idx in range(min(num_heads, rows * cols)):
        row = head_idx // cols
        col = head_idx % cols
        
        if head_idx >= num_heads:
            break
        
        # Get attention for this head (average over batch)
        attn = attention_weights[head_idx, 0].mean(axis=0)
        
        im = axes[row, col].imshow(attn, cmap='viridis', 
                                       aspect='auto', vmin=0, vmax=1)
        axes[row, col].set_title(f'Head {head_idx}', 
                                 fontsize=10, color=COLOR_SCHEME['text'])
        
        # Add token labels for first and last few
        if len(tokens) <= 20:
            axes[row, col].set_xticks(range(len(tokens)))
            axes[row, col].set_xticklabels(tokens, 
                                           rotation=90, 
                                           fontsize=8,
                                           color=COLOR_SCHEME['text'])
            axes[row, col].set_yticks(range(len(tokens)))
            axes[row, col].set_yticklabels(tokens, 
                                           fontsize=8,
                                           color=COLOR_SCHEME['text'])
        
        plt.colorbar(im, ax=axes[row, col], label='Attention Weight')
        axes[row, col].grid(False)
        axes[row, col].set_facecolor(COLOR_SCHEME['background'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
    plt.close()
    print(f"Attention heatmaps saved: {save_path}")


def create_dashboard(experiment_logs: Dict[str, Dict], save_path: str):
    """
    Create comprehensive training dashboard.
    
    Combines loss, gradient norms, and samples into one publication-ready figure.
    """
    setup_seaborn_style()
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLOR_SCHEME['background'])
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Loss comparison
    ax1 = fig.add_subplot(gs[0, :])
    for exp_name, logs in experiment_logs.items():
        train_loss = logs.get('train_loss', [])
        val_loss = logs.get('val_loss', [])
        ax1.plot(range(len(train_loss)), train_loss,
                 label=f'{exp_name} (train)', linewidth=2)
        ax1.plot(range(len(val_loss)), val_loss,
                 label=f'{exp_name} (val)', linewidth=2, linestyle='--')
    ax1.set_title('Training Loss Comparison', color=COLOR_SCHEME['text'], fontsize=14, weight='bold')
    ax1.set_xlabel('Step', color=COLOR_SCHEME['text'])
    ax1.set_ylabel('Loss', color=COLOR_SCHEME['text'])
    ax1.legend(facecolor=COLOR_SCHEME['grid'], edgecolor=COLOR_SCHEME['text'])
    ax1.grid(True, color=COLOR_SCHEME['grid'], alpha=0.3)
    ax1.set_facecolor(COLOR_SCHEME['background'])
    ax1.tick_params(colors=COLOR_SCHEME['text'])
    
    # Gradient norms
    ax2 = fig.add_subplot(gs[1, 0])
    for exp_name, logs in experiment_logs.items():
        grad_norm = logs.get('gradient_norm', [])
        ax2.plot(range(len(grad_norm)), grad_norm,
                 label=f'{exp_name}', linewidth=2)
    ax2.set_title('Gradient Norm (Training Stability)', color=COLOR_SCHEME['text'], fontsize=14, weight='bold')
    ax2.set_xlabel('Step', color=COLOR_SCHEME['text'])
    ax2.set_ylabel('Gradient Norm', color=COLOR_SCHEME['text'])
    ax2.legend(facecolor=COLOR_SCHEME['grid'], edgecolor=COLOR_SCHEME['text'])
    ax2.grid(True, color=COLOR_SCHEME['grid'], alpha=0.3)
    ax2.set_facecolor(COLOR_SCHEME['background'])
    ax2.tick_params(colors=COLOR_SCHEME['text'])
    ax2.set_yscale('log')
    
    # Sample generations text box
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_facecolor(COLOR_SCHEME['background'])
    ax3.set_title('Sample Generations', color=COLOR_SCHEME['text'], fontsize=14, weight='bold')
    
    sample_text = "Sample outputs will be added here after training."
    ax3.text(0.05, 0.5, sample_text, transform=ax3.transAxes,
               fontsize=12, color=COLOR_SCHEME['text'], verticalalignment='top')
    
    fig.suptitle('Transformer Training Dashboard', fontsize=18, 
                  color=COLOR_SCHEME['text'], weight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
    plt.close()
    print(f"Dashboard saved: {save_path}")


def create_all_visualizations():
    """
    Generate all visualizations from experiment logs.
    """
    output_dir = 'outputs/visualizations/'
    
    # Load experiment logs
    experiment_logs = {}
    
    # Find all experiment history files
    log_files = [f for f in os.listdir('outputs/logs/') if f.endswith('_history.json')]
    
    for log_file in log_files:
        with open(f'outputs/logs/{log_file}', 'r') as f:
            history = json.load(f)
        
        exp_name = log_file.replace('_history.json', '')
        if exp_name not in experiment_logs:
            experiment_logs[exp_name] = []
        experiment_logs[exp_name].append(history)
    
    if not experiment_logs:
        print("No experiment logs found. Run experiments first.")
        return
    
    print(f"\nFound {len(experiment_logs)} experiments")
    
    # Group by experiment type
    pos_enc_logs = {k: v for k, v in experiment_logs.items() 
                   if 'pos' in k or 'encoding' in k}
    norm_logs = {k: v for k, v in experiment_logs.items() 
                 if 'norm' in k}
    
    # Create visualizations
    if pos_enc_logs:
        print("Creating loss comparison...")
        plot_loss_comparison(pos_enc_logs, 
                         os.path.join(output_dir, 'pos_enc_loss_comparison.png'))
    
    if norm_logs:
        print("Creating gradient norm comparison...")
        plot_gradient_norms(norm_logs,
                           os.path.join(output_dir, 'gradient_norm_comparison.png'))
    
    # Dashboard
    print("Creating dashboard...")
    create_dashboard(experiment_logs,
                    os.path.join(output_dir, 'training_dashboard.png'))
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate experiment visualizations')
    parser.add_argument('--create-all', action='store_true',
                       help='Generate all visualizations')
    args = parser.parse_args()
    
    if args.create_all:
        create_all_visualizations()
    else:
        print("Use --create-all to generate all visualizations")
        parser.print_help()
