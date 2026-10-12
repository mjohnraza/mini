import os
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def set_seed(seed: int = 42) -> None:
    """Ensure full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(requested: str = "auto") -> torch.device:
    """Select best available hardware device."""
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    return torch.device(device)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "tiny_gpt_best.pt"
) -> str:
    """Save training checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"[Checkpoint] Model checkpoint saved to: {filepath}")
    return filepath

def load_checkpoint(filepath: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load training checkpoint from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    print(f"[Checkpoint] Loaded checkpoint from: {filepath}")
    return checkpoint

def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    steps: list[int],
    output_path: str = "outputs/figures/loss_curve.png"
) -> None:
    """Plot and save training vs validation loss curves (Figure 6 in assignment)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(steps, val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("MiniGPT Training vs Validation Loss Curve", fontsize=14, pad=12)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure] Loss curve figure saved to: {output_path}")
