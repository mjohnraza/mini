import os
import time
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.dataset import prepare_datasets, get_batch_fast
from src.utils import set_seed, get_device

def train_ablation_run(
    name: str,
    model_kwargs: Dict[str, Any],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    context_length: int,
    steps: int = 50,
    batch_size: int = 32,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Train a single ablation configuration and record metrics."""
    set_seed(42)
    print(f"\n>>> Running Ablation: {name} (steps={steps}, batch={batch_size})", flush=True)

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        **model_kwargs
    ).to(device)

    param_count = model.get_num_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    start_time = time.time()
    for step in range(steps):
        x, y = get_batch_fast(train_data, context_length, batch_size, device=str(device))
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    # Evaluate validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(15):
            x, y = get_batch_fast(val_data, context_length, batch_size, device=str(device))
            _, loss, _ = model(x, y)
            val_losses.append(loss.item())
    final_val_loss = sum(val_losses) / len(val_losses)

    print(f"[{name}] Params: {param_count:,} | Val Loss: {final_val_loss:.4f} | Time: {train_time:.1f}s", flush=True)

    return {
        "name": name,
        "params": param_count,
        "val_loss": final_val_loss,
        "train_time": train_time
    }

def run_all_ablations(steps_per_run: int = 50):
    device = get_device("auto")
    os.makedirs("outputs/figures", exist_ok=True)
    raw_path = "data/input.txt"

    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer.train_from_text(text)
    train_ds, val_ds, _ = prepare_datasets(raw_path, context_length=256, tokenizer=tokenizer)

    results = []

    # 1. Depth Ablation: 2 vs 4 vs 6 layers (Heads=4, d_model=128, context=128)
    print("\n================ 1. Depth Ablation ================")
    for depth in [2, 4, 6]:
        res = train_ablation_run(
            name=f"Depth_{depth}L",
            model_kwargs={"d_model": 128, "n_heads": 4, "n_layers": depth, "ffn_dim": 512},
            train_data=train_ds.data,
            val_data=val_ds.data,
            vocab_size=tokenizer.vocab_size,
            context_length=128,
            steps=steps_per_run,
            device=str(device)
        )
        results.append(res)

    # 2. Heads Ablation: 2 vs 4 vs 8 heads (Layers=4, d_model=128, context=128)
    print("\n================ 2. Attention Heads Ablation ================")
    for heads in [2, 4, 8]:
        res = train_ablation_run(
            name=f"Heads_{heads}H",
            model_kwargs={"d_model": 128, "n_heads": heads, "n_layers": 4, "ffn_dim": 512},
            train_data=train_ds.data,
            val_data=val_ds.data,
            vocab_size=tokenizer.vocab_size,
            context_length=128,
            steps=steps_per_run,
            device=str(device)
        )
        results.append(res)

    # 3. Context Length Ablation: 64 vs 128 vs 256 (Layers=4, Heads=4, d_model=128)
    print("\n================ 3. Context Length Ablation ================")
    for ctx in [64, 128, 256]:
        res = train_ablation_run(
            name=f"Context_{ctx}T",
            model_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 4, "ffn_dim": 512},
            train_data=train_ds.data,
            val_data=val_ds.data,
            vocab_size=tokenizer.vocab_size,
            context_length=ctx,
            steps=steps_per_run,
            device=str(device)
        )
        results.append(res)

    # 4. Positional Encoding Ablation: Sinusoidal vs Learned
    print("\n================ 4. Positional Method Ablation ================")
    for pos_type in ["sinusoidal", "learned"]:
        res = train_ablation_run(
            name=f"Pos_{pos_type.capitalize()}",
            model_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 4, "ffn_dim": 512, "pos_encoding_type": pos_type},
            train_data=train_ds.data,
            val_data=val_ds.data,
            vocab_size=tokenizer.vocab_size,
            context_length=128,
            steps=steps_per_run,
            device=str(device)
        )
        results.append(res)

    # Plot Figure 9
    plot_ablation_results(results, "outputs/figures/ablation_plot.png")

    # Print Table
    print("\n=== Final Ablation Results Table (Figure 9) ===")
    print(f"{'Experiment':<20} | {'Params':<12} | {'Val Loss':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} | {r['params']:<12,} | {r['val_loss']:<10.4f} | {r['train_time']:<10.1f}")

    return results

def plot_ablation_results(results: List[Dict[str, Any]], output_path: str):
    names = [r["name"] for r in results]
    val_losses = [r["val_loss"] for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, val_losses, color="#2b5c8f", width=0.6)
    plt.title("MiniGPT Ablation Study: Validation Loss Comparison", fontsize=14, pad=12)
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    # Annotate bar values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure 9] Ablation plot saved to: {output_path}")

if __name__ == "__main__":
    run_all_ablations(steps_per_run=50)
