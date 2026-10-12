"""
Plotting module for LLM Training & Serving ROI Analysis.
Generates publication-quality charts in roi/figures/:
1. kv_cache_scaling.png: KV Cache footprint (GB) across context windows (128 to 128k).
2. serving_throughput_and_cost.png: Concurrent users, TPS, and $/1M tokens for MHA vs GQA vs MLA vs MQA.
3. breakeven_curve.png: Cumulative cost comparison and ROI break-even point.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from .calculate_roi import compute_kv_cache_size, compute_serving_capacity, compute_breakeven_analysis

def plot_all_roi_figures(output_dir: str = "roi/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Figure 1: KV Cache Footprint Across Sequence Lengths
    # -------------------------------------------------------------
    seq_lengths = [128, 512, 2048, 8192, 32768, 131072]
    seq_labels = ["128", "512", "2k", "8k", "32k", "128k"]

    mha_vals = [compute_kv_cache_size(32, 32, 128, s, 1, 2, "MHA") for s in seq_lengths]
    gqa_vals = [compute_kv_cache_size(32, 32, 128, s, 1, 2, "GQA", num_kv_heads=8) for s in seq_lengths]
    mla_vals = [compute_kv_cache_size(32, 32, 128, s, 1, 2, "MLA") for s in seq_lengths]
    mqa_vals = [compute_kv_cache_size(32, 32, 128, s, 1, 2, "MQA") for s in seq_lengths]

    plt.figure(figsize=(10, 6))
    plt.plot(seq_labels, mha_vals, marker="o", linewidth=2.5, color="#d9534f", label="MHA (32 heads) - Baseline")
    plt.plot(seq_labels, gqa_vals, marker="s", linewidth=2.5, color="#f0ad4e", label="GQA (8 groups) - 4x Savings")
    plt.plot(seq_labels, mla_vals, marker="^", linewidth=2.5, color="#0275d8", label="MLA (DeepSeek Latent) - 14.2x Savings")
    plt.plot(seq_labels, mqa_vals, marker="d", linewidth=2.5, color="#5cb85c", label="MQA (1 head) - 32x Savings")

    plt.yscale("log")
    plt.title("KV Cache Memory Footprint per User (7B Model, FP16)\nLog Scale Across Context Window Horizons", fontsize=13, pad=14)
    plt.xlabel("Context Sequence Length (Tokens)", fontsize=11)
    plt.ylabel("KV Cache Size per User (GB - Log Scale)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    # Highlight 80GB GPU boundary
    plt.axhline(y=80.0, color="gray", linestyle="--", alpha=0.7)
    plt.text(0.1, 85.0, "Single A100/H100 80GB VRAM Limit", color="gray", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "kv_cache_scaling.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[Plot 1] Saved KV cache scaling to: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Serving Economics & Throughput Comparison
    # -------------------------------------------------------------
    serving_metrics = compute_serving_capacity(
        model_params=7e9,
        gpu_vram_gb=80.0,
        gpu_bandwidth_gb_s=2039.0,
        gpu_hourly_rate=2.50,
        layers=32,
        num_q_heads=32,
        head_dim=128,
        seq_len=4096,
        precision_bytes=2
    )

    names = [s["architecture"] for s in serving_metrics]
    batches = [s["max_batch_size"] for s in serving_metrics]
    tps = [s["tokens_per_sec"] for s in serving_metrics]
    costs = [s["cost_per_1m_tokens_usd"] for s in serving_metrics]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Ax 1: Max Batch Capacity
    colors = ["#d9534f", "#f0ad4e", "#5cb85c", "#0275d8"]
    bars1 = ax1.bar(names, batches, color=colors, width=0.55)
    ax1.set_title("Max Concurrent Users / Batch\n(A100 80GB @ 4k Context)", fontsize=11, pad=10)
    ax1.set_ylabel("Concurrent Sequences", fontsize=10)
    ax1.tick_params(axis='x', rotation=25)
    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, y + 10, f"{int(y)}", ha="center", va="bottom", fontsize=9)

    # Ax 2: Throughput (TPS)
    bars2 = ax2.bar(names, tps, color=colors, width=0.55)
    ax2.set_title("Decoding Throughput (Tokens / Sec)\n(Memory-Bandwidth Bound)", fontsize=11, pad=10)
    ax2.set_ylabel("Total Tokens / Sec", fontsize=10)
    ax2.tick_params(axis='x', rotation=25)
    for bar in bars2:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, y + 400, f"{int(y):,}", ha="center", va="bottom", fontsize=9)

    # Ax 3: Serving Cost per 1M Tokens ($)
    bars3 = ax3.bar(names, costs, color=colors, width=0.55)
    ax3.set_title("Serving Cost per 1M Tokens ($)\n(Cloud Rental: $2.50 / hr)", fontsize=11, pad=10)
    ax3.set_ylabel("Cost in USD ($)", fontsize=10)
    ax3.tick_params(axis='x', rotation=25)
    for bar in bars3:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, y + 0.02, f"${y:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "serving_throughput_and_cost.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[Plot 2] Saved serving economics to: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Break-Even ROI Analysis Curve
    # -------------------------------------------------------------
    tokens_millions = np.linspace(0, 3000, 100) # 0 to 3 Billion tokens
    api_rate = 5.00        # $5.00 per 1M tokens (blended commercial API)
    inhouse_rate = 0.21    # $0.21 per 1M tokens (In-house 7B GQA)
    initial_setup = 5026.71 # Training compute + data prep

    cumulative_api = tokens_millions * api_rate
    cumulative_inhouse = initial_setup + (tokens_millions * inhouse_rate)

    plt.figure(figsize=(9, 5.5))
    plt.plot(tokens_millions, cumulative_api, label="Commercial Cloud API ($5.00 / 1M tokens)", color="#d9534f", linewidth=2.5)
    plt.plot(tokens_millions, cumulative_inhouse, label="In-House Domain Model (GQA + $5k SFT)", color="#5cb85c", linewidth=2.5)

    # Intersection point
    breakeven_m = initial_setup / (api_rate - inhouse_rate)
    breakeven_cost = breakeven_m * api_rate
    plt.scatter([breakeven_m], [breakeven_cost], color="#0275d8", s=100, zorder=5)
    plt.annotate(
        f"Break-Even Point:\n{breakeven_m:.1f}M Tokens (~1.05B)\nCost: ${breakeven_cost:,.0f}",
        xy=(breakeven_m, breakeven_cost),
        xytext=(breakeven_m + 250, breakeven_cost - 2500),
        arrowprops=dict(facecolor="#0275d8", shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#0275d8")
    )

    plt.title("LLM ROI Break-Even Analysis: In-House Domain Model vs Commercial Cloud API", fontsize=13, pad=12)
    plt.xlabel("Total Processed Tokens (Millions)", fontsize=11)
    plt.ylabel("Cumulative Operational Cost ($ USD)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig3_path = os.path.join(output_dir, "breakeven_curve.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[Plot 3] Saved break-even curve to: {fig3_path}")

if __name__ == "__main__":
    plot_all_roi_figures()
