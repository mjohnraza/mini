"""
ROI Calculation Engine for LLM Training and Serving.
Analyzes:
1. Training Compute & Cost (FLOPs, Hardware, Dollars) for Baseline MiniGPT vs 7B LLM.
2. KV Cache Footprint: MHA vs GQA vs MQA vs MLA across context lengths (128 to 128k).
3. Inference Throughput & Memory Bandwidth Constraints on Modern GPUs (A100, H100, RTX 4090).
4. Cost per Million Tokens Served ($/M tokens) across attention architectures.
5. Break-even Analysis: Custom Domain Model vs Commercial Cloud APIs (GPT-4o, Claude 3.5).
"""

import math
from typing import Dict, List, Any

# ---------------------------------------------------------
# 1. Training Compute & Cost Calculations
# ---------------------------------------------------------

def compute_training_cost(
    model_name: str,
    num_params: float,
    num_tokens: float,
    context_length: int,
    hardware_name: str,
    gpu_peak_tflops: float,
    mfu: float,
    gpu_cost_per_hour: float,
    num_gpus: int = 1
) -> Dict[str, Any]:
    """
    Computes total training FLOPs (6*N*D approximation + attention quadratic term),
    wall-clock GPU hours, and total financial cost.
    """
    # Standard Chinchilla / Kaplan approximation for non-embedding FLOPs: 6 * N * D
    dense_flops = 6.0 * num_params * num_tokens
    
    # Effective TFLOPS per GPU accounting for Model FLOPs Utilization (MFU)
    effective_tflops_per_gpu = gpu_peak_tflops * mfu
    effective_flops_cluster = effective_tflops_per_gpu * 1e12 * num_gpus

    # Total training time
    total_seconds = dense_flops / effective_flops_cluster
    total_gpu_hours = (total_seconds / 3600.0) * num_gpus
    total_cost_usd = total_gpu_hours * gpu_cost_per_hour

    return {
        "model_name": model_name,
        "num_params": num_params,
        "num_tokens": num_tokens,
        "context_length": context_length,
        "dense_flops": dense_flops,
        "dense_petaflops": dense_flops / 1e15,
        "hardware_name": hardware_name,
        "num_gpus": num_gpus,
        "gpu_hours": total_gpu_hours,
        "wall_clock_days": total_seconds / (3600.0 * 24.0),
        "total_cost_usd": total_cost_usd
    }

# ---------------------------------------------------------
# 2. KV Cache Footprint across Attention Variants
# ---------------------------------------------------------

def compute_kv_cache_size(
    layers: int,
    num_q_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    precision_bytes: int = 2,  # FP16 / BF16 = 2 bytes
    architecture: str = "MHA",
    num_kv_heads: int = None,
    mla_dc: int = 512,         # MLA compressed latent dimension
    mla_dr: int = 64           # MLA decoupled rotary key dimension
) -> float:
    """
    Computes total KV Cache memory in Gigabytes (GB).
    
    - MHA: 2 * L * H_q * d_k * seq_len * precision
    - GQA: 2 * L * H_kv * d_k * seq_len * precision
    - MQA: 2 * L * 1 * d_k * seq_len * precision
    - MLA: L * (d_c + d_r) * seq_len * precision (DeepSeek-style)
    """
    arch = architecture.upper()
    if arch == "MHA":
        h_kv = num_q_heads
        elements_per_token = 2 * layers * h_kv * head_dim
    elif arch == "GQA":
        h_kv = num_kv_heads if num_kv_heads is not None else max(1, num_q_heads // 4)
        elements_per_token = 2 * layers * h_kv * head_dim
    elif arch == "MQA":
        h_kv = 1
        elements_per_token = 2 * layers * h_kv * head_dim
    elif arch == "MLA":
        # DeepSeek Multi-Head Latent Attention compresses KV into latent vector dc + decoupled key dr
        elements_per_token = layers * (mla_dc + mla_dr)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    total_bytes = batch_size * seq_len * elements_per_token * precision_bytes
    return total_bytes / (1024 ** 3)  # Convert to GB

# ---------------------------------------------------------
# 3. Inference Serving Economics & Capacity
# ---------------------------------------------------------

def compute_serving_capacity(
    model_params: float,
    gpu_vram_gb: float,
    gpu_bandwidth_gb_s: float,
    gpu_hourly_rate: float,
    layers: int = 32,
    num_q_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 4096,
    precision_bytes: int = 2
) -> List[Dict[str, Any]]:
    """
    Computes maximum batch capacity (concurrent users), decoding tokens/sec,
    and serving cost per 1M generated tokens for MHA, GQA, MQA, and MLA.
    """
    # Weights footprint in FP16/BF16
    weights_vram = (model_params * precision_bytes) / (1024 ** 3)
    # Reserve memory for CUDA runtime, activation buffers, OS overhead (~10%)
    runtime_reserve = max(4.0, gpu_vram_gb * 0.10)
    available_kv_vram = gpu_vram_gb - weights_vram - runtime_reserve

    architectures = [
        ("MHA", 32),
        ("GQA (8 groups)", 8),
        ("MQA (1 head)", 1),
        ("MLA (Latent)", None)
    ]

    results = []
    for name, kv_heads in architectures:
        arch_type = "MLA" if "MLA" in name else ("MQA" if "MQA" in name else ("GQA" if "GQA" in name else "MHA"))
        # KV cache per single user
        single_kv_gb = compute_kv_cache_size(
            layers=layers,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            batch_size=1,
            precision_bytes=precision_bytes,
            architecture=arch_type,
            num_kv_heads=kv_heads
        )

        # Max batch size without OOM
        max_batch = max(1, int(available_kv_vram / single_kv_gb))

        # Memory bandwidth bound decoding speed:
        # Each decoding step loads: model_weights + total_active_KV_cache
        total_kv_loaded_gb = max_batch * single_kv_gb
        total_bytes_per_step_gb = weights_vram + total_kv_loaded_gb

        # Tokens per second across the whole batch
        steps_per_sec = gpu_bandwidth_gb_s / total_bytes_per_step_gb
        total_throughput_tokens_sec = steps_per_sec * max_batch

        # Cost per 1 Million generated tokens
        tokens_per_hour = total_throughput_tokens_sec * 3600.0
        cost_per_1m_tokens = (gpu_hourly_rate / tokens_per_hour) * 1e6

        # Savings relative to MHA
        results.append({
            "architecture": name,
            "kv_cache_per_user_gb": single_kv_gb,
            "max_batch_size": max_batch,
            "tokens_per_sec": total_throughput_tokens_sec,
            "tokens_per_hour_m": tokens_per_hour / 1e6,
            "cost_per_1m_tokens_usd": cost_per_1m_tokens
        })

    # Add relative savings relative to MHA
    mha_cost = results[0]["cost_per_1m_tokens_usd"]
    for r in results:
        r["cost_reduction_percent"] = ((mha_cost - r["cost_per_1m_tokens_usd"]) / mha_cost) * 100.0

    return results

# ---------------------------------------------------------
# 4. Break-Even Analysis: Custom Model vs Commercial APIs
# ---------------------------------------------------------

def compute_breakeven_analysis(
    training_cost_usd: float,
    inhouse_serving_cost_per_1m: float,
    api_blended_cost_per_1m: float = 5.00  # e.g., GPT-4o blended rate ($2.50 in, $10.00 out)
) -> Dict[str, Any]:
    """
    Calculates break-even token volume and monthly query thresholds.
    """
    savings_per_1m = api_blended_cost_per_1m - inhouse_serving_cost_per_1m
    if savings_per_1m <= 0:
        return {"breakeven_tokens_m": float("inf"), "status": "Not profitable compared to API"}

    breakeven_tokens_m = training_cost_usd / savings_per_1m
    
    # Assume typical enterprise prompt + response is ~1,000 tokens
    queries_breakeven = (breakeven_tokens_m * 1e6) / 1000.0

    return {
        "training_investment_usd": training_cost_usd,
        "api_cost_per_1m_usd": api_blended_cost_per_1m,
        "inhouse_cost_per_1m_usd": inhouse_serving_cost_per_1m,
        "savings_per_1m_usd": savings_per_1m,
        "breakeven_tokens_millions": breakeven_tokens_m,
        "breakeven_tokens_billions": breakeven_tokens_m / 1000.0,
        "breakeven_enterprise_queries": queries_breakeven
    }

# ---------------------------------------------------------
# Execution & Markdown Summary Generator
# ---------------------------------------------------------

def run_all_roi_calculations() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING COMPREHENSIVE LLM TRAINING & SERVING ROI CALCULATIONS")
    print("=" * 70)

    # 1. Baseline MiniGPT (This Assignment)
    # Parameters: 809,984 (~0.81M), 1000 steps * 64 batch * 128 context = 8.192M tokens trained
    minigpt_train = compute_training_cost(
        model_name="MiniGPT (Assignment Baseline)",
        num_params=809984,
        num_tokens=8192000,
        context_length=128,
        hardware_name="Intel/AMD Core CPU",
        gpu_peak_tflops=0.5,
        mfu=0.20,
        gpu_cost_per_hour=0.05,  # Electricity cost rate ($0.05/hr)
        num_gpus=1
    )

    # 2. Scaled 7B Foundation Model (e.g., Llama-style domain pretraining)
    llama_7b_train = compute_training_cost(
        model_name="7B Domain Pretrained LLM",
        num_params=7e9,
        num_tokens=2e12,  # 2 Trillion tokens
        context_length=4096,
        hardware_name="NVIDIA H100 SXM (80GB)",
        gpu_peak_tflops=989.0,  # Dense BF16
        mfu=0.40,               # 40% MFU
        gpu_cost_per_hour=3.00, # Cloud rental per H100
        num_gpus=1024
    )

    # 3. Domain Adaptation / Fine-Tuning (LoRA / SFT on 100M tokens)
    sft_7b_train = compute_training_cost(
        model_name="7B Domain SFT / Fine-Tuning",
        num_params=7e9,
        num_tokens=1e8,   # 100M domain tokens
        context_length=4096,
        hardware_name="NVIDIA A100 (80GB)",
        gpu_peak_tflops=312.0,  # Dense BF16
        mfu=0.35,
        gpu_cost_per_hour=2.50,
        num_gpus=8
    )

    # 4. KV Cache Across Sequence Lengths
    seq_lengths = [128, 512, 2048, 8192, 32768, 131072]
    kv_matrix = []
    for seq in seq_lengths:
        mha = compute_kv_cache_size(32, 32, 128, seq, batch_size=1, architecture="MHA")
        gqa = compute_kv_cache_size(32, 32, 128, seq, batch_size=1, architecture="GQA", num_kv_heads=8)
        mqa = compute_kv_cache_size(32, 32, 128, seq, batch_size=1, architecture="MQA")
        mla = compute_kv_cache_size(32, 32, 128, seq, batch_size=1, architecture="MLA")
        kv_matrix.append({
            "seq_len": seq,
            "mha_gb": mha,
            "gqa_gb": gqa,
            "mqa_gb": mqa,
            "mla_gb": mla
        })

    # 5. Serving Capacity on NVIDIA A100 80GB (Bandwidth = 2039 GB/s)
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

    # 6. Break-Even Analysis against Commercial APIs
    gqa_serving_cost = next(s["cost_per_1m_tokens_usd"] for s in serving_metrics if "GQA" in s["architecture"])
    breakeven_sft = compute_breakeven_analysis(
        training_cost_usd=sft_7b_train["total_cost_usd"] + 5000.0, # include data prep labor
        inhouse_serving_cost_per_1m=gqa_serving_cost,
        api_blended_cost_per_1m=5.00
    )

    print("\n--- 1. Training Compute & Economics ---")
    print(f"MiniGPT (Assignment Baseline): FLOPs={minigpt_train['dense_flops']:.2e} | Time={minigpt_train['gpu_hours']*3600:.1f}s | Cost=${minigpt_train['total_cost_usd']:.4f}")
    print(f"7B Foundation Pretraining:    FLOPs={llama_7b_train['dense_flops']:.2e} | Time={llama_7b_train['wall_clock_days']:.1f} days (1024 H100s) | Cost=${llama_7b_train['total_cost_usd']:,.2f}")
    print(f"7B Domain SFT (100M tokens):   FLOPs={sft_7b_train['dense_flops']:.2e} | Time={sft_7b_train['gpu_hours']:.1f} GPU-hrs | Cost=${sft_7b_train['total_cost_usd']:,.2f}")

    print("\n--- 2. Single-User KV Cache Sizing (GB) for 7B Model ---")
    print(f"{'Context Length':<16} | {'MHA (32h)':<10} | {'GQA (8h)':<10} | {'MQA (1h)':<10} | {'MLA (Latent)':<10}")
    print("-" * 65)
    for row in kv_matrix:
        print(f"{row['seq_len']:<16} | {row['mha_gb']:<10.4f} | {row['gqa_gb']:<10.4f} | {row['mqa_gb']:<10.4f} | {row['mla_gb']:<10.4f}")

    print("\n--- 3. Serving Economics on A100 80GB (Context = 4k) ---")
    print(f"{'Architecture':<18} | {'KV/User':<10} | {'Max Batch':<10} | {'TPS':<10} | {'$/1M Tokens':<12} | {'Savings %':<10}")
    print("-" * 80)
    for s in serving_metrics:
        print(f"{s['architecture']:<18} | {s['kv_cache_per_user_gb']:<10.3f} | {s['max_batch_size']:<10} | {s['tokens_per_sec']:<10.1f} | ${s['cost_per_1m_tokens_usd']:<11.2f} | {s['cost_reduction_percent']:<9.1f}%")

    print("\n--- 4. Break-Even Analysis (Domain Model vs GPT-4o API) ---")
    print(f"Initial Investment:        ${breakeven_sft['training_investment_usd']:,.2f}")
    print(f"API Cost / 1M Tokens:      ${breakeven_sft['api_cost_per_1m_usd']:.2f}")
    print(f"In-House GQA Cost / 1M:    ${breakeven_sft['inhouse_cost_per_1m_usd']:.2f}")
    print(f"Savings / 1M Tokens:       ${breakeven_sft['savings_per_1m_usd']:.2f}")
    print(f"Break-Even Token Volume:   {breakeven_sft['breakeven_tokens_billions']:.3f} Billion tokens ({breakeven_sft['breakeven_enterprise_queries']:,.0f} queries)")

    return {
        "minigpt_train": minigpt_train,
        "llama_7b_train": llama_7b_train,
        "sft_7b_train": sft_7b_train,
        "kv_matrix": kv_matrix,
        "serving_metrics": serving_metrics,
        "breakeven_sft": breakeven_sft
    }

if __name__ == "__main__":
    run_all_roi_calculations()
