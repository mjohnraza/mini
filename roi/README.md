# LLM Return on Investment (ROI) & Attention Architecture Analysis

This directory contains the computational and financial ROI calculations for training and serving Large Language Models across modern attention variants.

---

## Directory Structure

```
roi/
├── README.md                  # Overview and instructions
├── roi_analysis.md            # Comprehensive technical & financial report
├── calculate_roi.py           # Core calculation engine (FLOPs, KV cache, TPS, Break-even)
├── plot_roi.py                # Visual figure generator
└── figures/
    ├── kv_cache_scaling.png             # Log-scale KV cache memory across context lengths
    ├── serving_throughput_and_cost.png  # Max batch size, TPS, and $/1M tokens across architectures
    └── breakeven_curve.png              # Custom model vs commercial API break-even curve
```

---

## How to Run

### 1. Execute ROI Calculations
```bash
python -m roi.calculate_roi
```
This prints the exact mathematical and economic outputs for:
- MiniGPT baseline training cost (\$0.0055).
- 7B pretraining (\$176,946.41) vs 7B domain SFT (\$26.71).
- Single-user KV cache sizing from 128 to 128k context lengths.
- Serving capacity and \$/1M tokens on NVIDIA A100.
- Commercial API break-even point (~1.05B tokens).

### 2. Generate ROI Visual Figures
```bash
python -m roi.plot_roi
```
Saves high-resolution figures into `roi/figures/`.
