# 5–7 Minute Live Demo Script: MiniGPT from Scratch

This guide is designed for the student screen recording and presentation required in **Section 17** of the assignment handout. Follow each step sequentially during your demo video.

---

### Timing Breakdown (Total: 6 Minutes)

| Time | Step | Action | Speaking Points |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:45** | **1. Repository Overview** | `tree -L 2` or show file explorer | "Here is our repository. The core architecture lives in `src/gpt.py`, supported by modular components in `src/attention.py`, `src/transformer_block.py`, and `src/positional_encoding.py`. No external pretrained checkpoints are loaded." |
| **0:45 – 1:30** | **2. Attention & Causal Mask** | Open [`src/attention.py`](file:///src/attention.py) | "In `SelfAttention`, we project $Q, K, V$, compute $(Q K^T) / \sqrt{d_k}$, and apply our lower-triangular causal mask where $M_{i,j} = -\infty$ for $j > i$. In `MultiHeadedAttention`, we split into 4 heads, compute attention in parallel, concatenate outputs, and linearly project back to $d_{\text{model}}$." |
| **1:30 – 2:15** | **3. Test Suite Execution** | `python -m pytest -v` | "We run our unit tests. All 8 pass, verifying hand calculations ($T=3, d_k=2$), causal mask strictness, zero backprop gradient leakage from future tokens, tensor-shape tracing, and rapid overfit on a tiny batch." |
| **2:15 – 3:00** | **4. Model Config & Parameters** | `python -m experiments.evaluate` | "Our model has 4 layers, 4 heads, $d_{\text{model}}=128$, and context length $128$. Total parameter count is exactly 809,984 (~0.81M). It achieves validation loss 1.84 and perplexity 6.31 on Tiny Shakespeare." |
| **3:00 – 3:45** | **5. Loss Curves** | Open `outputs/figures/loss_curve.png` | "Here is our training vs validation loss curve (Figure 6). Training began at uniform entropy $\ln(65) \approx 4.17$ and converged down to 1.68 train loss and 1.84 validation loss over 1,000 steps without overfitting divergence." |
| **3:45 – 4:30** | **6. Live CLI Text Generation** | `python -m src.generate --prompt "ROMEO:\nWhat light through" --temperature 0.8` | "We run autoregressive inference via our CLI matching Section 12.1. Notice how MiniGPT captures Shakespearean dramatic headers, iambic meter, dialogue turns, and vocabulary." |
| **4:30 – 5:15** | **7. Attention Heatmap Inspection** | Open `outputs/figures/attention_map.png` | "Inspecting attention weights from Layer 1, Head 1 (Figure 7), we observe high attention diagonal and localized attending backward to word start tokens and punctuation delimiters." |
| **5:15 – 6:00** | **8. Live API Deployment** | `python -m uvicorn src.api:app --host 0.0.0.0 --port 8000` & open `http://localhost:8000/docs` | "We deploy the model with FastAPI. Endpoints include `/health`, `/model-info`, and `/generate`. Sending a POST request generates text dynamically." |
| **6:00 – 6:30** | **9. Key Ablation & Limitation** | Show `outputs/figures/ablation_plot.png` | "Our most critical ablation revealed that 4 heads ($d_k=32$) outperformed 2 and 8 heads. One key limitation is character-level vocabulary: rare words can induce character-level hallucinations, which subword BPE tokenization would eliminate." |

---

### One-Line Terminal Commands to Run During Demo

```bash
# 1. Run full unit tests
python -m pytest -v

# 2. View model parameter count and evaluate
python -m experiments.evaluate

# 3. Run CLI generation (Section 12.1 format)
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80

# 4. Start live FastAPI REST server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
