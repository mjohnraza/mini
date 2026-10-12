# 6-to-7 Minute Complete Demo Presentation Script: MiniGPT from Scratch & ROI Analysis

Use this step-by-step presentation script while recording your screen. It includes **exact commands**, **what to display**, **exact words to speak**, and **sample prompts**.

---

## Quick Navigation / Timing Breakdown

| Timestamp | Phase | Topic / Action | Terminal Command / File to Open |
| :---: | :---: | :--- | :--- |
| **0:00 – 0:45** | **Phase 1** | Architecture & Repo Overview | Open VS Code / File tree (`src/`, `configs/`) |
| **0:45 – 1:30** | **Phase 2** | Code Walkthrough: Causal Attention | Open `src/attention.py` & `src/gpt.py` |
| **1:30 – 2:30** | **Phase 3** | Automated Test Suite (8/8 Pass) | `python -m pytest -v` |
| **2:30 – 3:30** | **Phase 4** | Model Evaluation & Loss Curves | `python -m experiments.evaluate`<br>Open `outputs/figures/loss_curve.png` |
| **3:30 – 4:45** | **Phase 5** | Live CLI Text Generation | `python -m src.generate ...` (2 Prompts) |
| **4:45 – 5:30** | **Phase 6** | REST API Microservice (FastAPI) | `python -m uvicorn src.api:app --reload`<br>Open Swagger UI in browser |
| **5:30 – 6:45** | **Phase 7** | Attention Map, Ablations & ROI | Open `outputs/figures/attention_map.png`<br>Open `roi/figures/breakeven_curve.png` |
| **6:45 – 7:15** | **Phase 8** | Wrap-Up & Conclusion | Final slide / GitHub Repo |

---

## Detailed Step-by-Step Script

### Phase 1: Introduction & Architecture (0:00 – 0:45)
**Screen to Show:** VS Code workspace or file tree showing `src/`, `tests/`, `experiments/`, `report/`, `roi/`.

> **What to Say:**
> *"Hello everyone. Today I'm demonstrating our MiniGPT project: a causal Transformer language model implemented completely from scratch using PyTorch tensor primitives—without loading any pretrained checkpoints or high-level Hugging Face abstractions.*
> 
> *Our project is fully structured with modular components: `src/tokenizer.py` for character tokenization, `src/attention.py` for scaled dot-product and multi-head attention, `src/transformer_block.py` implementing Pre-LayerNorm residual blocks, and `src/gpt.py` assembling the complete autoregressive decoder. We trained on the Tiny Shakespeare dataset (~1.1 MB) for 1,000 steps."*

---

### Phase 2: Causal Masking & Pre-LN Code Walkthrough (0:45 – 1:30)
**Screen to Show:** Open [`src/attention.py`](file:///src/attention.py#L30-L65) and [`src/transformer_block.py`](file:///src/transformer_block.py#L40-L65).

> **What to Say:**
> *"Let's examine the mathematical core in `src/attention.py`. In `scaled_dot_product_attention`, we compute $Q K^T / \sqrt{d_k}$. For autoregressive language modeling, we apply a lower-triangular causal mask where future entries $j > i$ are set to $-\infty$, ensuring that softmax assigns them exactly zero attention weight. This strictly enforces the autoregressive probability chain rule: predicting token $t+1$ never leaks ground-truth tokens from the future.*
> 
> *In `src/transformer_block.py`, we implement Pre-LayerNorm architecture, where normalization occurs before the attention and feed-forward sublayers. This creates an uninterrupted identity gradient highway across the residual connections, ensuring stable training from scratch without complex learning rate warmups."*

---

### Phase 3: Automated Test Suite (1:30 – 2:30)
**Command to Run in Terminal:**
```powershell
python -m pytest -v
```

> **What to Say:**
> *"Now let's run our test suite using pytest. All 8 tests pass immediately.*
> 
> *These tests verify:*
> 1. *Our manual hand calculations for a 3-token sequence with $d_k=2$ matching PyTorch output to four decimal places.*
> 2. *Causal masking strictness and zero backpropagation gradient leakage from future tokens.*
> 3. *End-to-end tensor shape tracing across all model layers.*
> 4. *An overfit convergence test, driving loss on a tiny batch from 3.40 down to 0.11, proving the architecture has the capacity to memorize before generalizing."*

---

### Phase 4: Model Evaluation & Loss Curves (2:30 – 3:30)
**Command to Run in Terminal:**
```powershell
python -m experiments.evaluate
```
**File to Open:** `outputs/figures/loss_curve.png`

> **What to Say:**
> *"Next, we run our model evaluation script. MiniGPT contains exactly 809,984 parameters (~0.81M), utilizing 4 layers, 4 attention heads, an embedding dimension of 128, and a context length of 128 tokens.*
> 
> *Looking at our loss curve in Figure 1, training began at uniform entropy $(\ln(65) \approx 4.17)$ and declined smoothly down to a final training loss of 1.68 and a validation loss of 1.83, achieving a validation perplexity of 6.31 without overfitting divergence."*

---

### Phase 5: Live Autoregressive CLI Text Generation (3:30 – 4:45)
**Command to Run (Prompt 1 - Shakespearean Dialogue):**
```powershell
python -m src.generate --prompt "ROMEO:" --max-new-tokens 70 --temperature 0.8
```

> **What to Say:**
> *"Now let's test autoregressive text generation using our command-line interface matching Section 12.1. Notice how the model correctly maintains character dialogue formatting, line breaks, and Elizabethan cadence."*

**Command to Run (Prompt 2 - General/AI Prompt):**
```powershell
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8
```

> **What to Say:**
> *"Now we prompt it with 'The future of artificial intelligence'. Because our model was trained exclusively from scratch on Shakespeare, it interprets this novel prefix through the lens of its training data—producing dramatic dialogue and verse structure. This vividly demonstrates that language models internalize the exact statistical distribution of their training corpus."*

---

### Phase 6: Live REST API Microservice (4:45 – 5:30)
**Command to Run in Terminal:**
```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
**Action:** Open browser at `http://127.0.0.1:8000/docs` (Interactive Swagger UI).
- Test `/health` $\to$ Returns `{"status": "healthy", "model_loaded": true}`
- Test `/model-info` $\to$ Returns `{"parameters": 809984, "context_length": 128, ...}`
- Test `/generate` $\to$ Send `{"prompt": "HAMLET:", "max_new_tokens": 50, "temperature": 0.8}`

> **What to Say:**
> *"To demonstrate production readiness, we packaged MiniGPT inside a high-performance FastAPI microservice. Through Swagger UI, we can ping `/health` to verify model weights are in memory, inspect `/model-info`, and execute `/generate` to receive low-latency JSON responses. This container-ready service can be deployed directly to cloud environments like Railway, Render, or Hugging Face Spaces."*

*(Press `Ctrl + C` in terminal to stop server).*

---

### Phase 7: Attention Visualization, Ablations & ROI (5:30 – 6:45)
**Files to Show in Editor/Image Viewer:**
1. `outputs/figures/attention_map.png`
2. `outputs/figures/ablation_plot.png`
3. `roi/figures/breakeven_curve.png`

> **What to Say:**
> *"Here in Figure 2, we inspect our trained attention map from Layer 1, Head 1. We observe strong self-attention along the causal diagonal as well as backward attention focusing on prefix word-start tokens.*
> 
> *Our ablation experiments in Figure 3 showed that 4 attention heads ($d_k=32$) produced optimal validation loss over 2 and 8 heads, and doubling context window from 128 to 256 tokens doubled training time due to $O(T^2)$ attention scaling.*
> 
> *Finally, our Return on Investment analysis highlights the economics of scaling. Our baseline model cost only \$0.0055 in electricity. However, scaling to a 7B foundation model pretraining run costs over \$176,000, while fine-tuning (LoRA/SFT) costs only \$26.71—a 6,600x cost advantage! Furthermore, autoregressive inference is bound by the KV cache: adopting Grouped-Query Attention (GQA) reduces KV cache by 75%, allowing an A100 GPU to serve 117 concurrent users versus just 29 under standard Multi-Head Attention.*
> 
> *Our break-even curve proves that an in-house 7B GQA model breaks even against commercial APIs like GPT-4o at approximately 1.05 billion tokens (~1 million enterprise queries), saving \$4,790 per billion tokens thereafter."*

---

### Phase 8: Conclusion & GitHub Repository (6:45 – 7:15)
**Screen to Show:** Browser showing [https://github.com/mjohnraza/mini](https://github.com/mjohnraza/mini) or repository README.

> **What to Say:**
> *"In conclusion, this project proved that a transformer language model can be mathematically derived, implemented, tested, trained, and served from scratch. Understanding these low-level tensor mechanics is essential for diagnosing real-world memory bandwidth bottlenecks and maximizing AI return on investment.*
> 
> *All code, test suites, evaluation scripts, and reports are published on GitHub at `github.com/mjohnraza/mini`. Thank you!"*
