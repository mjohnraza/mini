# 6-to-7 Minute Complete Demo Presentation Script: MiniGPT from Scratch (Roman Urdu)

Yeh script video recording ke liye tayyar ki gayi hai. Is mein exact timing, screen par kya dikhana hai, terminal commands, aur aapne word-to-word kya bolna hai sab shamil hai.

---

## ⏱️ Timing aur Flow Summary (Total: 6–7 Minutes)

| Waqt (Time) | Step / Phase | Screen par kya dikhana hai | Terminal Command / Action |
| :---: | :---: | :--- | :--- |
| **0:00 – 0:45** | **Phase 1: Intro** | VS Code Explorer (`src/`, `tests/`, etc.) | File structure overview |
| **0:45 – 1:30** | **Phase 2: Code Core** | `src/attention.py` & `src/transformer_block.py` | Causal Mask aur Pre-LN code |
| **1:30 – 2:30** | **Phase 3: Tests** | Terminal window | `python -m pytest -v` (8/8 pass) |
| **2:30 – 3:30** | **Phase 4: Metrics** | Terminal aur Loss curve image | `python -m experiments.evaluate`<br>`outputs/figures/loss_curve.png` |
| **3:30 – 4:45** | **Phase 5: CLI Live Run** | Terminal | 2 Prompts par generation |
| **4:45 – 5:30** | **Phase 6: REST API** | Browser (Swagger UI) | `uvicorn` server & `/docs` testing |
| **5:30 – 6:45** | **Phase 7: ROI & Attention** | Attention Map aur Breakeven Plot | `attention_map.png` & `breakeven_curve.png` |
| **6:45 – 7:15** | **Phase 8: Conclusion** | GitHub Repo page | [github.com/mjohnraza/mini](https://github.com/mjohnraza/mini) |

---

## 🎙️ Word-for-Word Script (Bolne ke liye)

### 0:00 – 0:45 | Phase 1: Intro & Repository Overview
**Screen:** VS Code file tree dikhayein (`src/`, `configs/`, `tests/`, `roi/`).

> *"Assalam-o-Alaikum / Hello everyone. Aaj main apne MiniGPT project ka live demo pesh kar raha hoon. Is project mein humne ek complete Causal Transformer Language Model bilkul scratch se, yani pure PyTorch tensor primitives se design aur train kiya hai—bina kisi external pretrained weights ya Hugging Face ki black-box abstraction ke.*
> 
> *Aap repository structure dekh sakte hain: `src/tokenizer.py` mein hamara 65-character level deterministic tokenizer hai, `src/attention.py` mein Scaled Dot-Product aur Multi-Head Attention hai, `src/transformer_block.py` mein Pre-LayerNorm residual blocks hain, aur `src/gpt.py` mein complete autoregressive causal GPT model assemble kiya gaya hai. Humne isko Tiny Shakespeare dataset par 1,000 steps ke liye train kiya hai."*

---

### 0:45 – 1:30 | Phase 2: Causal Masking aur Pre-LN Code Walkthrough
**Screen:** Open karein `src/attention.py` aur `src/transformer_block.py`.

> *"Ab aate hain core architecture par. `src/attention.py` ke andar humne $Q K^T / \sqrt{d_k}$ implement kiya hai. Causal text generation ke liye humne lower-triangular causal mask lagaya hai, jahan future tokens ko $-\infty$ set kiya jata hai taake Softmax unhe 0 probability de. Isse autoregressive chain rule enforce hota hai aur model training ke waqt aane wale tokens ko cheat nahi kar sakta.*
> 
> *Doosri ahem baat `src/transformer_block.py` mein Pre-LayerNorm ki placement hai. Har sublayer se pehle LayerNorm lagane se gradient ko residual stream mein clear rasta milta hai, jiski wajah se model bina kisi learning rate warmup ke shuru se hi stably train hota hai."*

---

### 1:30 – 2:30 | Phase 3: Automated Test Suite (8/8 Tests Pass)
**Terminal Command:**
```powershell
python -m pytest -v
```

> *(Command run karein aur green test results dikhayein)*  
> *"Ab hum automated unit test suite execute karte hain. Jaisa ke aap dekh sakte hain, tamam 8 ke 8 tests pass ho chuke hain.*
> 
> *Yeh tests mathematically verify karte hain:*
> 1. *Hamari manual hand calculations (3 tokens, dimension 2) jo PyTorch tensors ke saath exact 4 decimal places tak match karti hain.*
> 2. *Causal mask strictness, yani future tokens se 0% gradient leak hota hai.*
> 3. *End-to-end tensor shape tracing tamam layers ke through.*
> 4. *Ek overfit test, jisne tiny batch par loss ko 3.40 se 0.11 tak drop kiya—jo prove karta hai ke model memorize karne aur sikhne ki poori capacity rakhta hai."*

---

### 2:30 – 3:30 | Phase 4: Model Parameters & Loss Curves
**Terminal Command:**
```powershell
python -m experiments.evaluate
```
**File Open Karein:** `outputs/figures/loss_curve.png`

> *"Ab hum model evaluation run karte hain. Hamare MiniGPT mein exactly 809,984 parameters hain (~0.81 Million). Isme 4 layers, 4 attention heads, 128 embedding dimension, aur 128 tokens ka context window hai.*
> 
> *Yeh dekhiye Figure 1 mein hamara Training vs Validation loss curve. Training initial uniform entropy $(\ln(65) \approx 4.17)$ se smoothly drop hui aur final training loss 1.68 jabke validation loss 1.83 par settle hui, jo 6.31 validation perplexity deti hai bina kisi overfitting divergence ke."*

---

### 3:30 – 4:45 | Phase 5: Live Autoregressive CLI Text Generation
**Terminal Command 1 (Shakespeare Prompt):**
```powershell
python -m src.generate --prompt "ROMEO:" --max-new-tokens 70 --temperature 0.8
```

> *(Output generate hone par)*  
> *"Ab hum Section 12.1 ke mutabiq live inference run karte hain. Pehla prompt humne diya `'ROMEO:'`. Dekhiye model ne bilkul authentic Shakespearean style, dialogue turn labels, aur verse format maintain kiya hai."*

**Terminal Command 2 (Modern / AI Prompt):**
```powershell
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8
```

> *(Output generate hone par)*  
> *"Ab doosra prompt: `'The future of artificial intelligence'`. Choonke hamara model sirf Shakespeare par train hua hai, isne is modern prompt ko bhi apne statistical training distribution ke mutabiq dramatic character dialogue aur theatrical style mein convert kar diya. Yeh saaf demonstrate karta hai ke language models apne training corpus ki distribution ko follow karte hain."*

---

### 4:45 – 5:30 | Phase 6: Production REST API Microservice (FastAPI)
**Terminal Command:**
```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
**Action:** Browser kholein: `http://127.0.0.1:8000/docs`
- `/health` execute karein $\to$ `{"status": "healthy", "model_loaded": true}`
- `/generate` par click karein $\to$ JSON prompt bhejein $\to$ Completion receive karein.

> *"Production deployment ke liye humne is model ko FastAPI microservice mein wrap kiya hai. Swagger UI par hum `/health` endpoint se status check kar sakte hain, `/model-info` se details dekh sakte hain, aur `/generate` POST request bhej kar real-time JSON text generation le sakte hain. Yeh container-ready API cloud par deploy ki ja sakti hai."*  
> *(Terminal mein `Ctrl + C` daba kar server band karein).*

---

### 5:30 – 6:45 | Phase 7: Attention Visualization, Ablation aur ROI Analysis
**Images Open Karein:**
1. `outputs/figures/attention_map.png`
2. `outputs/figures/ablation_plot.png`
3. `roi/figures/breakeven_curve.png`

> *"Figure 2 mein hamara trained attention map hai. Layer 1 Head 1 mein strong causal diagonal nazar aa rahi hai jahan model pichle word-start tokens par focus kar raha hai.*
> 
> *Hamare ablation experiments ne show kiya ke 4 heads ($d_k=32$) ne 2 aur 8 heads se behtar result diya. Aur context length ko 128 se 256 double karne par time double ho gaya kyunke attention complexity $O(T^2)$ hoti hai.*
> 
> *Aakhri aur sabse ahem hissa hamara Return on Investment (ROI) analysis hai: Hamare baseline model ki training cost sirf \$0.0055 (aadhay cent se bhi kam) thi. Lekin jab hum 7B parameter foundation model pretrain karte hain, to compute cost \$176,000 se zyada ho jati hai! Iske muqable mein LoRA ya Fine-Tuning sirf \$26.71 leti hai—jo 6,600x sasti hai.*
> 
> *Inference time par sabse bada bottleneck KV Cache hota hai. Agar hum Multi-Head Attention ki jagah Grouped-Query Attention (GQA) use karein, to KV cache 75% shrink ho jata hai, aur ek hi A100 GPU 29 users ke bajaye 117 concurrent users ko serve kar sakta hai (\$0.21 per million tokens).*
> 
> *Hamare break-even graph ke mutabiq, commercial APIs (jaise GPT-4o) ke muqable mein in-house 7B GQA model 1.05 billion tokens (yani kareeb 10 lakh queries) par break-even karta hai, aur uske baad har billion token par \$4,790 bachaata hai, saath mein 100% data privacy bhi deta hai."*

---

### 6:45 – 7:15 | Phase 8: Conclusion & Wrap-Up
**Screen:** Browser par apna GitHub repo dikhayein: [github.com/mjohnraza/mini](https://github.com/mjohnraza/mini)

> *"Conclusion yeh hai ke scratch se Transformer banana na sirf attention mechanics aur tensor shapes ko crystal clear karta hai, balki real-world GPU memory bottlenecks aur AI engineering ROI ki deep understanding deta hai.*
> 
> *Tamam code, unit tests, scripts, aur comprehensive reports hamare GitHub repository `github.com/mjohnraza/mini` par available hain. Bohat shukriya!"*
