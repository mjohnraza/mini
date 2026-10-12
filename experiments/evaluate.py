import os
import math
import torch
import argparse
from typing import Dict, Any

from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.dataset import prepare_datasets, get_batch_fast
from src.utils import load_checkpoint, get_device

def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/tiny_gpt_best.pt",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    data_path: str = "data/input.txt",
    eval_iters: int = 100
):
    device = get_device("auto")
    print("=== MiniGPT Evaluation Suite ===")
    print(f"Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    # Load tokenizer and checkpoint
    tokenizer = CharTokenizer.load(tokenizer_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    config = checkpoint["config"]
    model_cfg = config["model"]

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=0.0,
        pos_encoding_type=model_cfg.get("pos_encoding_type", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Parameter Breakdown (Figure 5)
    print("\n--- Model Architecture & Parameter Count Breakdown ---")
    total_params = model.get_num_params()
    tok_emb_params = sum(p.numel() for p in model.tok_embeddings.parameters())
    blocks_params = sum(p.numel() for p in model.blocks.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    norm_params = sum(p.numel() for p in model.final_norm.parameters())

    print(f"Layers (Depth):            {model_cfg['n_layers']}")
    print(f"Attention Heads:           {model_cfg['n_heads']}")
    print(f"Embedding Dimension:       {model_cfg['d_model']}")
    print(f"Context Length:            {model_cfg['context_length']}")
    print(f"Vocabulary Size:           {tokenizer.vocab_size}")
    print(f"Token Embedding Params:    {tok_emb_params:,}")
    print(f"Transformer Blocks Params: {blocks_params:,}")
    print(f"LM Head Params:            {lm_head_params:,}")
    print(f"Total Parameters:          {total_params:,} ({total_params/1e6:.2f}M)")

    # Dataset evaluation
    if os.path.exists(data_path):
        train_ds, val_ds, _ = prepare_datasets(
            text_path=data_path,
            context_length=model_cfg["context_length"],
            tokenizer=tokenizer
        )
        
        @torch.no_grad()
        def eval_split(tensor_data):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch_fast(tensor_data, model_cfg["context_length"], batch_size=64, device=str(device))
                _, loss, _ = model(x, y)
                losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            ppl = math.exp(mean_loss)
            return mean_loss, ppl

        print("\n--- Quantitative Evaluation ---")
        train_loss, train_ppl = eval_split(train_ds.data)
        val_loss, val_ppl = eval_split(val_ds.data)
        print(f"Train Cross-Entropy Loss: {train_loss:.4f} | Perplexity (PPL): {train_ppl:.2f}")
        print(f"Val Cross-Entropy Loss:   {val_loss:.4f} | Perplexity (PPL): {val_ppl:.2f}")

    # Generation evaluation (Figure 8)
    print("\n--- Qualitative Text Generation (Figure 8) ---")
    test_prompts = [
        "ROMEO:\n",
        "JULIET:\n",
        "HAMLET:\n",
        "First Citizen:\n",
        "The king hath spoken:"
    ]
    for prompt in test_prompts:
        tok_in = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        gen = model.generate(tok_in, max_new_tokens=80, temperature=0.8, top_k=40)
        output_text = tokenizer.decode(gen[0].tolist())
        print(f"Prompt: {repr(prompt)}")
        print(f"Completion:\n{output_text}\n{'-'*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tiny_gpt_best.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--data", type=str, default="data/input.txt")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.tokenizer, args.data)
