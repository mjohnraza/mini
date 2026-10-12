import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Any

from .tokenizer import CharTokenizer
from .dataset import prepare_datasets, get_batch_fast
from .gpt import MiniGPT
from .utils import set_seed, get_device, load_config, save_checkpoint, plot_loss_curves

def train_model(config_path: str = "configs/tiny_gpt.yaml", override_steps: int = None):
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    gen_cfg = config["generation"]

    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"=== MiniGPT Training Pipeline ===")
    print(f"Hardware Device: {device}")

    # Ensure output directories exist
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Prepare data
    raw_file = data_cfg.get("raw_file", "data/input.txt")
    if not os.path.exists(raw_file):
        from data.prepare_data import download_tiny_shakespeare
        raw_file = download_tiny_shakespeare()

    train_ds, val_ds, tokenizer = prepare_datasets(
        text_path=raw_file,
        context_length=model_cfg["context_length"],
        train_split=data_cfg.get("train_split", 0.9)
    )

    # Save tokenizer for inference reload
    tokenizer.save("checkpoints/tokenizer.json")

    # 2. Instantiate Model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=model_cfg.get("dropout", 0.1),
        pos_encoding_type=model_cfg.get("pos_encoding_type", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False)
    ).to(device)

    total_params = model.get_num_params()
    print(f"Model Architecture: {model_cfg['n_layers']} layers, {model_cfg['n_heads']} heads, d_model={model_cfg['d_model']}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01)
    )

    max_steps = override_steps if override_steps is not None else train_cfg.get("max_steps", 2000)
    batch_size = train_cfg.get("batch_size", 64)
    eval_interval = train_cfg.get("eval_interval", 200)
    eval_iters = train_cfg.get("eval_iters", 50)
    context_length = model_cfg["context_length"]

    train_tensor = train_ds.data
    val_tensor = val_ds.data

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split, d_tensor in [("train", train_tensor), ("val", val_tensor)]:
            split_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch_fast(d_tensor, context_length, batch_size, device=str(device))
                _, loss, _ = model(x, y)
                split_losses[k] = loss.item()
            losses[split] = split_losses.mean().item()
        model.train()
        return losses

    # Logging history
    history_steps = []
    history_train_loss = []
    history_val_loss = []

    best_val_loss = float("inf")
    start_time = time.time()
    log_file_path = "outputs/logs/train.log"

    with open(log_file_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"MiniGPT Training Log | Parameters: {total_params:,} | Device: {device}\n")
        log_f.write(f"Step\tTrain Loss\tVal Loss\tTime (s)\n")

    print(f"\nStarting training for {max_steps} steps...")
    for step in range(1, max_steps + 1):
        # Sample batch
        x, y = get_batch_fast(train_tensor, context_length, batch_size, device=str(device))
        
        # Forward & backward pass
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluation interval
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{max_steps} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Elapsed: {elapsed:.1f}s")
            
            history_steps.append(step)
            history_train_loss.append(losses['train'])
            history_val_loss.append(losses['val'])

            with open(log_file_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"{step}\t{losses['train']:.4f}\t{losses['val']:.4f}\t{elapsed:.1f}\n")

            # Checkpoint best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": losses['train'],
                        "val_loss": losses['val'],
                        "config": config,
                        "vocab_size": tokenizer.vocab_size
                    },
                    checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
                    filename=train_cfg.get("checkpoint_name", "tiny_gpt_best.pt")
                )

            # Generate preview sample
            prompt_str = "ROMEO:\n"
            prompt_tokens = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)
            gen_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=60,
                temperature=gen_cfg.get("temperature", 0.8),
                top_k=gen_cfg.get("top_k", 40)
            )
            preview = tokenizer.decode(gen_tokens[0].tolist())
            print(f"--- [Sample Preview @ Step {step}] ---\n{preview}\n----------------------------------")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Plot loss curves (Figure 6)
    if history_steps:
        plot_loss_curves(
            history_train_loss,
            history_val_loss,
            history_steps,
            output_path="outputs/figures/loss_curve.png"
        )

    # Generate final samples from fixed prompts
    fixed_prompts = [
        "ROMEO:\nWhat light through",
        "JULIET:\nO Romeo, Romeo!",
        "HAMLET:\nTo be, or not to be,",
        "The king doth wake",
        "What is the cause of"
    ]
    print("\nGenerating final samples from fixed prompts (Milestone 5)...")
    samples_output_path = "outputs/samples/samples.txt"
    with open(samples_output_path, "w", encoding="utf-8") as sf:
        sf.write(f"=== MiniGPT Generated Samples ===\n")
        sf.write(f"Training Wall-Clock Time: {total_time:.2f}s | Final Val Loss: {best_val_loss:.4f}\n\n")
        for i, prompt in enumerate(fixed_prompts, 1):
            p_tok = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            gen_out = model.generate(p_tok, max_new_tokens=100, temperature=0.8, top_k=40)
            decoded_text = tokenizer.decode(gen_out[0].tolist())
            sf.write(f"--- Prompt {i}: {repr(prompt)} ---\n{decoded_text}\n\n")
            print(f"--- Prompt {i} ---\n{decoded_text}\n")
    print(f"Saved 5 samples to {samples_output_path}")

    return model, tokenizer, best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT from scratch")
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml", help="Path to config YAML")
    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
    args = parser.parse_args()
    train_model(config_path=args.config, override_steps=args.steps)
