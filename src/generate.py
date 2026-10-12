import argparse
import os
import torch

from .gpt import MiniGPT
from .tokenizer import CharTokenizer
from .utils import load_checkpoint, get_device

def generate_cli(
    prompt: str,
    checkpoint_path: str = "checkpoints/tiny_gpt_best.pt",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    device_str: str = "auto"
):
    device = get_device(device_str)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first via: python -m src.train")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run training or prepare tokenizer first.")

    # Load tokenizer
    tokenizer = CharTokenizer.load(tokenizer_path)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    config = checkpoint["config"]
    model_cfg = config["model"]

    # Reconstruct model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=0.0, # zero dropout for inference
        pos_encoding_type=model_cfg.get("pos_encoding_type", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_params = model.get_num_params()

    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        encoded_prompt = [0]
    prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    # Autoregressive generation
    output_tokens = model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    generated_text = tokenizer.decode(output_tokens[0].tolist())

    # Format output according to assignment Section 12.1
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}\n")
    print(f"model parameters: {num_params / 1e6:.2f}M ({num_params:,})")
    print(f"context length: {model_cfg['context_length']}")
    print(f"temperature: {temperature}")

    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text autoregressively using trained MiniGPT")
    parser.add_argument("--prompt", type=str, default="First Citizen:\nBefore we proceed any further,", help="Initial prompt text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tiny_gpt_best.pt", help="Path to checkpoint file")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer json file")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling cutoff")
    args = parser.parse_args()

    generate_cli(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
