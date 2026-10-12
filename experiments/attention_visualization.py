import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import load_checkpoint, get_device

def plot_causal_mask(context_length: int = 16, output_path: str = "outputs/figures/causal_mask.png"):
    """
    Visualizes the Causal Mask matrix heatmap (Figure 2 in assignment handout).
    Demonstrates that future positions are strictly blocked from receiving attention.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Lower triangular matrix of 1s (allowed) and 0s (blocked future tokens)
    mask = torch.tril(torch.ones(context_length, context_length)).numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(mask, cmap="Blues", interpolation="nearest")
    plt.title(f"Causal Attention Mask ({context_length} x {context_length})\n[1 = Allowed, 0 = Masked / Future]", fontsize=13, pad=12)
    plt.xlabel("Key Position (Source)", fontsize=11)
    plt.ylabel("Query Position (Target)", fontsize=11)
    plt.colorbar(label="Attention Mask Value", ticks=[0, 1])

    # Annotate positions
    for i in range(context_length):
        for j in range(context_length):
            text = "1" if mask[i, j] == 1 else "0"
            color = "white" if mask[i, j] == 1 else "black"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure 2] Causal mask saved to: {output_path}")


def plot_attention_map(
    prompt: str = "ROMEO:\nWhat light",
    checkpoint_path: str = "checkpoints/tiny_gpt_best.pt",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    layer_idx: int = 0,
    head_idx: int = 0,
    output_path: str = "outputs/figures/attention_map.png"
):
    """
    Visualizes attention weights from a trained MiniGPT model (Figure 7 in assignment).
    Inspects actual learned attention patterns across token positions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    device = get_device("auto")

    if not os.path.exists(checkpoint_path) or not os.path.exists(tokenizer_path):
        print(f"[Warning] Trained checkpoint not found. Creating preview with initialized model.")
        tokenizer = CharTokenizer(list(" \n!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
        model = MiniGPT(vocab_size=tokenizer.vocab_size, context_length=64, d_model=64, n_heads=4, n_layers=2).to(device)
    else:
        tokenizer = CharTokenizer.load(tokenizer_path)
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
        model_cfg = checkpoint["config"]["model"]
        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            context_length=model_cfg["context_length"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
            dropout=0.0
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    encoded = tokenizer.encode(prompt)
    if len(encoded) > 32:
        encoded = encoded[:32]
        prompt = prompt[:32]
        
    tokens_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, _, all_attentions = model(tokens_tensor, return_attentions=True)

    # all_attentions is list of layers: each tensor is (B, num_heads, T, T)
    attn_layer = all_attentions[layer_idx][0, head_idx].cpu().numpy()
    
    char_labels = [repr(ch)[1:-1] for ch in prompt]

    plt.figure(figsize=(8, 7))
    plt.imshow(attn_layer, cmap="magma", interpolation="nearest")
    plt.title(f"MiniGPT Attention Weights Map\nLayer {layer_idx + 1}, Head {head_idx + 1}", fontsize=13, pad=12)
    plt.xlabel("Key Position (Attended To)", fontsize=11)
    plt.ylabel("Query Position (Attending From)", fontsize=11)
    plt.xticks(range(len(char_labels)), char_labels, rotation=45, fontsize=9)
    plt.yticks(range(len(char_labels)), char_labels, fontsize=9)
    plt.colorbar(label="Attention Weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure 7] Attention map saved to: {output_path}")

if __name__ == "__main__":
    plot_causal_mask(context_length=12, output_path="outputs/figures/causal_mask.png")
    plot_attention_map(output_path="outputs/figures/attention_map.png")
