import os
import torch
import matplotlib.pyplot as plt
from src.positional_encoding import PositionalEncoding

def plot_positional_encoding(seq_len: int = 128, d_model: int = 128, output_path: str = "outputs/figures/positional_encoding.png"):
    """
    Visualizes Sinusoidal Positional Encoding (Figure 4 in assignment handout).
    Demonstrates that position information changes representations systematically across dimensions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pe_module = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    
    # pe buffer shape is (1, max_len, d_model) -> extract (seq_len, d_model)
    pe_matrix = pe_module.pe[0, :seq_len, :].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(pe_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.title("Sinusoidal Positional Encoding Matrix (Figure 4)\nPE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))", fontsize=12, pad=12)
    plt.xlabel("Embedding Dimension (d_model = 128)", fontsize=11)
    plt.ylabel("Token Position in Sequence (0 to 127)", fontsize=11)
    plt.colorbar(label="Encoding Value (-1 to 1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Figure 4] Positional encoding heatmap saved to: {output_path}")

if __name__ == "__main__":
    plot_positional_encoding()
