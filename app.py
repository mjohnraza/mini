"""
Gradio Web Application for MiniGPT
Interactive text generation and model analysis demo.
"""

import os
import gradio as gr

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

from src.gpt import MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import load_checkpoint

# Load Model & Tokenizer
CHECKPOINT_PATH = "checkpoints/tiny_gpt_best.pt"
TOKENIZER_PATH = "checkpoints/tokenizer.json"

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = CharTokenizer.load(TOKENIZER_PATH)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location="cpu")
model_cfg = checkpoint["config"]["model"]

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    context_length=model_cfg["context_length"],
    d_model=model_cfg["d_model"],
    n_heads=model_cfg["n_heads"],
    n_layers=model_cfg["n_layers"]
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

num_params = sum(p.numel() for p in model.parameters())

def _generate(prompt: str, max_tokens: int, temperature: float, top_k: int):
    if not prompt or len(prompt.strip()) == 0:
        return "Please enter a non-empty starting prompt."
    
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        out_indices = model.generate(
            idx,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None
        )
    
    generated_text = tokenizer.decode(out_indices[0].tolist())
    return generated_text

if has_spaces:
    generate_text = spaces.GPU(_generate)
else:
    generate_text = _generate

custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
"""

with gr.Blocks(title="MiniGPT: Causal Transformer from Scratch") as demo:
    gr.Markdown(
        f"""
        # 🎭 MiniGPT: Causal Transformer from Scratch
        **A 0.81M-parameter Causal GPT language model built entirely with PyTorch tensor primitives and trained on the Tiny Shakespeare corpus.**
        
        * **Architecture:** 4 Layers, 4 Attention Heads, $d_{{\\text{{model}}}} = 128$, Context Window $= 128$
        * **Total Parameters:** {num_params:,} (~0.81M)
        * **Pretrained weights:** None (trained completely from scratch)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt (Seed Text)",
                placeholder="Enter starting text (e.g. ROMEO:, JULIET:, To be, or not to be)...",
                lines=4,
                value="ROMEO:\n"
            )
            
            with gr.Accordion("Sampling Controls", open=True):
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=80,
                    step=5,
                    label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.8,
                    step=0.05,
                    label="Temperature (Creativity)"
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=65,
                    value=40,
                    step=5,
                    label="Top-k Filtering (0 = disabled)"
                )
            
            generate_btn = gr.Button("Generate Text", variant="primary")
            
            gr.Examples(
                examples=[
                    ["ROMEO:\n", 80, 0.8, 40],
                    ["JULIET:\n", 80, 0.8, 40],
                    ["HAMLET:\nTo be, or not to be,", 80, 0.8, 40],
                    ["The future of artificial intelligence", 80, 0.8, 40],
                    ["First Citizen:\n", 80, 0.8, 40]
                ],
                inputs=[prompt_input, max_tokens, temperature, top_k]
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Output",
                lines=12,
                interactive=False
            )
            
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(css=custom_css)
