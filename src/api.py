import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from .gpt import MiniGPT
from .tokenizer import CharTokenizer
from .utils import load_checkpoint, get_device

app = FastAPI(
    title="MiniGPT REST API",
    description="Inference API for Causal Language Model trained from scratch (AI Engineering Assignment).",
    version="1.0.0"
)

# Global model state
model = None
tokenizer = None
model_config = None
device = None

class GenerateRequest(BaseModel):
    prompt: str = Field(default="ROMEO:\n", description="Input prompt for generation")
    max_new_tokens: int = Field(default=80, ge=1, le=500, description="Tokens to generate")
    temperature: float = Field(default=0.8, ge=0.01, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=40, ge=1, description="Top-k filtering parameter")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    num_tokens_generated: int
    model_parameters: int
    context_length: int
    temperature: float

@app.on_event("startup")
def startup_event():
    global model, tokenizer, model_config, device
    device = get_device("auto")
    checkpoint_path = "checkpoints/tiny_gpt_best.pt"
    tokenizer_path = "checkpoints/tokenizer.json"

    if os.path.exists(tokenizer_path):
        tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        print("[API Warning] Tokenizer file not found. Ensure training is executed.")

    if os.path.exists(checkpoint_path) and tokenizer is not None:
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
        model_config = checkpoint["config"]["model"]
        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            context_length=model_config["context_length"],
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"],
            ffn_dim=model_config.get("ffn_dim", 4 * model_config["d_model"]),
            dropout=0.0,
            pos_encoding_type=model_config.get("pos_encoding_type", "sinusoidal"),
            tie_weights=model_config.get("tie_weights", False)
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[API] MiniGPT successfully loaded with {model.get_num_params():,} parameters.")

@app.get("/")
def root():
    return {
        "message": "MiniGPT Autoregressive Causal Language Model API is online.",
        "endpoints": ["/health", "/model-info", "/generate", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {
        "total_parameters": model.get_num_params(),
        "context_length": model.context_length,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "d_model": model.d_model,
        "vocab_size": model.vocab_size,
        "pos_encoding_type": model.pos_encoding_type
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer is not initialized.")
    
    encoded = tokenizer.encode(req.prompt)
    if not encoded:
        encoded = [0]
    prompt_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
    
    output_tokens = model.generate(
        prompt_tensor,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k
    )
    
    generated_text = tokenizer.decode(output_tokens[0].tolist())
    
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=generated_text,
        num_tokens_generated=req.max_new_tokens,
        model_parameters=model.get_num_params(),
        context_length=model.context_length,
        temperature=req.temperature
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
