import torch
import pytest
from src.gpt import MiniGPT

def test_tiny_overfit_batch():
    """
    Overfit test on a tiny batch (Milestone 4 gate & Submission Acceptance Test 3).
    Verifies that the model can reduce training loss substantially, proving that:
        1. Backpropagation and gradient flow are healthy.
        2. Target shifting is aligned with predictions.
        3. Loss formulation and optimizer updates work properly.
    """
    torch.manual_seed(42)
    vocab_size = 30
    context_length = 16
    batch_size = 4
    d_model = 64
    n_heads = 4
    n_layers = 2

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Fixed single tiny batch
    x = torch.randint(0, vocab_size, (batch_size, context_length))
    y = torch.randint(0, vocab_size, (batch_size, context_length))

    # Initial loss
    model.eval()
    with torch.no_grad():
        _, initial_loss, _ = model(x, y)
    initial_loss_val = initial_loss.item()
    print(f"\n[Overfit Test] Initial Loss: {initial_loss_val:.4f}")

    # Overfit loop for 35 iterations
    model.train()
    final_loss_val = initial_loss_val
    for step in range(35):
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()
        final_loss_val = loss.item()

    print(f"[Overfit Test] Final Loss after 35 steps: {final_loss_val:.4f}")
    
    # Assert loss decreased substantially (at least 70% reduction)
    assert final_loss_val < initial_loss_val * 0.3, f"Loss did not decrease sufficiently: {initial_loss_val:.4f} -> {final_loss_val:.4f}"
    assert final_loss_val < 0.8, f"Final loss is still high: {final_loss_val:.4f}"
    print("[Overfit Test] PASSED: Model easily overfits tiny batch!")
