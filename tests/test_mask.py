import torch
import pytest
from src.gpt import MiniGPT

def test_causal_mask_structure():
    """Verify causal mask is strictly lower-triangular."""
    model = MiniGPT(vocab_size=65, context_length=16, d_model=32, n_heads=2, n_layers=2)
    mask = model.causal_mask[:8, :8]
    expected_tril = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert torch.equal(mask, expected_tril)

def test_future_tokens_receive_zero_attention():
    """Verify that attention weights for future positions (j > i) are exactly 0.0."""
    model = MiniGPT(vocab_size=65, context_length=16, d_model=32, n_heads=2, n_layers=2)
    model.eval()
    x = torch.randint(0, 65, (1, 8))
    _, _, attentions = model(x, return_attentions=True)
    
    for layer_idx, layer_attn in enumerate(attentions):
        # Shape: (B, num_heads, seq_len, seq_len)
        for h in range(layer_attn.size(1)):
            head_weights = layer_attn[0, h]
            upper_tri = torch.triu(head_weights, diagonal=1)
            assert torch.all(upper_tri == 0.0), f"Layer {layer_idx} Head {h} leaks future attention!"

def test_causal_gradient_isolation():
    """
    Rigorously verifies causal correctness:
    The representation and output at position t MUST NOT have any gradient
    with respect to input tokens at position t' > t.
    """
    model = MiniGPT(vocab_size=65, context_length=16, d_model=32, n_heads=2, n_layers=2)
    model.eval()

    # Pass dummy embeddings with requires_grad=True
    seq_len = 8
    dummy_input = torch.randint(0, 65, (1, seq_len))
    emb = model.tok_embeddings(dummy_input)
    emb.retain_grad()

    # Modify model forward slightly to feed emb directly
    x = emb * (model.d_model ** 0.5)
    x = model.pos_encoding(x)
    mask = model.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
    for block in model.blocks:
        x = block(x, mask=mask)
    x = model.final_norm(x)
    logits = model.lm_head(x)

    # Compute a scalar loss using ONLY position t=2
    target_pos = 2
    loss_at_t = logits[0, target_pos].sum()
    loss_at_t.backward()

    # Check gradients on input embeddings:
    # positions 0, 1, 2 should have non-zero gradients (they contribute to position 2)
    # positions 3, 4, 5, 6, 7 MUST have strictly zero gradients!
    future_grads = emb.grad[0, target_pos + 1 :]
    assert torch.all(future_grads == 0.0), "Information leak: Future tokens have non-zero gradients on past position!"
    past_grads = emb.grad[0, : target_pos + 1]
    assert not torch.all(past_grads == 0.0), "Past tokens should contribute gradients to target position."
