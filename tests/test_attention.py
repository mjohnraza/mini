import math
import torch
import pytest
from src.attention import scaled_dot_product_attention, SelfAttention, MultiHeadedAttention

def test_hand_calculated_attention_step_by_step():
    """
    Validates the exact 3-token hand calculation required in Section 6:
    Sequence length = 3, d_k = 2, d_v = 2.
    Explicit matrices:
        Q = [[1.0, 0.0],
             [0.0, 2.0],
             [1.0, 1.0]]
        K = [[1.0, 1.0],
             [2.0, 0.0],
             [0.0, 1.0]]
        V = [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]]
    """
    Q = torch.tensor([[1.0, 0.0],
                      [0.0, 2.0],
                      [1.0, 1.0]])
    K = torch.tensor([[1.0, 1.0],
                      [2.0, 0.0],
                      [0.0, 1.0]])
    V = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])

    d_k = 2.0
    scale = math.sqrt(d_k)

    # 1. Dot product scores
    scores_expected = (Q @ K.T) / scale
    # Q @ K^T:
    # row 0: [1*1 + 0*1, 1*2 + 0*0, 1*0 + 0*1] = [1.0, 2.0, 0.0]
    # row 1: [0*1 + 2*1, 0*2 + 2*0, 0*0 + 2*1] = [2.0, 0.0, 2.0]
    # row 2: [1*1 + 1*1, 1*2 + 1*0, 1*0 + 1*1] = [2.0, 2.0, 1.0]
    expected_qk = torch.tensor([[1.0, 2.0, 0.0],
                                [2.0, 0.0, 2.0],
                                [2.0, 2.0, 1.0]])
    assert torch.allclose(Q @ K.T, expected_qk)

    # 2. Unmasked attention weights and output
    out, weights = scaled_dot_product_attention(Q, K, V)
    expected_weights = torch.softmax(scores_expected, dim=-1)
    assert torch.allclose(weights, expected_weights, atol=1e-5)
    expected_out = expected_weights @ V
    assert torch.allclose(out, expected_out, atol=1e-5)

    # 3. Causal masked attention
    # Causal mask: 1 for past and present, 0 for future
    mask = torch.tensor([[1, 0, 0],
                         [1, 1, 0],
                         [1, 1, 1]], dtype=torch.bool)
    out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)

    # Check that upper triangle of attention weights is strictly 0.0
    assert weights_masked[0, 1] == 0.0
    assert weights_masked[0, 2] == 0.0
    assert weights_masked[1, 2] == 0.0

    # Token 1 attends only to Token 1 -> weight is exactly 1.0
    assert torch.allclose(weights_masked[0, 0], torch.tensor(1.0))
    # Output for Token 1 must equal V[0]
    assert torch.allclose(out_masked[0], V[0])

def test_self_attention_module():
    batch_size = 2
    seq_len = 5
    d_model = 16
    out_size = 8
    layer = SelfAttention(d_model=d_model, output_size=out_size)
    layer.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    out, weights = layer(x, x, x, return_weights=True)
    assert out.shape == (batch_size, seq_len, out_size)
    assert weights.shape == (batch_size, seq_len, seq_len)
    # Rows of weights must sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

def test_multi_head_attention():
    batch_size = 3
    seq_len = 7
    d_model = 32
    num_heads = 4
    mha = MultiHeadedAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    out, attn_weights = mha(x, x, x, return_attention=True)
    assert out.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
