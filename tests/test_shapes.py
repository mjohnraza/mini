import torch
import pytest
from src.gpt import MiniGPT
from src.positional_encoding import PositionalEncoding
from src.attention import SelfAttention, MultiHeadedAttention
from src.transformer_block import FeedForward, TransformerBlock

def test_tensor_shape_trace_full_pipeline():
    """
    Traces and verifies tensor shapes through every transformation stage
    as required for Milestone 2 and Figure 3.
    """
    batch_size = 2
    seq_len = 16
    vocab_size = 65
    d_model = 64
    num_heads = 4
    head_dim = d_model // num_heads  # 16
    ffn_dim = 256

    print("\n--- Tensor Shape Trace Verification ---")
    # 1. Input Token IDs
    x_idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"1. Input Token IDs: {list(x_idx.shape)} (Expected: [B, T])")
    assert x_idx.shape == (batch_size, seq_len)

    # 2. Token Embedding
    tok_emb = torch.nn.Embedding(vocab_size, d_model)
    x_tok = tok_emb(x_idx)
    print(f"2. Token Embedding: {list(x_tok.shape)} (Expected: [B, T, d_model])")
    assert x_tok.shape == (batch_size, seq_len, d_model)

    # 3. Positional Encoding
    pos_enc = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)
    x_pos = pos_enc(x_tok)
    print(f"3. After Positional Encoding: {list(x_pos.shape)} (Expected: [B, T, d_model])")
    assert x_pos.shape == (batch_size, seq_len, d_model)

    # 4. Multi-Head Attention Head Projections
    single_head = SelfAttention(d_model=d_model, output_size=head_dim, dropout=0.0)
    q = single_head.query(x_pos)
    k = single_head.key(x_pos)
    v = single_head.value(x_pos)
    print(f"4. Single Head Q, K, V Projections: {list(q.shape)} (Expected: [B, T, head_dim])")
    assert q.shape == (batch_size, seq_len, head_dim)
    assert k.shape == (batch_size, seq_len, head_dim)
    assert v.shape == (batch_size, seq_len, head_dim)

    # 5. Attention Weights
    scores = torch.matmul(q, k.transpose(-2, -1))
    print(f"5. Attention Scores Matrix: {list(scores.shape)} (Expected: [B, T, T])")
    assert scores.shape == (batch_size, seq_len, seq_len)

    # 6. Multi-Head Attention Module
    mha = MultiHeadedAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mha_out, mha_weights = mha(x_pos, x_pos, x_pos, mask=mask, return_attention=True)
    print(f"6. Multi-Head Attention Output: {list(mha_out.shape)} (Expected: [B, T, d_model])")
    print(f"   Multi-Head Weights (all heads): {list(mha_weights.shape)} (Expected: [B, n_heads, T, T])")
    assert mha_out.shape == (batch_size, seq_len, d_model)
    assert mha_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # 7. Feed-Forward Network
    ffn = FeedForward(d_model=d_model, d_ff=ffn_dim, dropout=0.0)
    ffn_out = ffn(mha_out)
    print(f"7. Feed-Forward Output: {list(ffn_out.shape)} (Expected: [B, T, d_model])")
    assert ffn_out.shape == (batch_size, seq_len, d_model)

    # 8. Full MiniGPT Model Logits
    model = MiniGPT(vocab_size=vocab_size, context_length=seq_len, d_model=d_model, n_heads=num_heads, n_layers=2)
    logits, _, _ = model(x_idx)
    print(f"8. MiniGPT Vocabulary Logits: {list(logits.shape)} (Expected: [B, T, vocab_size])")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("All tensor shapes verified successfully!")
