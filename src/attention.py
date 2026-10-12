import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core Scaled Dot-Product Attention mechanism (Milestone 1).
    Calculates attention outputs and softmax weights given Q, K, V.
    
    Formula:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V
        
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor where 1/True allows attention, 0/False forbids it.
        dropout: Optional Dropout module applied to attention weights.
        
    Returns:
        output: Weighted sum over values of shape (..., seq_len_q, d_v)
        weights: Attention weight distribution of shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    # Scaled dot-product: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # Match dimensions between mask and scores
        while mask.dim() > scores.dim():
            mask = mask.squeeze(0)
        # Fill forbidden positions (where mask == 0 or False) with -infinity
        scores = scores.masked_fill(mask == 0, -float('inf'))
        
    weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        weights = dropout(weights)
        
    output = torch.matmul(weights, v)
    return output, weights


class SelfAttention(nn.Module):
    """
    Single Self-Attention head with linear projections for Query, Key, and Value.
    Follows pedagogical architecture shown in course curriculum screenshot.
    """
    def __init__(self, d_model: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_size = output_size
        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Linear projections
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        
        output, weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        
        if return_weights:
            return output, weights
        return output


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module (Milestone 2).
    Spans multiple independent SelfAttention heads, concatenates their outputs,
    and applies a final linear projection.
    
    Matches exact structure depicted in coding screenshot:
        - List of self-attention heads: d_model -> output_size (d_model // num_heads)
        - Concatenation along the feature dimension
        - Linear projection back to d_model
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_output_size = d_model // num_heads
        
        # Parallel attention heads
        self.attentions = nn.ModuleList([
            SelfAttention(d_model, self.attn_output_size, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # Output projection layer
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Collect outputs and attention weights from each head
        head_outputs = []
        head_weights = []
        
        for head in self.attentions:
            out, weights = head(q, k, v, mask=mask, return_weights=True)
            head_outputs.append(out)
            head_weights.append(weights)
            
        # Concatenate heads along feature dimension: (B, T, num_heads * head_dim) == (B, T, d_model)
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Final linear projection
        projected = self.output(concatenated)
        projected = self.dropout(projected)
        
        if return_attention:
            # Stack attention weights across heads: (B, num_heads, T, T)
            stacked_weights = torch.stack(head_weights, dim=1)
            return projected, stacked_weights
            
        return projected
