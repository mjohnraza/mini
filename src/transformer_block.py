import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadedAttention

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    Two linear transformations with a non-linearity (GELU) and dropout in between.
    
    Formula:
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer Block for Causal Language Modeling (Milestone 3 & Screenshot 3).
    Includes:
        1. Multi-Head Attention (masked for causal autoregression)
        2. Layer Normalization
        3. Residual Connection
        4. Position-wise Feed-Forward Network
        5. Layer Normalization & Residual Connection
        
    Supports both Pre-LN (modern GPT default for training stability) and Post-LN (original paper).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pre_ln: bool = True
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.attn = MultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = None
        if self.pre_ln:
            # Modern Pre-LN architecture (GPT-2/GPT-3 style)
            normed_x = self.attn_norm(x)
            if return_attention:
                attn_out, attn_weights = self.attn(normed_x, normed_x, normed_x, mask=mask, return_attention=True)
            else:
                attn_out = self.attn(normed_x, normed_x, normed_x, mask=mask)
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))
        else:
            # Post-LN architecture (Original Attention is All You Need style)
            if return_attention:
                attn_out, attn_weights = self.attn(x, x, x, mask=mask, return_attention=True)
            else:
                attn_out = self.attn(x, x, x, mask=mask)
            x = self.attn_norm(x + attn_out)
            x = self.ffn_norm(x + self.ffn(x))
            
        if return_attention:
            return x, attn_weights
        return x


class EncoderLayer(nn.Module):
    """
    Bidirectional Encoder Layer (for BERT MLM comparison in Milestone 3 / Lesson 06).
    Applies unmasked (or bidirectional masked) self-attention across the entire sequence.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard Post-LN encoder layer matching reference screenshot
        x = self.attn_norm(x + self.attn(x, x, x, mask=mask))
        x = self.ffn_norm(x + self.ffn(x))
        return x
