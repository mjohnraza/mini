import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadedAttention

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
 
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
        
            normed_x = self.attn_norm(x)
            if return_attention:
                attn_out, attn_weights = self.attn(normed_x, normed_x, normed_x, mask=mask, return_attention=True)
            else:
                attn_out = self.attn(normed_x, normed_x, normed_x, mask=mask)
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))
        else:
         
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
