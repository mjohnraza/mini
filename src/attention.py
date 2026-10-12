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
  
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        while mask.dim() > scores.dim():
            mask = mask.squeeze(0)
       
        scores = scores.masked_fill(mask == 0, -float('inf'))
        
    weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        weights = dropout(weights)
        
    output = torch.matmul(weights, v)
    return output, weights


class SelfAttention(nn.Module):
   
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
  
        head_outputs = []
        head_weights = []
        
        for head in self.attentions:
            out, weights = head(q, k, v, mask=mask, return_weights=True)
            head_outputs.append(out)
            head_weights.append(weights)
            
    
        concatenated = torch.cat(head_outputs, dim=-1)
        
       
        projected = self.output(concatenated)
        projected = self.dropout(projected)
        
        if return_attention:
          
            stacked_weights = torch.stack(head_weights, dim=1)
            return projected, stacked_weights
            
        return projected
