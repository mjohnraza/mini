import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .transformer_block import TransformerBlock
from .positional_encoding import PositionalEncoding, LearnedPositionalEncoding

class MiniGPT(nn.Module):
    """
    MiniGPT: Autoregressive Causal Language Model built from scratch.
    Implements all core components required by the AI Engineering assignment:
        - Token Embedding
        - Positional Encoding (Sinusoidal or Learned)
        - Causal Masked Multi-Head Attention Blocks
        - Layer Normalization & Residual Paths
        - Language Model Head with optional Weight Tying
        - Cross-Entropy Next-Token Prediction
        - Autoregressive Generation with Temperature & Top-K Sampling
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        pos_encoding_type: str = "sinusoidal",
        tie_weights: bool = False,
        pre_ln: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model
        self.dropout_rate = dropout
        self.pos_encoding_type = pos_encoding_type
        self.tie_weights = tie_weights
        self.pre_ln = pre_ln

        # 1. Token Embedding
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        if pos_encoding_type == "sinusoidal":
            self.pos_encoding = PositionalEncoding(d_model, max_len=context_length, dropout=dropout)
        elif pos_encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=context_length, dropout=dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding_type}")

        # 3. Stack of Causal Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=self.ffn_dim,
                dropout=dropout,
                pre_ln=pre_ln
            )
            for _ in range(n_layers)
        ])

        # 4. Final Normalization
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Language Modeling Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional Weight Tying (Press & Wolf, 2017)
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # Causal mask buffer: lower triangular matrix where 1 allows attention, 0 forbids future tokens
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        )

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total number of model parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass for next-token prediction.
        
        Args:
            idx: Input token IDs of shape (batch_size, seq_len)
            targets: Optional shifted target token IDs of shape (batch_size, seq_len)
            return_attentions: Whether to collect attention weight maps from all blocks
            
        Returns:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size)
            loss: Token cross-entropy loss if targets provided, else None
            attentions: List of attention matrices per layer if requested
        """
        B, T = idx.shape
        assert T <= self.context_length, f"Cannot forward sequence length {T}, context length is {self.context_length}"

        # Token embedding
        x = self.tok_embeddings(idx) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Slice causal mask for the current sequence length T
        # causal_mask has 1 for allowed positions (j <= i), 0 for future (j > i)
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, T)

        # Pass through Transformer blocks
        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            if return_attentions:
                x, attn = block(x, mask=mask, return_attention=True)
                all_attentions.append(attn)
            else:
                x = block(x, mask=mask)

        # Final LayerNorm
        x = self.final_norm(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        # Compute next-token cross-entropy loss if targets are given
        loss = None
        if targets is not None:
            # Flatten to (B * T, vocab_size) and (B * T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        if return_attentions:
            return logits, loss, all_attentions
        return logits, loss, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            idx: Starting prompt token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of future tokens to predict and append
            temperature: Softmax temperature (> 0.0). Lower = more conservative, higher = more random
            top_k: Optional top-k filter to retain only k highest logits
            
        Returns:
            Generated sequence of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds context_length
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            # Take logits at the final time-step: shape (B, vocab_size)
            logits = logits[:, -1, :]
            
            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature
            else:
                # Greedy argmax
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                continue
                
            # Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            # Concatenate to generated context
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx
