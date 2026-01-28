"""Transformer block with pre-norm architecture."""

import torch
import torch.nn as nn
from .attention import MultiHeadedAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
    x -> LayerNorm -> Attention -> + residual
      -> LayerNorm -> FFN -> + residual
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadedAttention(n_embd, n_heads, dropout)
        self.ffw = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights = self.attention(self.layer_norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ffw(self.layer_norm2(x))
            return x, attn_weights
        else:
            x = x + self.attention(self.layer_norm1(x))
            x = x + self.ffw(self.layer_norm2(x))
            return x
