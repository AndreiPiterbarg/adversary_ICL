"""Multi-head causal attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadedAttention(nn.Module):

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        if emb_dim % n_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        self.head_dim = emb_dim // n_heads

        self.linear_Q = nn.Linear(emb_dim, emb_dim)
        self.linear_K = nn.Linear(emb_dim, emb_dim)
        self.linear_V = nn.Linear(emb_dim, emb_dim)
        self.output_projection = nn.Linear(emb_dim, emb_dim)

        # Register causal mask buffer
        self.register_buffer('causal_mask', torch.tril(torch.ones(512, 512)).view(1, 1, 512, 512))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch, seq_len, _ = x.size()

        Q = self.linear_Q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.linear_K(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.linear_V(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = self.dropout(attn_weights) @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_dim)
        out = self.output_projection(out)

        if return_attention:
            return out, attn_weights
        return out
