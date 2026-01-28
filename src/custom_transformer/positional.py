"""Positional encoding."""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, emb_dim: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
