"""
Component embedders for vectors, matrices, and scalars.
"""

import torch
import torch.nn as nn


class VectorEmbedder(nn.Module):
    """Embeds vectors of dimension d to n_embd."""

    def __init__(self, d: int, n_embd: int):
        super().__init__()
        self.linear = nn.Linear(d, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MatrixEmbedder(nn.Module):
    """Embeds d x d matrices to n_embd (flattened row-major)."""

    def __init__(self, d: int, n_embd: int):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d * d, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(*batch_shape, self.d * self.d)
        return self.linear(x_flat)


class ScalarEmbedder(nn.Module):
    """Embeds scalars to n_embd."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.linear = nn.Linear(1, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        return self.linear(x)


class ComponentEmbedders(nn.Module):
    """Container for vector, matrix, and scalar embedders."""

    def __init__(self, d: int, n_embd: int):
        super().__init__()
        self.d = d
        self.n_embd = n_embd
        self.vector = VectorEmbedder(d, n_embd)
        self.matrix = MatrixEmbedder(d, n_embd)
        self.scalar = ScalarEmbedder(n_embd)
