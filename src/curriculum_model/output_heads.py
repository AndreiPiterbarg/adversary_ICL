"""
Output head for the curriculum transformer.

Projects from hidden dimension to vector output.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class OutputHeadResult:
    """Result from output head."""
    vector_output: torch.Tensor  # (batch_size, d)
    scalar_output: torch.Tensor  # (batch_size, 1) - computed but typically unused


class VectorHead(nn.Module):
    """Output head for vector predictions."""

    def __init__(self, n_embd: int, d: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_embd, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DualOutputHead(nn.Module):
    """
    Dual output head module (vector and scalar).

    Both heads computed for consistent gradient flow,
    but typically only vector output is used.
    """

    def __init__(self, n_embd: int, d: int, bias: bool = True):
        super().__init__()
        self.vector_head = VectorHead(n_embd, d, bias=bias)
        self.scalar_head = nn.Linear(n_embd, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> OutputHeadResult:
        return OutputHeadResult(
            vector_output=self.vector_head(x),
            scalar_output=self.scalar_head(x),
        )
