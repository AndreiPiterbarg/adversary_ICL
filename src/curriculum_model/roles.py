"""
Role embeddings for semantic roles in token composition.

Token = embed_component(component) + embed_role(role_index)
"""

import torch
import torch.nn as nn
from enum import IntEnum


class Role(IntEnum):
    """Semantic roles for components."""
    MATRIX = 0        # The operator matrix (e.g., A in Ax = b)
    VEC_PRIMARY = 1   # Primary vector operand
    VEC_SECONDARY = 2 # Secondary vector / current estimate
    VEC_BIAS = 3      # Bias vector (e.g., b in Ax - b)
    SCALAR = 4        # Scalar value
    OUTPUT = 5        # Output/target position


NUM_ROLES = len(Role)


class RoleEmbedding(nn.Module):
    """Learned embeddings for semantic roles."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.embedding = nn.Embedding(NUM_ROLES, n_embd)

    def forward(self, role_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(role_indices)

    def get_role(self, role: Role) -> torch.Tensor:
        idx = torch.tensor(int(role), device=self.embedding.weight.device)
        return self.embedding(idx)

    # Convenience methods for tests
    def get_matrix_role(self) -> torch.Tensor:
        return self.get_role(Role.MATRIX)

    def get_primary_role(self) -> torch.Tensor:
        return self.get_role(Role.VEC_PRIMARY)

    def get_secondary_role(self) -> torch.Tensor:
        return self.get_role(Role.VEC_SECONDARY)

    def get_bias_role(self) -> torch.Tensor:
        return self.get_role(Role.VEC_BIAS)

    def get_scalar_role(self) -> torch.Tensor:
        return self.get_role(Role.SCALAR)

    def get_output_role(self) -> torch.Tensor:
        return self.get_role(Role.OUTPUT)


def compose_token(component_embedding: torch.Tensor, role_embedding: torch.Tensor) -> torch.Tensor:
    """Compose token from component and role embeddings: token = component + role."""
    return component_embedding + role_embedding
