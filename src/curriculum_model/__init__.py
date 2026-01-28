"""
Curriculum Model Package for Self-Refine ICL experiments.
"""

from .embedders import ComponentEmbedders
from .roles import Role, RoleEmbedding
from .special_tokens import SpecialTokens
from .sequence_builder import PositionalEncoder
from .output_heads import DualOutputHead, OutputHeadResult
from .component_model import (
    ComponentModelConfig,
    ComponentModelOutput,
    ComponentTransformerModel,
)

__all__ = [
    "ComponentEmbedders",
    "Role",
    "RoleEmbedding",
    "SpecialTokens",
    "PositionalEncoder",
    "DualOutputHead",
    "OutputHeadResult",
    "ComponentModelConfig",
    "ComponentModelOutput",
    "ComponentTransformerModel",
]
