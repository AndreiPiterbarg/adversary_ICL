"""Transformer configuration."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_positions: int = 128
    dropout: float = 0.0
    pos_encoding_type: str = "none"  # "none", "learned"
