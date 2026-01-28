"""Custom GPT backbone."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional

from .block import TransformerBlock
from .positional import PositionalEncoding
from .config import TransformerConfig


@dataclass
class GPTOutput:
    """Output from transformer backbone."""
    last_hidden_state: torch.Tensor
    attention_maps: Optional[List[torch.Tensor]] = None


class CustomGPTBackbone(nn.Module):
    """GPT-style transformer backbone."""

    def __init__(self, config: TransformerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()

        self.config = config

        # Positional encoding (only for learned mode)
        self.pos_encoding = None
        if config.pos_encoding_type == "learned":
            self.pos_encoding = PositionalEncoding(config.n_embd, max_len=config.n_positions)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head, config.dropout)
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_attention: bool = False
    ) -> GPTOutput:
        x = inputs_embeds

        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_maps = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, w = block(x, return_attention=True)
                attn_maps.append(w)
            else:
                x = block(x)

        x = self.final_norm(x)
        return GPTOutput(last_hidden_state=x, attention_maps=attn_maps)
