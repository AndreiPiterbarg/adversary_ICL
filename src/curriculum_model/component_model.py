"""
ComponentTransformerModel for in-context learning.

Integrates:
- Component embedders (vector, matrix, scalar)
- Role embedding layer
- SEP and MASK special tokens
- Transformer backbone
- Output head
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn

from .embedders import ComponentEmbedders
from .roles import RoleEmbedding
from .special_tokens import SpecialTokens
from .sequence_builder import PositionalEncoder
from .output_heads import DualOutputHead, OutputHeadResult

from custom_transformer import CustomGPTBackbone, TransformerConfig, GPTOutput


@dataclass
class ComponentModelConfig:
    """Configuration for ComponentTransformerModel."""
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_positions: int = 128
    max_examples: int = 64
    dropout: float = 0.0
    output_bias: bool = True
    max_iterations: int = 0  # 0 = no iteration embedding

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to TransformerConfig for backbone."""
        return TransformerConfig(
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_positions=self.n_positions,
            pos_encoding_type="none",  # We use example-level positional encoding
            dropout=self.dropout,
        )


@dataclass
class ComponentModelOutput:
    """Output from ComponentTransformerModel."""
    vector_output: torch.Tensor  # (batch_size, d)
    scalar_output: torch.Tensor  # (batch_size, 1)
    hidden_at_mask: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    attention_maps: Optional[List[torch.Tensor]] = None


class ComponentTransformerModel(nn.Module):
    """Component-based transformer for in-context learning."""

    def __init__(self, config: Optional[ComponentModelConfig] = None):
        super().__init__()

        if config is None:
            config = ComponentModelConfig()

        self.config = config
        self.d = config.d
        self.n_embd = config.n_embd

        # Component embedders
        self.embedders = ComponentEmbedders(config.d, config.n_embd)

        # Role embeddings
        self.role_embedding = RoleEmbedding(config.n_embd)

        # Special tokens
        self.special_tokens = SpecialTokens(config.n_embd)

        # Example-level positional encoder
        self.positional_encoder = PositionalEncoder(
            config.n_embd,
            max_examples=config.max_examples,
        )

        # Iteration embedding (for multi-step backprop)
        if config.max_iterations > 0:
            self.iteration_embedding = nn.Embedding(config.max_iterations, config.n_embd)
        else:
            self.iteration_embedding = None

        # Transformer backbone
        transformer_config = config.to_transformer_config()
        self.backbone = CustomGPTBackbone(transformer_config)

        # Output head
        self.output_head = DualOutputHead(
            config.n_embd,
            config.d,
            bias=config.output_bias,
        )

    def get_iteration_embedding(self, iteration_index: int) -> torch.Tensor:
        """Get embedding for a given refinement iteration index.

        Args:
            iteration_index: Which refinement step (0-indexed).

        Returns:
            Embedding tensor of shape (n_embd,), or zeros if disabled.
        """
        if self.iteration_embedding is None:
            return torch.zeros(self.n_embd, device=next(self.parameters()).device)
        idx = min(iteration_index, self.config.max_iterations - 1)
        return self.iteration_embedding.weight[idx]

    def forward(
        self,
        tokens: torch.Tensor,
        example_positions: torch.Tensor,
        mask_positions: torch.Tensor,
        return_hidden: bool = False,
        return_attention: bool = False,
        iteration_index: Optional[int] = None,
    ) -> ComponentModelOutput:
        """
        Forward pass.

        Args:
            tokens: Pre-embedded tokens, shape (batch_size, seq_len, n_embd)
            example_positions: Example index per token, shape (batch_size, seq_len)
            mask_positions: MASK token position, shape (batch_size,)
            return_hidden: Whether to return hidden states
            return_attention: Whether to return attention maps
            iteration_index: Which refinement iteration (for iteration embedding)

        Returns:
            ComponentModelOutput with predictions
        """
        batch_size = tokens.shape[0]

        # Add example-level positional encoding
        tokens_with_pos = self.positional_encoder(tokens, example_positions)

        # Add iteration embedding if enabled and provided
        if iteration_index is not None and self.iteration_embedding is not None:
            idx = min(iteration_index, self.config.max_iterations - 1)
            iter_emb = self.iteration_embedding.weight[idx]  # (n_embd,)
            tokens_with_pos = tokens_with_pos + iter_emb.unsqueeze(0).unsqueeze(0)

        # Pass through transformer backbone
        backbone_output = self.backbone(
            inputs_embeds=tokens_with_pos,
            return_attention=return_attention,
        )

        hidden_states = backbone_output.last_hidden_state

        # Extract hidden state at MASK positions
        batch_indices = torch.arange(batch_size, device=tokens.device)
        hidden_at_mask = hidden_states[batch_indices, mask_positions]

        # Compute outputs
        output_result: OutputHeadResult = self.output_head(hidden_at_mask)

        return ComponentModelOutput(
            vector_output=output_result.vector_output,
            scalar_output=output_result.scalar_output,
            hidden_at_mask=hidden_at_mask if return_hidden else None,
            last_hidden_state=hidden_states if return_hidden else None,
            attention_maps=backbone_output.attention_maps if return_attention else None,
        )
