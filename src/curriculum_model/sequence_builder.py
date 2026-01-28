"""
Positional encoding for the curriculum transformer.

Adds example-level positional encoding to sequences.
"""

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Adds example-level positional encoding to sequences.

    All tokens from the same example receive the same position embedding,
    indexed by example number rather than absolute token position.
    """

    def __init__(self, n_embd: int, max_examples: int = 64):
        super().__init__()
        self.n_embd = n_embd
        self.max_examples = max_examples
        self.position_embedding = nn.Embedding(max_examples, n_embd)

    def forward(
        self,
        tokens: torch.Tensor,
        example_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional encoding based on example positions.

        Args:
            tokens: Token embeddings of shape (batch_size, seq_len, n_embd)
            example_positions: Example index for each token, shape (batch_size, seq_len)

        Returns:
            Tokens with positional encoding added
        """
        pos_emb = self.position_embedding(example_positions)
        return tokens + pos_emb
