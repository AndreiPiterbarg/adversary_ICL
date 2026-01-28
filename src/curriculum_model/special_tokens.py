"""
Special tokens (SEP and MASK) for sequence structure.
"""

import torch
import torch.nn as nn


class SpecialTokens(nn.Module):
    """Learned SEP and MASK tokens."""

    def __init__(self, n_embd: int, init_std: float = 0.02):
        super().__init__()
        self.sep = nn.Parameter(torch.empty(n_embd))
        self.mask = nn.Parameter(torch.empty(n_embd))
        nn.init.normal_(self.sep, mean=0.0, std=init_std)
        nn.init.normal_(self.mask, mean=0.0, std=init_std)

    def get_sep_batch(self, batch_size: int) -> torch.Tensor:
        return self.sep.unsqueeze(0).expand(batch_size, -1)

    def get_mask_batch(self, batch_size: int) -> torch.Tensor:
        return self.mask.unsqueeze(0).expand(batch_size, -1)
