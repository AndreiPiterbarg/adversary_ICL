"""Per-sequence mixture sampler for VRM-style retrain.

Each sequence is independently drawn from either:
  - the base distribution FFL(base_p_i)         with prob replay_frac
  - one of the given families (uniform)         with prob 1 - replay_frac

Pluggable into train.py via `train(cfg, sampler=MixedSampler(...))`.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import sample_ffl
from .family import Family


class MixedSampler:
    def __init__(
        self,
        T: int,
        base_p_i: float,
        families: list[Family],
        replay_frac: float = 0.5,
    ):
        assert 0.0 <= replay_frac <= 1.0
        assert len(families) >= 1, "need at least one family for the adversarial mixture"
        self.T = T
        self.base_p_i = base_p_i
        self.families = families
        self.replay_frac = replay_frac

    def __call__(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        is_base = rng.random(batch_size) < self.replay_frac
        n_base = int(is_base.sum())
        n_adv = batch_size - n_base

        out = torch.empty(batch_size, self.T, dtype=torch.long)

        if n_base > 0:
            out[is_base] = sample_ffl(self.T, self.base_p_i, n_base, rng)

        if n_adv > 0:
            fam_idx = rng.integers(0, len(self.families), size=n_adv)
            adv_positions = np.where(~is_base)[0]
            # Batch per family for efficiency.
            for f_i, fam in enumerate(self.families):
                mask = fam_idx == f_i
                if not mask.any():
                    continue
                n = int(mask.sum())
                fam_tokens = fam.sample(n, rng)
                out[adv_positions[mask]] = fam_tokens

        return out

    def describe(self) -> dict:
        return {
            "T": self.T,
            "base_p_i": self.base_p_i,
            "replay_frac": self.replay_frac,
            "families": [f.to_dict() for f in self.families],
        }
