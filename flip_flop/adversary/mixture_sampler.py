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


class FixedDatasetSampler:
    """Sampler backed by a saved (N, T) LongTensor of pre-materialized
    sequences. Replays uniformly with replacement, so a training run that
    asks for `train_steps * batch_size` total sequences will see each
    saved sequence ~equally often when N >= train_steps * batch_size, and
    will epoch over a smaller set otherwise.

    Pluggable into train.py via `train(cfg, sampler=FixedDatasetSampler(...))`.
    Used by run_retrain_from_dataset.py to retrain a fresh model on the
    dataset that a prior training run dumped.
    """

    def __init__(self, tokens: torch.LongTensor, source: str = ""):
        assert tokens.ndim == 2, f"expected (N, T), got {tuple(tokens.shape)}"
        assert tokens.dtype == torch.long, f"expected int64, got {tokens.dtype}"
        self.tokens = tokens
        self.T = int(tokens.shape[1])
        self.N = int(tokens.shape[0])
        self.source = source

    def __call__(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        idx = rng.integers(0, self.N, size=batch_size)
        return self.tokens[idx]

    def describe(self) -> dict:
        return {"kind": "FixedDatasetSampler", "T": self.T, "N": self.N,
                "source": self.source}
