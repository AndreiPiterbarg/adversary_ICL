"""Liu R4 sampler: uniform mixture of {FFL(0.1), FFL(0.9), FFL(0.98)}.

Used as the published-baseline control. Per the paper, R4 trains from scratch
(not continue-train) for the standard 10000 steps on a per-sequence-uniform
draw from these three stationary FFL distributions. We replicate exactly that
sampling, no clustering, no curriculum, no bisection — pure paper-baseline.

Why we need this: without a hand-crafted control, our automated cumulative
loop has no comparator. After R1+R2 we will be able to claim
"automated loop {beats|matches|loses to} hand-crafted R4" only if R4 is run
under the same evaluation battery.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import sample_ffl


# Paper R4: uniform mixture of these three. Order doesn't matter; uniform draw.
R4_PAPER_PIs: tuple[float, ...] = (0.10, 0.90, 0.98)


class R4MixSampler:
    """Per-sequence uniform draw from K stationary FFL(p_i) distributions.

    Equivalent to MixedSampler with replay_frac=0 and K Stationary "families,"
    but specialized so it does not require constructing Family objects (and
    thus does not activate the balanced-selection path in train.py — Liu R4
    explicitly does NOT do early stopping; it runs the full 10k steps).
    """

    def __init__(self, T: int, p_i_values: tuple[float, ...] = R4_PAPER_PIs):
        assert T % 2 == 0 and T >= 4
        assert len(p_i_values) >= 1
        self.T = T
        self.p_i_values = tuple(p_i_values)

    def __call__(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        out = torch.empty(batch_size, self.T, dtype=torch.long)
        # Per-sequence uniform draw of which p_i to use.
        idx = rng.integers(0, len(self.p_i_values), size=batch_size)
        for i, p_i in enumerate(self.p_i_values):
            mask = idx == i
            n = int(mask.sum())
            if n == 0:
                continue
            out[mask] = sample_ffl(self.T, p_i, n, rng)
        return out

    def describe(self) -> dict:
        return {"T": self.T, "p_i_values": list(self.p_i_values),
                "kind": "R4MixSampler"}
