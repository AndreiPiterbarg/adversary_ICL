"""CMA encoder for the feature-controlled controller + copied search infra.

`DiagonalCMAES` and the `Encoder` Protocol are COPIED VERBATIM from
`flip_flop/adversary/search.py` so the controller is proven to integrate with
the *exact* optimizer the project uses (ask/tell over `n_dims`), on CPU, with
no model forward pass.

`FeatureControlledEncoder` is the new contribution: a flat real-vector <->
`FeatureControlledFFL` codec satisfying the `Encoder` protocol, so the existing
`cma_search` would drive it unchanged (PLAN §3 C2 — 2 wiring lines, no other
pipeline change).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .distribution import D_FEATURES, N_PARAMS, FeatureControlledFFL, FFLDistribution


# ---------------------------------------------------------------------------
# Encoder protocol (verbatim)
# ---------------------------------------------------------------------------
@runtime_checkable
class Encoder(Protocol):
    """Structural type for every CMA encoder: a dim count plus a vector<->dist
    codec. `cma_search` only relies on this surface."""
    n_dims: int

    def decode(self, x: np.ndarray) -> FFLDistribution: ...
    def random_init(self, rng: np.random.Generator) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Diagonal CMA-ES (verbatim from flip_flop/adversary/search.py)
# ---------------------------------------------------------------------------
class DiagonalCMAES:
    """Minimal diagonal CMA-ES (sep-CMA-ES). Maintains only diagonal
    covariance (O(d) memory). Minimizes; we negate fitness externally."""

    def __init__(self, x0: np.ndarray, sigma: float, pop_size: int, seed: int = 0):
        self.d = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)

        self.C_diag = np.ones(self.d)
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1)
                        + self.c_sigma)
        self.c_c = (4 + self.mu_eff / self.d) / (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.d + 2) ** 2 + self.mu_eff),
        )
        self.chi_d = np.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        self.mu = mu
        self.generation = 0

    def ask(self) -> list:
        std = self.sigma * np.sqrt(self.C_diag)
        return [self.mean + std * self.rng.standard_normal(self.d)
                for _ in range(self.pop_size)]

    def tell(self, solutions: list, fitnesses: list):
        order = np.argsort(fitnesses)
        selected = [solutions[i] for i in order[: self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum([w * x for w, x in zip(self.weights, selected)], axis=0)

        std = np.sqrt(self.C_diag)
        std_safe = np.where(std > 1e-30, std, 1e-30)

        displacement = (self.mean - old_mean) / (self.sigma * std_safe)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * displacement

        h_sigma = int(
            np.linalg.norm(self.p_sigma)
            / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))
            < (1.4 + 2 / (self.d + 1)) * self.chi_d
        )
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * (self.mean - old_mean) / self.sigma

        artmp = np.array([(x - old_mean) / self.sigma for x in selected])
        C_mu_update = np.sum(
            [w * (a / std_safe) ** 2 for w, a in zip(self.weights, artmp)], axis=0
        )
        self.C_diag = (
            (1 - self.c_1 - self.c_mu) * self.C_diag
            + self.c_1 * (self.p_c ** 2 + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C_diag)
            + self.c_mu * C_mu_update * self.C_diag
        )
        self.C_diag = np.maximum(self.C_diag, 1e-20)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_d - 1)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)
        self.generation += 1


# ---------------------------------------------------------------------------
# The contribution: FeatureControlledEncoder
# ---------------------------------------------------------------------------
class FeatureControlledEncoder:
    """Real-vector <-> FeatureControlledFFL.

    Layout (length N_PARAMS = 3*(d+1) = 24 for d=7):
        [ wR(d) , cR(1) , wI(d) , cI(1) , wb(d) , cb(1) ]

    Logits are passed through unscaled (matching PiecewiseEncoder's
    raw-logit convention); small-sigma `random_init` keeps the initial
    instruction simplex near-uniform and bit_p1 ~ 0.5, so CMA starts from a
    well-posed, non-degenerate point. Floors / caps are fixed structural
    config (not searched) so degeneracy stays unreachable throughout search.
    """

    def __init__(self, T: int, *, p_w_min: float = 0.02, p_r_min: float = 0.02,
                 p_i_min: float = 0.0, gap_cap_frac: float = 0.5,
                 nwrites_cap_frac: float = 0.5, runlen_cap_frac: float = 0.25):
        self.T = T
        self.d = D_FEATURES
        self.n_dims = N_PARAMS
        self._fixed = dict(
            p_w_min=p_w_min, p_r_min=p_r_min, p_i_min=p_i_min,
            gap_cap_frac=gap_cap_frac, nwrites_cap_frac=nwrites_cap_frac,
            runlen_cap_frac=runlen_cap_frac,
        )

    def decode(self, x: np.ndarray) -> FeatureControlledFFL:
        x = np.asarray(x, dtype=float).reshape(self.n_dims)
        d = self.d
        i = 0
        wR = x[i:i + d]; i += d
        cR = float(x[i]); i += 1
        wI = x[i:i + d]; i += d
        cI = float(x[i]); i += 1
        wb = x[i:i + d]; i += d
        cb = float(x[i]); i += 1
        return FeatureControlledFFL(
            T=self.T, wR=wR, cR=cR, wI=wI, cI=cI, wb=wb, cb=cb, **self._fixed
        )

    def random_init(self, rng: np.random.Generator) -> np.ndarray:
        # Same convention/scale as PiecewiseEncoder: small Gaussian -> logits
        # near 0 -> near-uniform simplex (then affine-floored) and bit_p1 ~ 0.5.
        return rng.standard_normal(self.n_dims) * 0.3
