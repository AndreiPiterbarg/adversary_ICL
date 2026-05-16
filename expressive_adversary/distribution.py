"""FFL distribution samplers.

Reference families (`Stationary`, `BitMarkov`, `WriteFlipRate`, `Piecewise`)
are COPIED VERBATIM from `flip_flop/adversary/distribution.py` so this package
is standalone and so the subsumption tests compare the controller against the
*exact* project code, not a re-implementation.

`FeatureControlledFFL` is the new contribution: one causal log-linear controller
over the instruction / free-bit emission process. Read-position bits are forced
by the centralized `enforce_read_determinism` (in `_finalize`), so EVERY sample
is valid FFL and the LSTM register-automaton skyline stays at 0% for ANY weight
setting — validity and legitimacy are free, exactly as
`docs/PLAN_expressive_adversary.md` §1 argues.

Every hand-coded family is this controller with the feature weights restricted
to one feature group; the subsumption-witness constructors
(`emulate_stationary`, `emulate_write_flip`, `emulate_bit_markov`,
`emulate_piecewise_ramp`) are the executable proof of that claim.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .data import W, R, I, enforce_read_determinism, interleave


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class FFLDistribution(abc.ABC):
    """Abstract sampler over valid FFL(T) strings."""

    T: int

    @abc.abstractmethod
    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        """Return a (batch_size, T) LongTensor of valid FFL tokens."""

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """JSON-serializable config (round-trips through from_dict)."""

    @classmethod
    def from_dict(cls, d: dict) -> "FFLDistribution":
        name = d["name"]
        sub = REGISTRY[name]
        return sub._from_dict(d)

    @classmethod
    @abc.abstractmethod
    def _from_dict(cls, d: dict) -> "FFLDistribution":
        ...

    def descriptor(self) -> dict:
        """Short human-readable summary for logs."""
        return self.to_dict()

    # Helper used by all samplers — the single centralized validity gate.
    def _finalize(self, inst: np.ndarray, data: np.ndarray) -> torch.LongTensor:
        inst[:, 0] = W
        inst[:, -1] = R
        enforce_read_determinism(inst, data)
        return interleave(inst, data)


def _sample_bits_biased(shape, p1: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform {0,1} bits with P(bit=1) = p1."""
    return (rng.random(size=shape) < p1).astype(np.int64)


def _sample_instructions(
    p_w_per_pos: np.ndarray, p_r_per_pos: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample instructions (B, n_inst) given per-position (p_w, p_r) arrays."""
    u = rng.random(size=p_w_per_pos.shape)
    inst = np.where(
        u < p_w_per_pos,
        W,
        np.where(u < p_w_per_pos + p_r_per_pos, R, I),
    ).astype(np.int64)
    return inst


# ---------------------------------------------------------------------------
# Reference families (verbatim from flip_flop) — subsumption targets
# ---------------------------------------------------------------------------
@dataclass
class Stationary(FFLDistribution):
    """iid instructions with independent (p_w, p_r) and bit bias."""
    T: int = 512
    p_w: float = 0.1
    p_r: float = 0.1
    bit_p1: float = 0.5
    name: str = "stationary"

    def __post_init__(self):
        assert self.T % 2 == 0 and self.T >= 4
        assert 0.0 <= self.p_w and 0.0 <= self.p_r and self.p_w + self.p_r <= 1.0, (
            f"invalid (p_w, p_r) = ({self.p_w}, {self.p_r})"
        )
        assert 0.0 <= self.bit_p1 <= 1.0

    def _inst_probs(self, B: int):
        n_inst = self.T // 2
        p_w = np.full((B, n_inst), self.p_w)
        p_r = np.full((B, n_inst), self.p_r)
        return p_w, p_r

    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        p_w, p_r = self._inst_probs(B)
        inst = _sample_instructions(p_w, p_r, rng)
        data = _sample_bits_biased((B, n_inst), self.bit_p1, rng)
        return self._finalize(inst, data)

    def to_dict(self):
        return {"name": "stationary", "T": self.T, "p_w": self.p_w, "p_r": self.p_r,
                "bit_p1": self.bit_p1}

    @classmethod
    def _from_dict(cls, d):
        return cls(T=d["T"], p_w=d["p_w"], p_r=d["p_r"], bit_p1=d.get("bit_p1", 0.5))


@dataclass
class BitMarkov(Stationary):
    """Data bits from a two-state Markov chain. `bit_stay` = P(next == prev)."""
    bit_stay: float = 0.5
    name: str = "bit_markov"

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.bit_stay <= 1.0

    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        p_w, p_r = self._inst_probs(B)
        inst = _sample_instructions(p_w, p_r, rng)

        data = np.empty((B, n_inst), dtype=np.int64)
        data[:, 0] = (rng.random(B) < self.bit_p1).astype(np.int64)
        for k in range(1, n_inst):
            stay = rng.random(B) < self.bit_stay
            data[:, k] = np.where(stay, data[:, k - 1], 1 - data[:, k - 1])
        return self._finalize(inst, data)

    def to_dict(self):
        d = super().to_dict()
        d.update(name="bit_markov", bit_stay=self.bit_stay)
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(T=d["T"], p_w=d["p_w"], p_r=d["p_r"],
                   bit_p1=d.get("bit_p1", 0.5), bit_stay=d.get("bit_stay", 0.5))


@dataclass
class WriteFlipRate(Stationary):
    """Override each write's bit so P(new_bit != stored_bit) = flip_rate."""
    flip_rate: float = 0.5
    name: str = "write_flip"

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.flip_rate <= 1.0

    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        p_w, p_r = self._inst_probs(B)
        inst = _sample_instructions(p_w, p_r, rng)
        data = _sample_bits_biased((B, n_inst), self.bit_p1, rng)

        stored = data[:, 0].copy()
        flips = rng.random((B, n_inst)) < self.flip_rate
        for k in range(1, n_inst):
            is_w = inst[:, k] == W
            new_bit = np.where(flips[:, k], 1 - stored, stored)
            data[:, k] = np.where(is_w, new_bit, data[:, k])
            stored = np.where(is_w, new_bit, stored)
        return self._finalize(inst, data)

    def to_dict(self):
        d = super().to_dict()
        d.update(name="write_flip", flip_rate=self.flip_rate)
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(T=d["T"], p_w=d["p_w"], p_r=d["p_r"],
                   bit_p1=d.get("bit_p1", 0.5), flip_rate=d.get("flip_rate", 0.5))


@dataclass
class Piecewise(FFLDistribution):
    """K segments with independent (p_w, p_r, bit_p1) each.

    `segments` is a list of (start_frac, p_w, p_r, bit_p1) tuples, with
    start_frac in [0, 1). Sorted by start; last segment runs to end.
    """
    T: int = 512
    segments: list = field(default_factory=list)
    name: str = "piecewise"

    def __post_init__(self):
        assert self.T % 2 == 0 and self.T >= 4
        assert len(self.segments) >= 1
        self.segments = sorted(self.segments, key=lambda s: s[0])
        assert self.segments[0][0] == 0.0, "first segment must start at 0"
        for (_, pw, pr, bp1) in self.segments:
            assert -1e-9 <= pw and -1e-9 <= pr and pw + pr <= 1.0 + 1e-9
            assert 0.0 <= bp1 <= 1.0

    def _per_position_params(self):
        n_inst = self.T // 2
        p_w = np.empty(n_inst)
        p_r = np.empty(n_inst)
        b_p1 = np.empty(n_inst)
        starts = [int(round(s[0] * n_inst)) for s in self.segments]
        ends = starts[1:] + [n_inst]
        for (s, e), (_, pw, pr, bp1) in zip(zip(starts, ends), self.segments):
            p_w[s:e] = pw
            p_r[s:e] = pr
            b_p1[s:e] = bp1
        return p_w, p_r, b_p1

    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        p_w_1d, p_r_1d, b_p1_1d = self._per_position_params()
        p_w = np.broadcast_to(p_w_1d, (B, n_inst)).copy()
        p_r = np.broadcast_to(p_r_1d, (B, n_inst)).copy()
        inst = _sample_instructions(p_w, p_r, rng)
        u = rng.random(size=(B, n_inst))
        data = (u < np.broadcast_to(b_p1_1d, (B, n_inst))).astype(np.int64)
        return self._finalize(inst, data)

    def to_dict(self):
        return {"name": "piecewise", "T": self.T,
                "segments": [list(s) for s in self.segments]}

    @classmethod
    def _from_dict(cls, d):
        segs = [tuple(s) for s in d["segments"]]
        return cls(T=d["T"], segments=segs)


# ---------------------------------------------------------------------------
# Numerics (vectorized, stable) — batched over B
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax of (B, C); numerically stable."""
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------
# The contribution: feature-conditioned log-linear controller
# ---------------------------------------------------------------------------
# v1 causal feature set (see docs/PLAN_expressive_adversary.md §1.1). One entry
# per non-constant feature; biases are separate (the "+1" in the param count).
FEATURE_NAMES = (
    "p",              # position fraction k/(n_inst-1)        -> subsumes Piecewise/Periodic
    "p2",             # p**2                                  -> curved position trends
    "gap_norm",       # slots since last emitted W, capped    -> subsumes gap-shape axis
    "nwrites_norm",   # # emitted W so far, capped            -> subsumes churn axis
    "runlen_norm",    # current consecutive-R run, capped     -> subsumes read-run/maintenance
    "last_write_bit", # bit at most-recent emitted W (stored) -> subsumes WriteFlipRate
    "prev_data_bit",  # bit emitted at slot k-1               -> subsumes BitMarkov
)
D_FEATURES = len(FEATURE_NAMES)            # 7
N_PARAMS = 3 * (D_FEATURES + 1)            # wR,cR | wI,cI | wb,cb  -> 24


class FeatureControlledFFL(FFLDistribution):
    """Causal log-linear controller over the FFL instruction/free-bit process.

    At every instruction slot k it forms a bounded causal feature vector
    phi_k (FEATURE_NAMES), then:

      instruction:  z_W = 0 ; z_R = wR·phi + cR ; z_I = wI·phi + cI
                    q   = softmax([z_W, z_R, z_I])                  # (B,3) W,R,I
                    p   = floor + (1 - sum(floor)) * q              # affine floor map
      free bit:     p_bit1 = sigmoid(wb·phi + cb)

    The affine floor map makes `p_w >= p_w_min` and `p_r >= p_r_min` hold at
    EVERY position by construction — read/write starvation (the `p_r->0`
    degeneracy that turned bit_markov into an edge case) is unreachable by the
    parameterization, not merely discouraged by the objective. This is the
    plan's `pi = (1-eps)*softmax + eps*pi_floor` written in its exact-bound
    affine form (equivalent with eps = sum(floor), pi_floor = floor/sum(floor)).

    Read-position bits are still overwritten by `_finalize`/
    `enforce_read_determinism`, so every sample is valid FFL and the exact
    register automaton (1-layer LSTM skyline) stays at 0% for ANY weights.

    Feature caps are stored as fractions of n_inst so the controller is
    T-agnostic and round-trips exactly.
    """

    name = "feature_controlled"

    def __init__(
        self,
        T: int = 512,
        *,
        wR: np.ndarray | list | None = None,
        cR: float = 0.0,
        wI: np.ndarray | list | None = None,
        cI: float = 0.0,
        wb: np.ndarray | list | None = None,
        cb: float = 0.0,
        p_w_min: float = 0.02,
        p_r_min: float = 0.02,
        p_i_min: float = 0.0,
        gap_cap_frac: float = 0.5,
        nwrites_cap_frac: float = 0.5,
        runlen_cap_frac: float = 0.25,
    ):
        assert T % 2 == 0 and T >= 4
        d = D_FEATURES
        self.T = T
        self.wR = np.zeros(d) if wR is None else np.asarray(wR, dtype=float).reshape(d)
        self.wI = np.zeros(d) if wI is None else np.asarray(wI, dtype=float).reshape(d)
        self.wb = np.zeros(d) if wb is None else np.asarray(wb, dtype=float).reshape(d)
        self.cR = float(cR)
        self.cI = float(cI)
        self.cb = float(cb)
        # Floors define an affine map onto the simplex; must leave headroom.
        assert 0.0 <= p_w_min and 0.0 <= p_r_min and 0.0 <= p_i_min
        S = p_w_min + p_r_min + p_i_min
        assert S < 1.0, f"instruction floors must sum < 1 (got {S})"
        self.p_w_min = float(p_w_min)
        self.p_r_min = float(p_r_min)
        self.p_i_min = float(p_i_min)
        assert 0.0 < gap_cap_frac and 0.0 < nwrites_cap_frac and 0.0 < runlen_cap_frac
        self.gap_cap_frac = float(gap_cap_frac)
        self.nwrites_cap_frac = float(nwrites_cap_frac)
        self.runlen_cap_frac = float(runlen_cap_frac)

    # ---- caps (in instruction-slot units) --------------------------------
    def _caps(self, n_inst: int):
        G = max(1.0, self.gap_cap_frac * n_inst)
        Wm = max(1.0, self.nwrites_cap_frac * n_inst)
        Rm = max(1.0, self.runlen_cap_frac * n_inst)
        return G, Wm, Rm

    @property
    def _floor(self) -> np.ndarray:
        return np.array([self.p_w_min, self.p_r_min, self.p_i_min], dtype=float)

    # ---- core map: features -> (instruction probs, bit prob) -------------
    def instruction_probs(self, phi: np.ndarray) -> np.ndarray:
        """phi: (B, d) -> (B, 3) instruction probabilities in (W, R, I) order.

        Guaranteed elementwise >= floor and row-sums == 1. Pure function (no
        sampling) — used by the sampler and directly unit-tested.
        """
        B = phi.shape[0]
        zW = np.zeros(B)
        zR = phi @ self.wR + self.cR
        zI = phi @ self.wI + self.cI
        q = _softmax_rows(np.stack([zW, zR, zI], axis=1))      # (B,3)
        floor = self._floor
        return floor[None, :] + (1.0 - floor.sum()) * q

    def bit_prob(self, phi: np.ndarray) -> np.ndarray:
        """phi: (B, d) -> (B,) P(free bit == 1)."""
        return _sigmoid(phi @ self.wb + self.cb)

    # ---- causal sampling loop (vectorized over batch) --------------------
    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        d = D_FEATURES
        G, Wm, Rm = self._caps(n_inst)
        denom_p = max(1, n_inst - 1)

        inst = np.empty((B, n_inst), dtype=np.int64)
        data = np.empty((B, n_inst), dtype=np.int64)

        since_w = np.zeros(B)            # slots since last emitted W
        has_w = np.zeros(B, dtype=bool)
        n_w = np.zeros(B)               # # emitted W so far
        read_run = np.zeros(B)          # current consecutive-R run length
        last_bit = np.zeros(B)          # bit at most-recent emitted W
        prev_bit = np.zeros(B)          # bit emitted at slot k-1

        phi = np.empty((B, d))
        for k in range(n_inst):
            p = k / denom_p
            phi[:, 0] = p
            phi[:, 1] = p * p
            phi[:, 2] = np.where(has_w, np.minimum(since_w, G) / G, 0.0)
            phi[:, 3] = np.minimum(n_w, Wm) / Wm
            phi[:, 4] = np.minimum(read_run, Rm) / Rm
            phi[:, 5] = last_bit
            phi[:, 6] = prev_bit

            p_inst = self.instruction_probs(phi)               # (B,3) W,R,I
            u = rng.random(B)
            cum_w = p_inst[:, 0]
            cum_wr = cum_w + p_inst[:, 1]
            ins = np.where(u < cum_w, W, np.where(u < cum_wr, R, I)).astype(np.int64)

            pb1 = self.bit_prob(phi)
            bit = (rng.random(B) < pb1).astype(np.int64)

            inst[:, k] = ins
            data[:, k] = bit

            # Causal state update (AFTER the features that fed this step).
            is_w = ins == W
            is_r = ins == R
            since_w = np.where(is_w, 0.0, since_w + 1.0)
            has_w = has_w | is_w
            n_w = n_w + is_w
            last_bit = np.where(is_w, bit.astype(float), last_bit)
            read_run = np.where(is_r, read_run + 1.0, 0.0)
            prev_bit = bit.astype(float)

        return self._finalize(inst, data)

    # ---- analytic position schedule (no sampling) -----------------------
    def analytic_position_inst_probs(self, n_inst: int) -> np.ndarray:
        """Per-slot (n_inst, 3) instruction probs when ONLY position features
        (p, p2) are active (gap/nwrites/runlen/bits weighted 0). Deterministic;
        used to build an equivalent Piecewise for the position-subsumption
        witness."""
        denom_p = max(1, n_inst - 1)
        ks = np.arange(n_inst)
        phi = np.zeros((n_inst, D_FEATURES))
        phi[:, 0] = ks / denom_p
        phi[:, 1] = (ks / denom_p) ** 2
        return self.instruction_probs(phi)

    # ---- serialization ---------------------------------------------------
    def to_dict(self):
        return {
            "name": self.name, "T": self.T,
            "wR": self.wR.tolist(), "cR": self.cR,
            "wI": self.wI.tolist(), "cI": self.cI,
            "wb": self.wb.tolist(), "cb": self.cb,
            "p_w_min": self.p_w_min, "p_r_min": self.p_r_min,
            "p_i_min": self.p_i_min,
            "gap_cap_frac": self.gap_cap_frac,
            "nwrites_cap_frac": self.nwrites_cap_frac,
            "runlen_cap_frac": self.runlen_cap_frac,
            "feature_names": list(FEATURE_NAMES),
        }

    @classmethod
    def _from_dict(cls, d):
        return cls(
            T=d["T"],
            wR=d["wR"], cR=d["cR"], wI=d["wI"], cI=d["cI"],
            wb=d["wb"], cb=d["cb"],
            p_w_min=d.get("p_w_min", 0.02), p_r_min=d.get("p_r_min", 0.02),
            p_i_min=d.get("p_i_min", 0.0),
            gap_cap_frac=d.get("gap_cap_frac", 0.5),
            nwrites_cap_frac=d.get("nwrites_cap_frac", 0.5),
            runlen_cap_frac=d.get("runlen_cap_frac", 0.25),
        )

    def descriptor(self):
        """Compact log summary: which feature groups the optimizer activated
        (this is the 'read off what it learned' interpretability surface)."""
        def by_feat(w):
            return {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w)}
        return {
            "name": self.name, "T": self.T, "n_params": N_PARAMS,
            "floor": [self.p_w_min, self.p_r_min, self.p_i_min],
            "caps_frac": [self.gap_cap_frac, self.nwrites_cap_frac,
                          self.runlen_cap_frac],
            "wR_norm": round(float(np.linalg.norm(self.wR)), 4), "cR": round(self.cR, 4),
            "wI_norm": round(float(np.linalg.norm(self.wI)), 4), "cI": round(self.cI, 4),
            "wb_norm": round(float(np.linalg.norm(self.wb)), 4), "cb": round(self.cb, 4),
            "wR": by_feat(self.wR), "wI": by_feat(self.wI), "wb": by_feat(self.wb),
        }

    # =====================================================================
    # Subsumption witnesses — executable proof that the controller class is a
    # strict superset of the hand-coded families (PLAN §2 claim 2, Phase-0).
    # Each returns a FeatureControlledFFL whose induced distribution equals the
    # named family (exactly, or — for position schedules — up to basis order).
    # =====================================================================
    @classmethod
    def emulate_stationary(cls, T: int, p_w: float, p_r: float,
                           bit_p1: float = 0.5, **caps) -> "FeatureControlledFFL":
        """EXACT: all feature weights 0; biases solve the instruction simplex
        and the constant bit bias. Requires p_w,p_r,p_i within the floors."""
        d = cls(T=T, **caps)
        p_i = 1.0 - p_w - p_r
        fl = d._floor
        S = fl.sum()
        # p = floor + (1-S) q  =>  q = (p - floor)/(1-S)  (must be a valid simplex)
        q = (np.array([p_w, p_r, p_i]) - fl) / (1.0 - S)
        assert (q > 1e-9).all() and abs(q.sum() - 1.0) < 1e-9, (
            f"({p_w},{p_r}) not representable above floors {fl.tolist()}"
        )
        # z_W = 0  =>  q_W ∝ 1, q_R ∝ e^{cR}, q_I ∝ e^{cI}
        d.cR = float(np.log(q[1] / q[0]))
        d.cI = float(np.log(q[2] / q[0]))
        d.cb = _logit(bit_p1)
        return d

    @classmethod
    def emulate_write_flip(cls, T: int, p_w: float, p_r: float,
                           flip_rate: float, **caps) -> "FeatureControlledFFL":
        """EXACT (write-bit law): Stationary instructions; the bit head depends
        only on `last_write_bit`, reproducing P(new write bit != stored) =
        flip_rate. Filler bits at I-slots are unsupervised; read bits are
        overwritten — so the observable distribution matches WriteFlipRate."""
        d = cls.emulate_stationary(T, p_w, p_r, bit_p1=flip_rate, **caps)
        # P(bit=1 | last_write_bit=0) = flip_rate ; | =1) = 1-flip_rate
        c0 = _logit(flip_rate)               # last_write_bit = 0
        c1 = _logit(1.0 - flip_rate)         # last_write_bit = 1
        d.wb = np.zeros(D_FEATURES)
        d.wb[5] = c1 - c0                     # feature index 5 == last_write_bit
        d.cb = c0
        return d

    @classmethod
    def emulate_bit_markov(cls, T: int, p_w: float, p_r: float,
                           bit_stay: float, bit_p1: float = 0.5,
                           **caps) -> "FeatureControlledFFL":
        """EXACT (bit transition law): Stationary instructions; the bit head
        depends only on `prev_data_bit`, reproducing P(bit == prev) = bit_stay.
        The single first-slot marginal differs by O(1/n_inst); the conserved
        quantity (consecutive-same-bit rate) is matched exactly."""
        d = cls.emulate_stationary(T, p_w, p_r, bit_p1=bit_p1, **caps)
        c0 = _logit(1.0 - bit_stay)          # prev = 0 -> P(1) = 1 - bit_stay
        c1 = _logit(bit_stay)                # prev = 1 -> P(1) = bit_stay
        d.wb = np.zeros(D_FEATURES)
        d.wb[6] = c1 - c0                     # feature index 6 == prev_data_bit
        d.cb = c0
        return d

    @classmethod
    def emulate_piecewise_ramp(cls, T: int, pw_start: float, pw_end: float,
                               p_r: float, bit_p1: float = 0.5,
                               **caps):
        """APPROXIMATE (exact in the position-basis limit): a position-only
        controller whose write-rate trends from ~pw_start to ~pw_end across the
        sequence, plus the equivalent K-segment `Piecewise` built from the
        controller's OWN analytic per-slot probabilities. Demonstrates that any
        position schedule the basis expresses is reproducible as a Piecewise
        and vice-versa (position conditioning is in-class).

        Returns (controller, equivalent_piecewise).
        """
        d = cls(T=T, **caps)
        n_inst = T // 2
        # Choose a linear-in-p logit so the realized p_w sweeps start->end.
        # Keep p_r ~ constant by giving R the same position slope as the W
        # reference (z_W = 0): make z_R, z_I linear in p only.
        fl = d._floor
        free = 1.0 - fl.sum()

        def q_of_logits(zr, zi):
            e = np.array([1.0, np.exp(zr), np.exp(zi)])
            return e / e.sum()

        # Target simplex at p=0 and p=1 (above floors), constant p_r.
        def targets(pw):
            pi = 1.0 - pw - p_r
            return (np.array([pw, p_r, pi]) - fl) / free

        q0, q1 = targets(pw_start), targets(pw_end)
        zr0, zi0 = np.log(q0[1] / q0[0]), np.log(q0[2] / q0[0])
        zr1, zi1 = np.log(q1[1] / q1[0]), np.log(q1[2] / q1[0])
        d.cR, d.cI = float(zr0), float(zi0)
        d.wR = np.zeros(D_FEATURES); d.wR[0] = float(zr1 - zr0)   # slope on p
        d.wI = np.zeros(D_FEATURES); d.wI[0] = float(zi1 - zi0)
        d.cb = _logit(bit_p1)

        probs = d.analytic_position_inst_probs(n_inst)            # (n_inst,3)
        K = min(64, n_inst)
        segs = []
        for s in range(K):
            lo = int(round(s * n_inst / K))
            mid = min(n_inst - 1, lo + (n_inst // K) // 2)
            pw, pr, _ = probs[mid]
            segs.append((s / K, float(pw), float(pr), float(bit_p1)))
        equiv = Piecewise(T=T, segments=segs)
        return d, equiv


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type[FFLDistribution]] = {
    "stationary": Stationary,
    "bit_markov": BitMarkov,
    "write_flip": WriteFlipRate,
    "piecewise": Piecewise,
    "feature_controlled": FeatureControlledFFL,
}


def build(name: str, **kwargs: Any) -> FFLDistribution:
    """Instantiate a distribution by registry name."""
    if name not in REGISTRY:
        raise ValueError(f"unknown distribution {name!r}; known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
