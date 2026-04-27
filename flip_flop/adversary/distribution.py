"""FFL distribution samplers for the adversary.

Every sampler produces valid FFL strings (alternation, x_0 = w, x_{T-2} = r,
read-determinism). Enforcement is centralized in `enforce_read_determinism`,
so new schedules need only specify (a) how instructions are drawn per position
and (b) how raw data bits are drawn.

Axis coverage:
  A — Stationary (independent p_w, p_r; A1/A2/A3)
  B — Stationary + bit_p1 (B1); BitMarkov (B2); WriteFlipRate (B3)
  C — Piecewise (C1/C2/C4); Periodic (C3)
  D — Planted (D1/D2/D3/D4)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..data import W, R, I, enforce_read_determinism, interleave


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

    # Helper used by all samplers.
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
    """Sample instructions (B, n_inst) given per-position (p_w, p_r) arrays.

    p_i per position = 1 - p_w - p_r. Uses vectorized inverse-CDF.
    """
    u = rng.random(size=p_w_per_pos.shape)
    inst = np.where(
        u < p_w_per_pos,
        W,
        np.where(u < p_w_per_pos + p_r_per_pos, R, I),
    ).astype(np.int64)
    return inst


# ---------------------------------------------------------------------------
# A, B — stationary and bit-level variants
# ---------------------------------------------------------------------------
@dataclass
class Stationary(FFLDistribution):
    """Axis A1/A2/A3 + B1: iid instructions with independent (p_w, p_r) and bit bias."""
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
    """Axis B2: data bits drawn from a two-state Markov chain.

    `bit_stay` = P(next bit == previous bit). 0.5 recovers iid.
    First bit is drawn from Bernoulli(bit_p1).
    """
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
    """Axis B3: override each write's bit so P(new_bit != stored_bit) = flip_rate.

    Tests whether the model tracks state or tracks "last data token seen".
    flip_rate = 0.5 recovers iid-uniform-writes; flip_rate = 0.0 means all
    writes repeat the current stored bit.
    """
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

        # First instruction is forced to W by _finalize; seed stored bit.
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


# ---------------------------------------------------------------------------
# C — non-stationary schedules
# ---------------------------------------------------------------------------
@dataclass
class Piecewise(FFLDistribution):
    """Axis C1/C2/C4: K segments with independent (p_w, p_r, bit_p1) each.

    `segments` is a list of (start_frac, p_w, p_r, bit_p1) tuples, with
    start_frac in [0, 1). Sorted by start; last segment runs to end.
    """
    T: int = 512
    segments: list[tuple[float, float, float, float]] = field(default_factory=list)
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


@dataclass
class Periodic(FFLDistribution):
    """Axis C3: period-L schedule over (p_w_t, p_r_t, bit_p1_t).

    pattern has length L, each entry a (p_w, p_r, bit_p1) triple.
    Per instruction position k: params = pattern[k mod L].
    """
    T: int = 512
    period: int = 8
    pattern: list[tuple[float, float, float]] = field(default_factory=list)
    name: str = "periodic"

    def __post_init__(self):
        assert self.T % 2 == 0 and self.T >= 4
        assert self.period >= 1
        assert len(self.pattern) == self.period
        for (pw, pr, bp1) in self.pattern:
            assert -1e-9 <= pw and -1e-9 <= pr and pw + pr <= 1.0 + 1e-9
            assert 0.0 <= bp1 <= 1.0

    def sample(self, batch_size, rng):
        B = batch_size
        n_inst = self.T // 2
        pw_arr = np.array([p[0] for p in self.pattern])
        pr_arr = np.array([p[1] for p in self.pattern])
        bp1_arr = np.array([p[2] for p in self.pattern])
        idx = np.arange(n_inst) % self.period
        p_w_1d = pw_arr[idx]
        p_r_1d = pr_arr[idx]
        b_p1_1d = bp1_arr[idx]
        p_w = np.broadcast_to(p_w_1d, (B, n_inst)).copy()
        p_r = np.broadcast_to(p_r_1d, (B, n_inst)).copy()
        inst = _sample_instructions(p_w, p_r, rng)
        u = rng.random(size=(B, n_inst))
        data = (u < np.broadcast_to(b_p1_1d, (B, n_inst))).astype(np.int64)
        return self._finalize(inst, data)

    def to_dict(self):
        return {"name": "periodic", "T": self.T, "period": self.period,
                "pattern": [list(p) for p in self.pattern]}

    @classmethod
    def _from_dict(cls, d):
        pat = [tuple(p) for p in d["pattern"]]
        return cls(T=d["T"], period=d["period"], pattern=pat)


# ---------------------------------------------------------------------------
# D — planted-pattern constructions
# ---------------------------------------------------------------------------
@dataclass
class Planted(FFLDistribution):
    """Axis D: hand-crafted templates with a few controllable parameters.

    Templates (indices are over instruction positions k in [0, n_inst)):
      - "gap"        (D1): single W at k=k_write, R at k=n_inst-1, ignores elsewhere.
                           params: {"k_write": int}
      - "decoy"      (D2): decoy W at k_early (bit b_decoy), true W at k_true
                           (bit 1-b_decoy), R at k=n_inst-1, ignores elsewhere.
                           params: {"k_early": int, "k_true": int, "b_decoy": int}
      - "distractor" (D3): true W at k_true (bit b_true), late distractor W at
                           k=n_inst-1-d with opposite bit, R at n_inst-1.
                           params: {"k_true": int, "d": int, "b_true": int}
      - "disagree"   (D4): N_w writes at evenly-spaced positions; a fraction
                           frac_agree of them carry the bit that matches the
                           most-recent write; R at n_inst-1, ignores elsewhere.
                           params: {"N_w": int, "frac_agree": float,
                                    "b_last": int}

    Fillers between planted tokens are forced I (deterministic). `filler_p_i`
    is not used in the current templates but kept for parity with the plan and
    future noisy-filler variants.
    """
    T: int = 512
    template: str = "gap"
    filler_p_i: float = 1.0
    params: dict = field(default_factory=dict)
    name: str = "planted"

    def __post_init__(self):
        assert self.T % 2 == 0 and self.T >= 4
        assert self.template in ("gap", "decoy", "distractor", "disagree")

    def sample(self, batch_size, rng):
        n_inst = self.T // 2
        build = {
            "gap": self._build_gap,
            "decoy": self._build_decoy,
            "distractor": self._build_distractor,
            "disagree": self._build_disagree,
        }[self.template]
        inst_row, data_row = build(n_inst, rng)
        # Broadcast the template across the batch; templates are deterministic
        # up to their stated random choices (e.g. filler bits), which are drawn
        # per-sample below.
        B = batch_size
        inst = np.broadcast_to(inst_row, (B, n_inst)).copy()
        data = np.broadcast_to(data_row, (B, n_inst)).copy()
        # Randomize filler data bits (only affects W/I positions; reads are
        # overwritten by enforce_read_determinism).
        filler_bits = rng.integers(0, 2, size=(B, n_inst), dtype=np.int64)
        is_filler = data_row < 0  # sentinel in template rows
        mask = np.broadcast_to(is_filler, (B, n_inst))
        data = np.where(mask, filler_bits, data)
        return self._finalize(inst, data)

    def _blank(self, n_inst):
        inst = np.full(n_inst, I, dtype=np.int64)
        inst[-1] = R
        # sentinel -1 means "filler bit, randomize per sample"
        data = np.full(n_inst, -1, dtype=np.int64)
        return inst, data

    def _build_gap(self, n_inst, rng):
        k_write = int(self.params.get("k_write", 0))
        assert 0 <= k_write < n_inst - 1, f"k_write {k_write} out of range"
        inst, data = self._blank(n_inst)
        inst[k_write] = W
        # bit at the write position is a filler (random per-sample); read at
        # n_inst-1 will be overwritten to match it.
        return inst, data

    def _build_decoy(self, n_inst, rng):
        k_early = int(self.params.get("k_early", 1))
        k_true = int(self.params.get("k_true", n_inst - 3))
        b_decoy = int(self.params.get("b_decoy", 0)) & 1
        assert 0 <= k_early < k_true < n_inst - 1
        inst, data = self._blank(n_inst)
        inst[k_early] = W
        data[k_early] = b_decoy
        inst[k_true] = W
        data[k_true] = 1 - b_decoy
        return inst, data

    def _build_distractor(self, n_inst, rng):
        k_true = int(self.params.get("k_true", 1))
        d = int(self.params.get("d", 1))  # distractor at n_inst-1-d
        b_true = int(self.params.get("b_true", 0)) & 1
        k_distractor = n_inst - 1 - d
        assert 0 <= k_true < k_distractor < n_inst - 1
        inst, data = self._blank(n_inst)
        inst[k_true] = W
        data[k_true] = b_true
        inst[k_distractor] = W
        data[k_distractor] = 1 - b_true
        return inst, data

    def _build_disagree(self, n_inst, rng):
        N_w = int(self.params.get("N_w", 3))
        frac_agree = float(self.params.get("frac_agree", 0.5))
        b_last = int(self.params.get("b_last", 0)) & 1
        assert N_w >= 1
        # Evenly space writes in [0, n_inst-1), last write at latest feasible slot.
        if N_w == 1:
            positions = [n_inst - 3]
        else:
            positions = list(np.linspace(0, n_inst - 2, N_w, dtype=int))
        # Force strictly increasing positions and stop before the read.
        positions = sorted(set(int(p) for p in positions if p < n_inst - 1))
        assert len(positions) == N_w, f"position collision: {positions}"
        inst, data = self._blank(n_inst)
        # The most-recent write is the last position; it must carry b_last.
        # Earlier writes agree with b_last with prob frac_agree, else flip.
        for idx, k in enumerate(positions):
            inst[k] = W
            if idx == len(positions) - 1:
                data[k] = b_last
            else:
                # frac_agree handled deterministically per-position to keep
                # templates reproducible: agree if rank / (N_w-1) < frac_agree.
                rank = idx / max(1, (N_w - 1))
                data[k] = b_last if rank < frac_agree else (1 - b_last)
        return inst, data

    def to_dict(self):
        return {"name": "planted", "T": self.T, "template": self.template,
                "filler_p_i": self.filler_p_i, "params": dict(self.params)}

    @classmethod
    def _from_dict(cls, d):
        return cls(T=d["T"], template=d["template"],
                   filler_p_i=d.get("filler_p_i", 1.0),
                   params=dict(d.get("params", {})))


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type[FFLDistribution]] = {
    "stationary": Stationary,
    "bit_markov": BitMarkov,
    "write_flip": WriteFlipRate,
    "piecewise": Piecewise,
    "periodic": Periodic,
    "planted": Planted,
}


def build(name: str, **kwargs: Any) -> FFLDistribution:
    """Instantiate a distribution by registry name."""
    if name not in REGISTRY:
        raise ValueError(f"unknown distribution {name!r}; known: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
