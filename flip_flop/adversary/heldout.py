"""PLAN_beat_liu_r4 Sec.4 -- pre-registered held-out evaluation configs.

FROZEN and committed to git BEFORE Rung 2. Neither config is ever fed to the
adversary objective; they exist only to measure generalization. Changing these
after seeing results invalidates the pre-registration -- treat as read-only.

  HELDOUT_ANTICORR  (Sec.4 (ii)) -- the axis the STRETCH goal targets. A fixed
      interior-gap AntiCorrelatedBits config at full anti-correlation. CMA
      never sees these exact params (rho pinned to 1.0, (p_w,p_r) fixed), so a
      non-trivial B_liu error here with LSTM ~0 is genuine residual error.

  HELDOUT_PIECEWISE (Sec.4 (i)) -- the paper's segment-3 drift signature: a
      long, very-sparse-write early span then a dense-read tail, so most reads
      must recall a write from far back (long-range state tracking). The
      state-tracking LSTM stays ~0; an attention-glitching Transformer drifts.
"""
from __future__ import annotations

T_DEFAULT = 512

HELDOUT_ANTICORR: dict = {
    "name": "anti_correlated_bits", "T": T_DEFAULT,
    "p_w": 0.05, "p_r": 0.05, "bit_p1": 0.5, "rho": 1.0, "k_hist": 3,
}

HELDOUT_PIECEWISE: dict = {
    "name": "piecewise", "T": T_DEFAULT,
    "segments": [
        [0.0, 0.02, 0.0, 0.5],     # seg 1: rare writes, no reads (plant state)
        [0.25, 0.02, 0.02, 0.5],   # seg 2: still sparse
        [0.75, 0.0, 0.45, 0.5],    # seg 3: no writes, dense reads (drift zone)
    ],
}

# Liu's three frozen distributions (training data for B_liu / full_adv).
LIU_TAILS: dict = {
    "ffl_080": {"name": "stationary", "T": T_DEFAULT, "p_w": 0.1,  "p_r": 0.1,  "bit_p1": 0.5},
    "ffl_098": {"name": "stationary", "T": T_DEFAULT, "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.5},
    "ffl_010": {"name": "stationary", "T": T_DEFAULT, "p_w": 0.45, "p_r": 0.45, "bit_p1": 0.5},
}


def piecewise_band(T: int = T_DEFAULT, n: int = 16, seed: int = 20260516) -> dict[str, dict]:
    """PLAN Sec.2.C -- pre-registered robustness band around HELDOUT_PIECEWISE.

    16 valid piecewise configs on a FIXED seeded grid: the two interior segment
    boundaries jittered +/-0.08 and seg-3 read-rate jittered +/-0.10, clamped to
    a legal FFL piecewise (0 < b1 < b2 < 1, p_w+p_r<=1). Makes the beat claim
    region-level (the diagnostic already showed 59/60 region robustness; this
    freezes it). FROZEN -- never fed to any adversary. Changing this after
    seeing results invalidates the pre-registration.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for i in range(n):
        b1 = float(np.clip(0.25 + rng.uniform(-0.08, 0.08), 0.10, 0.40))
        b2 = float(np.clip(0.75 + rng.uniform(-0.08, 0.08), b1 + 0.15, 0.92))
        pr3 = float(np.clip(0.45 + rng.uniform(-0.10, 0.10), 0.20, 0.60))
        pw1 = float(np.clip(0.02 + rng.uniform(-0.01, 0.01), 0.005, 0.05))
        out[f"piecewise_band_{i:02d}"] = {
            "name": "piecewise", "T": T,
            "segments": [
                [0.0, pw1, 0.0, 0.5],
                [b1, pw1, 0.02, 0.5],
                [b2, 0.0, pr3, 0.5],
            ],
        }
    return out


def heldout_set(T: int = T_DEFAULT) -> dict[str, dict]:
    """Name -> config for the per-round loop gate (Sec.3 step 4). Liu tails are
    included so the round model is always measured against the R4 endpoints."""
    def _T(cfg: dict) -> dict:
        c = dict(cfg)
        c["T"] = T
        return c
    return {
        "heldout_anticorr": _T(HELDOUT_ANTICORR),
        "heldout_piecewise": _T(HELDOUT_PIECEWISE),
        "liu_ffl_098": _T(LIU_TAILS["ffl_098"]),
        "liu_ffl_010": _T(LIU_TAILS["ffl_010"]),
    }
