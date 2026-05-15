"""Family extraction from adversary logs.

Given a JSONL log of adversary evaluations, build K samplers (Family
instances) for retrain.

Stationary / Piecewise: dedup configs by 2-decimal-place params, keep the
best-fitness record per bin, take top-K by T_glitch; for each, α-bisect
the parameter interpolation between base FFL(0.8) and the adversary
config -> ClusterFamily.

Planted: group records by template, take the best-fitness record per
template; for templates with a bit parameter, add the bit-flipped twin;
α-bisect a mixture-of-generators between base and the variant set ->
MixtureFamily.

The α pull-back enforces:
  - LSTM_glitch(p_α) < max_lstm_glitch  (skyline still clean)
  - T_glitch(p_α)  ~ target_t_glitch    (Transformer still failing)
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .distribution import FFLDistribution, Piecewise, Stationary


# ---------------------------------------------------------------------------
# Family classes
# ---------------------------------------------------------------------------
class Family(abc.ABC):
    name: str

    @abc.abstractmethod
    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        ...

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": type(self).__name__}


@dataclass
class PassthroughFamily(Family):
    """Wrap one FFLDistribution as a Family. Used by tests."""
    dist: FFLDistribution
    name: str = "passthrough"

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {"name": self.name, "kind": "PassthroughFamily",
                "dist": self.dist.to_dict()}


@dataclass
class ClusterFamily(Family):
    """α-pulled-back distribution between base and an adversary config.

    At α=0 the distribution equals base; at α=1 it equals the adv config;
    intermediate α gives convex parameter combinations.
    """
    dist: FFLDistribution
    alpha: float
    cluster_mean_glitch: float
    rep_config: dict
    name: str = "cluster"

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": "ClusterFamily",
            "alpha": self.alpha,
            "cluster_mean_glitch": self.cluster_mean_glitch,
            "rep_config": self.rep_config,
            "dist": self.dist.to_dict(),
        }


@dataclass
class MixtureFamily(Family):
    """Per-sequence mixture between base and a list of adv distributions.

    Used for planted templates whose discrete params don't admit parameter
    interpolation. With prob α each sequence is drawn uniformly from
    `adv_dists`; else from `base_dist`. Including bit-flipped twins in
    `adv_dists` prevents the model from memorising a bit-bias on the
    symmetric attack instead of learning state tracking.
    """
    base_dist: FFLDistribution
    adv_dists: list[FFLDistribution]
    alpha: float
    cluster_mean_glitch: float = 0.0
    rep_config: dict = field(default_factory=dict)
    name: str = "mixture"

    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0, f"alpha out of [0,1]: {self.alpha}"
        assert len(self.adv_dists) >= 1, "MixtureFamily needs >=1 adv_dist"
        for d in self.adv_dists:
            assert self.base_dist.T == d.T, (
                f"T mismatch: base={self.base_dist.T}, adv={d.T}"
            )

    @property
    def adv_dist(self) -> FFLDistribution:
        """First (representative) adv variant."""
        return self.adv_dists[0]

    def sample(self, batch_size, rng):
        T = self.base_dist.T
        is_adv = rng.random(batch_size) < self.alpha
        n_adv = int(is_adv.sum())
        n_base = batch_size - n_adv
        out = torch.empty(batch_size, T, dtype=torch.long)
        if n_base > 0:
            out[~is_adv] = self.base_dist.sample(n_base, rng)
        if n_adv > 0:
            adv_indices = np.where(is_adv)[0]
            variant_per = rng.integers(0, len(self.adv_dists), size=n_adv)
            for v_idx, v_dist in enumerate(self.adv_dists):
                mask = variant_per == v_idx
                n_v = int(mask.sum())
                if n_v == 0:
                    continue
                out[adv_indices[mask]] = v_dist.sample(n_v, rng)
        return out

    def to_dict(self):
        return {
            "name": self.name,
            "kind": "MixtureFamily",
            "alpha": self.alpha,
            "cluster_mean_glitch": self.cluster_mean_glitch,
            "rep_config": self.rep_config,
            "base_dist": self.base_dist.to_dict(),
            "adv_dists": [d.to_dict() for d in self.adv_dists],
            "n_adv_variants": len(self.adv_dists),
        }


# ---------------------------------------------------------------------------
# Parameter interpolation
# ---------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _lift_to_piecewise(base: Stationary, K: int) -> Piecewise:
    segs = [(k / K, base.p_w, base.p_r, base.bit_p1) for k in range(K)]
    return Piecewise(T=base.T, segments=segs)


def interpolate_params(base_cfg: dict, adv_cfg: dict, alpha: float) -> FFLDistribution:
    """Linear param interpolation. Returns Stationary or Piecewise."""
    assert 0.0 <= alpha <= 1.0
    a_name = adv_cfg["name"]

    if a_name == "piecewise":
        if base_cfg["name"] != "piecewise":
            base_p = _lift_to_piecewise(
                FFLDistribution.from_dict(base_cfg), K=len(adv_cfg["segments"])
            ).to_dict()
        else:
            base_p = base_cfg
        base_segs = base_p["segments"]
        adv_segs = adv_cfg["segments"]
        assert len(base_segs) == len(adv_segs)
        new_segs = []
        for bs, as_ in zip(base_segs, adv_segs):
            new_segs.append([
                as_[0],
                _clip01((1 - alpha) * bs[1] + alpha * as_[1]),
                _clip01((1 - alpha) * bs[2] + alpha * as_[2]),
                _clip01((1 - alpha) * bs[3] + alpha * as_[3]),
            ])
        return Piecewise(T=adv_cfg["T"], segments=[tuple(s) for s in new_segs])

    if a_name in ("stationary", "bit_markov", "write_flip"):
        if base_cfg["name"] not in ("stationary", "bit_markov", "write_flip"):
            raise ValueError(f"cannot interpolate {base_cfg['name']} with {a_name}")
        return Stationary(
            T=adv_cfg["T"],
            p_w=_clip01((1 - alpha) * base_cfg["p_w"] + alpha * adv_cfg["p_w"]),
            p_r=_clip01((1 - alpha) * base_cfg["p_r"] + alpha * adv_cfg["p_r"]),
            bit_p1=_clip01((1 - alpha) * base_cfg.get("bit_p1", 0.5)
                            + alpha * adv_cfg.get("bit_p1", 0.5)),
        )

    raise ValueError(f"unsupported type {a_name}")


# ---------------------------------------------------------------------------
# α pull-back
# ---------------------------------------------------------------------------
def _eval_glitch(model, tokens: torch.LongTensor, batch_size: int, device: str) -> float:
    from ..eval import evaluate_dataset
    return evaluate_dataset(model, tokens, batch_size=batch_size, device=device)["error_rate"]


def _bisect_alpha(
    eval_at_alpha,
    *,
    max_lstm_glitch: float = 0.01,
    target_t_glitch: float = 0.5,
    max_iter: int = 8,
    tol: float = 0.02,
) -> tuple[float, float, float]:
    """Bisect α∈[0,1]; returns (α*, T_err@α*, lstm_err@α*).

    Picks the largest α that keeps the LSTM clean and the Transformer near
    target_t_glitch. Endpoint cases:
      - LSTM fails at α=1 -> return α=1.
      - Transformer too easy at α=1 -> return α=1.
      - Transformer too hard at α=0 -> return α=0.
    """
    t1, l1 = eval_at_alpha(1.0)
    t0, l0 = eval_at_alpha(0.0)

    if l1 >= max_lstm_glitch:
        return (1.0, t1, l1)
    if t1 <= target_t_glitch:
        return (1.0, t1, l1)
    if t0 >= target_t_glitch:
        return (0.0, t0, l0)

    lo, hi = 0.0, 1.0
    best = (1.0, t1, l1) if abs(t1 - target_t_glitch) < abs(t0 - target_t_glitch) else (0.0, t0, l0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t_mid, l_mid = eval_at_alpha(mid)
        if l_mid < max_lstm_glitch and abs(t_mid - target_t_glitch) < abs(best[1] - target_t_glitch):
            best = (mid, t_mid, l_mid)
        if abs(t_mid - target_t_glitch) < tol:
            break
        if t_mid < target_t_glitch:
            lo = mid
        else:
            hi = mid
    return best


def pull_back_alpha(
    base_cfg: dict,
    adv_cfg: dict,
    transformer,
    lstm,
    device: str,
    *,
    T: int,
    rng: np.random.Generator,
    n_probe: int = 1000,
    batch_size: int = 64,
    max_lstm_glitch: float = 0.01,
    target_t_glitch: float = 0.5,
    max_iter: int = 8,
    tol: float = 0.02,
    ref_glitch: float = 1.0,  # back-compat with tests; unused
    alphas: tuple = (),       # back-compat; unused
) -> tuple[float, float, float]:
    """Bisect α via parameter interpolation (Stationary / Piecewise)."""
    def eval_alpha(alpha):
        dist = interpolate_params(base_cfg, adv_cfg, alpha)
        tokens = dist.sample(n_probe, rng)
        t = _eval_glitch(transformer, tokens, batch_size, device)
        l = _eval_glitch(lstm, tokens, batch_size, device) if lstm is not None else 0.0
        return t, l
    return _bisect_alpha(eval_alpha, max_lstm_glitch=max_lstm_glitch,
                         target_t_glitch=target_t_glitch, max_iter=max_iter, tol=tol)


def pull_back_alpha_mixture(
    base_dist: FFLDistribution,
    adv_dists,
    transformer,
    lstm,
    device: str,
    *,
    rng: np.random.Generator,
    n_probe: int = 1000,
    batch_size: int = 64,
    max_lstm_glitch: float = 0.01,
    target_t_glitch: float = 0.5,
    max_iter: int = 8,
    tol: float = 0.02,
) -> tuple[float, float, float]:
    """Bisect α via mixture-of-generators (planted templates)."""
    if not isinstance(adv_dists, (list, tuple)):
        adv_dists = [adv_dists]
    def eval_alpha(alpha):
        fam = MixtureFamily(base_dist=base_dist, adv_dists=list(adv_dists), alpha=alpha)
        tokens = fam.sample(n_probe, rng)
        t = _eval_glitch(transformer, tokens, batch_size, device)
        l = _eval_glitch(lstm, tokens, batch_size, device) if lstm is not None else 0.0
        return t, l
    return _bisect_alpha(eval_alpha, max_lstm_glitch=max_lstm_glitch,
                         target_t_glitch=target_t_glitch, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
# Planted bit-flip twin
# ---------------------------------------------------------------------------
def _planted_bit_flip_twins(rep_cfg: dict) -> list[dict]:
    """Return [original, bit-flipped twin] for templates with a bit param.

    Including the twin during retrain prevents the model from memorising a
    bit-bias on the symmetric attack rather than learning state tracking.
    Gap has no bit param -> returns [original] only.
    """
    template = rep_cfg.get("template")
    params = dict(rep_cfg.get("params", {}))
    bit_field = {"decoy": "b_decoy", "distractor": "b_true",
                 "disagree": "b_last"}.get(template)
    if bit_field is None or bit_field not in params:
        return [rep_cfg]
    twin = dict(rep_cfg)
    twin_params = dict(params)
    twin_params[bit_field] = 1 - int(params[bit_field])
    twin["params"] = twin_params
    return [rep_cfg, twin]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _bin_config(cfg: dict, ndp: int = 2) -> tuple:
    """A hashable dedup key: round all floats to `ndp` decimal places."""
    name = cfg["name"]
    if name == "piecewise":
        segs = cfg["segments"]
        return (name, tuple((round(s[0], ndp), round(s[1], ndp),
                             round(s[2], ndp), round(s[3], ndp)) for s in segs))
    return (name, round(cfg.get("p_w", 0.0), ndp),
                  round(cfg.get("p_r", 0.0), ndp),
                  round(cfg.get("bit_p1", 0.5), ndp))


def extract_families_from_adversary_log(
    log_path: str,
    *,
    base_cfg: Optional[dict] = None,
    transformer=None,
    lstm=None,
    device: str = "cpu",
    top_k: int = 5,
    min_t_glitch: float = 0.01,
    max_lstm_glitch: float = 0.01,
    n_behavior: int = 64,  # back-compat with tests; unused
    seed: int = 0,
) -> list[Family]:
    """Build top-K families from an adversary log.

    If transformer/lstm/base_cfg is None, returns PassthroughFamily stubs
    over the top-K records by fitness (test fallback).
    """
    with open(log_path) as f:
        recs = [json.loads(l) for l in f]
    valid = [
        r for r in recs
        if r.get("is_valid", True)
        and r.get("T_glitch", 0.0) > min_t_glitch
        and r.get("lstm_glitch", 1.0) < max_lstm_glitch
    ]
    if not valid:
        return []

    if transformer is None or lstm is None or base_cfg is None:
        valid.sort(key=lambda r: r["fitness"], reverse=True)
        return [
            PassthroughFamily(FFLDistribution.from_dict(r["config"]),
                              name=f"adv_{i:02d}")
            for i, r in enumerate(valid[:top_k])
        ]

    rng = np.random.default_rng(seed)
    base_dist = FFLDistribution.from_dict(base_cfg)
    families: list[Family] = []

    # ---- Planted: one MixtureFamily per template (with bit-flip twin) ----
    planted = [r for r in valid if r["config"]["name"] == "planted"]
    if planted:
        by_template: dict[str, list[dict]] = {}
        for r in planted:
            by_template.setdefault(r["config"]["template"], []).append(r)
        for tmpl, recs_t in by_template.items():
            recs_t.sort(key=lambda r: r["fitness"], reverse=True)
            rep = recs_t[0]
            adv_cfgs = _planted_bit_flip_twins(rep["config"])
            adv_dists = [FFLDistribution.from_dict(c) for c in adv_cfgs]
            alpha, t_err, l_err = pull_back_alpha_mixture(
                base_dist=base_dist, adv_dists=adv_dists,
                transformer=transformer, lstm=lstm, device=device,
                rng=rng, n_probe=1000, batch_size=64,
                max_lstm_glitch=max_lstm_glitch,
            )
            families.append(MixtureFamily(
                base_dist=base_dist, adv_dists=adv_dists, alpha=alpha,
                cluster_mean_glitch=float(rep["T_glitch"]),
                rep_config=rep["config"],
                name=f"planted_{tmpl}_a{alpha:.2f}_n{len(adv_dists)}",
            ))
            print(f"[family] planted-{tmpl}: best_glitch={rep['T_glitch']:.3f} "
                  f"adv_variants={len(adv_dists)} alpha*={alpha:.2f} "
                  f"T@a*={t_err:.3f} lstm@a*={l_err:.4f}")

    # ---- Stationary / Piecewise: dedup + top-K -> ClusterFamily ----
    others = [r for r in valid if r["config"]["name"] != "planted"]
    if others:
        by_bin: dict[tuple, dict] = {}
        for r in others:
            key = _bin_config(r["config"])
            if key not in by_bin or r["fitness"] > by_bin[key]["fitness"]:
                by_bin[key] = r
        deduped = sorted(by_bin.values(), key=lambda r: r["T_glitch"], reverse=True)
        for r in deduped[:top_k]:
            adv_cfg = r["config"]
            alpha, t_err, l_err = pull_back_alpha(
                base_cfg=base_cfg, adv_cfg=adv_cfg,
                transformer=transformer, lstm=lstm, device=device,
                T=base_cfg["T"], rng=rng,
                max_lstm_glitch=max_lstm_glitch,
            )
            dist = interpolate_params(base_cfg, adv_cfg, alpha)
            families.append(ClusterFamily(
                dist=dist, alpha=alpha,
                cluster_mean_glitch=float(r["T_glitch"]),
                rep_config=adv_cfg,
                name=f"{adv_cfg['name']}_a{alpha:.2f}_g{r['T_glitch']:.2f}",
            ))
            print(f"[family] {adv_cfg['name']}: glitch={r['T_glitch']:.3f} "
                  f"alpha*={alpha:.2f} T@a*={t_err:.3f} lstm@a*={l_err:.4f}")

    families.sort(key=lambda f: f.cluster_mean_glitch, reverse=True)
    return families[:top_k]
