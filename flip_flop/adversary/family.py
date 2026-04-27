"""Families: distribution-of-distributions extracted from adversary output.

A family is a parametric sampler representing a distinct failure *mechanism*
found by the adversary, at an operating point that is less extreme than the
adversary's sharp optimum.

Pipeline (extract_families_from_adversary_log):
  1. Filter adversary log to valid, high-T_glitch, low-LSTM_glitch candidates.
  2. Group by distribution type (Stationary / Piecewise). Planted is skipped
     for now — its discrete `template` field has no natural interpolation.
  3. Featurize each candidate: config params + cheap behavior stats from
     sampled sequences; z-score.
  4. HDBSCAN cluster per group (density-based, auto-k). Noise discarded.
  5. Per cluster: geometric median on flattened config params (robust against
     boundary-piled configs) -> representative config c_rep.
  6. Pull-back α: largest α ∈ {1.0, 0.8, ..., 0.1} such that
          LSTM_glitch(FFL(p_α)) < 0.001   (skyline still clean)
      AND Transformer_glitch(FFL(p_α)) > 0.3 * best_adv_glitch  (still hard)
     with p_α = (1-α)*p_base + α*c_rep.
  7. Score each cluster: cluster_size * cluster_mean_T_glitch; take top-K.
  8. Each surviving cluster becomes one ClusterFamily sampling from FFL(p_α*).

Caveat tracked in CLAUDE.md Open TODOs: no per-sequence jitter (every draw
has identical params). That's the first knob to turn post-v1.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .distribution import (FFLDistribution, Piecewise, Stationary,
                            WriteFlipRate, BitMarkov)


# ---------------------------------------------------------------------------
# Abstract base + stub
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
    """Trivial: wrap one FFLDistribution as a 'family of one'. Test / fallback."""
    dist: FFLDistribution
    name: str = "passthrough"
    # Unified scoring fields so global ranking works across family types.
    # For ClusterFamily these are size and cluster_mean_glitch; for
    # PassthroughFamily we set size=1 and glitch from the source record.
    cluster_size: int = 1
    cluster_mean_glitch: float = 0.0
    axis: str = ""  # "stationary" | "piecewise" | "planted" | "" (legacy)

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {"name": self.name, "kind": "PassthroughFamily",
                "dist": self.dist.to_dict(),
                "cluster_size": self.cluster_size,
                "cluster_mean_glitch": self.cluster_mean_glitch,
                "axis": self.axis}


@dataclass
class ClusterFamily(Family):
    """An α-pulled-back distribution derived from one HDBSCAN cluster.

    Used for distribution types where parameter interpolation is well-defined
    (Stationary, Piecewise). At α=0 the distribution equals base; at α=1 it
    equals the cluster's representative config; at intermediate α the
    parameters are convex combinations.
    """
    dist: FFLDistribution               # the FFL(p_α*) distribution
    alpha: float                         # chosen pull-back
    cluster_size: int
    cluster_mean_glitch: float
    rep_config: dict                     # the cluster's representative config (pre-interpolation)
    name: str = "cluster"
    axis: str = ""

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": "ClusterFamily",
            "alpha": self.alpha,
            "cluster_size": self.cluster_size,
            "cluster_mean_glitch": self.cluster_mean_glitch,
            "rep_config": self.rep_config,
            "dist": self.dist.to_dict(),
            "axis": self.axis,
        }


@dataclass
class MixtureFamily(Family):
    """Mixture-of-generators family for distribution types where parameter
    interpolation is undefined (e.g., Planted with discrete templates).

    Each sequence is drawn from base_dist with probability (1-α), else from
    one of the `adv_dists` variants picked uniformly. At α=0 every sample is
    base; at α=1 every sample is adv; intermediate α gives a per-sequence
    Bernoulli mixture. Validity of every output sequence is preserved
    automatically (each sample comes from one valid FFL distribution).

    Multiple `adv_dists` provide PER-SEQUENCE PARAMETER JITTER over the
    template's discrete parameters (Tier-1 v2 fix). Phase-B-Tier-1 found that
    training on a single planted config (b_decoy=0) closed those points but
    opened the bit-flipped twin (b_decoy=1) — the model just flipped its
    bias rather than learning state-tracking. Including the bit-flipped
    twin in `adv_dists` exposes both directions during training.

    The α-bisection rule (largest α with skyline-clean LSTM and Transformer
    ~50% glitch) carries over: T_err is monotone in α by construction.
    """
    base_dist: FFLDistribution
    adv_dists: list[FFLDistribution]      # 1+ variants; uniform-pick per sequence
    alpha: float
    cluster_size: int = 1
    cluster_mean_glitch: float = 0.0
    rep_config: dict = field(default_factory=dict)
    name: str = "mixture"
    axis: str = "planted"

    def __post_init__(self):
        assert 0.0 <= self.alpha <= 1.0, f"alpha out of [0,1]: {self.alpha}"
        assert len(self.adv_dists) >= 1, "MixtureFamily needs >=1 adv_dist"
        for d in self.adv_dists:
            assert self.base_dist.T == d.T, (
                f"T mismatch: base={self.base_dist.T}, adv={d.T}"
            )

    @property
    def adv_dist(self) -> FFLDistribution:
        """Backwards-compat alias: the first (representative) adv variant."""
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
            # Per-sequence: pick which adv variant uniformly.
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
            "cluster_size": self.cluster_size,
            "cluster_mean_glitch": self.cluster_mean_glitch,
            "rep_config": self.rep_config,
            "axis": self.axis,
            "base_dist": self.base_dist.to_dict(),
            "adv_dists": [d.to_dict() for d in self.adv_dists],
            "n_adv_variants": len(self.adv_dists),
        }


# ---------------------------------------------------------------------------
# Reconstruction from saved sampler.json
# ---------------------------------------------------------------------------
def family_from_dict(d: dict) -> Family:
    """Reconstruct a Family from a dict produced by `Family.to_dict()`.

    Used by run_retrain_sweep to rebuild the exact families a prior retrain
    trained on, without re-running clustering or α-bisection. Older
    sampler.json files predate the `axis` field; default to "".
    """
    kind = d["kind"]
    axis = d.get("axis", "")
    if kind == "PassthroughFamily":
        return PassthroughFamily(
            dist=FFLDistribution.from_dict(d["dist"]),
            name=d.get("name", "passthrough"),
            cluster_size=int(d.get("cluster_size", 1)),
            cluster_mean_glitch=float(d.get("cluster_mean_glitch", 0.0)),
            axis=axis,
        )
    if kind == "ClusterFamily":
        return ClusterFamily(
            dist=FFLDistribution.from_dict(d["dist"]),
            alpha=float(d["alpha"]),
            cluster_size=int(d["cluster_size"]),
            cluster_mean_glitch=float(d["cluster_mean_glitch"]),
            rep_config=dict(d.get("rep_config", {})),
            name=d.get("name", "cluster"),
            axis=axis,
        )
    if kind == "MixtureFamily":
        return MixtureFamily(
            base_dist=FFLDistribution.from_dict(d["base_dist"]),
            adv_dists=[FFLDistribution.from_dict(x) for x in d["adv_dists"]],
            alpha=float(d["alpha"]),
            cluster_size=int(d.get("cluster_size", 1)),
            cluster_mean_glitch=float(d.get("cluster_mean_glitch", 0.0)),
            rep_config=dict(d.get("rep_config", {})),
            name=d.get("name", "mixture"),
            axis=axis or "planted",
        )
    raise ValueError(f"unknown Family kind: {kind!r}")


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def _flatten_config_params(cfg: dict) -> np.ndarray:
    """Flatten a distribution config dict to a fixed-length real vector.

    Handles Stationary + Piecewise (same segment count). Returns None if
    unsupported type (caller skips).
    """
    name = cfg.get("name")
    if name in ("stationary", "bit_markov", "write_flip"):
        return np.array([cfg["p_w"], cfg["p_r"],
                         cfg.get("bit_p1", 0.5)])
    if name == "piecewise":
        segs = cfg["segments"]
        # Each segment: (start_frac, p_w, p_r, bit_p1) — start_frac is fixed
        # in our encoder so skip; keep the other three.
        return np.array([v for s in segs for v in (s[1], s[2], s[3])])
    return None


def _behavior_stats(cfg: dict, T: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n sequences from the config's distribution and compute cheap stats."""
    from .distribution import build
    # Rebuild dist
    dist = FFLDistribution.from_dict(cfg)
    tokens = dist.sample(n, rng).numpy()
    # Stats from even (instructions) and odd (data) positions.
    inst = tokens[:, 0::2]
    data = tokens[:, 1::2]
    from ..data import W, R, I, ZERO, ONE
    # 1. Mean write density
    write_density = (inst == W).mean()
    # 2. Mean read density
    read_density = (inst == R).mean()
    # 3. Bit-1 rate among writes
    is_w = inst == W
    bit_ones = (data == ONE) & is_w
    bit_one_rate = bit_ones.sum() / max(1, is_w.sum())
    # 4. Segment-concentration Gini of writes: how clustered writes are along the sequence.
    # Compute per-segment write count (4 bins along T), then Gini over the 4.
    n_inst = inst.shape[1]
    bins = 4
    bin_counts = np.zeros(bins)
    for b in range(bins):
        s, e = b * n_inst // bins, (b + 1) * n_inst // bins
        bin_counts[b] = (inst[:, s:e] == W).mean()
    gini = _gini(bin_counts)
    # 5. Mean read-to-last-write gap (positions per read)
    gaps = []
    for row in inst[: min(32, inst.shape[0])]:   # subsample — expensive loop
        last_w = -1
        for k in range(row.shape[0]):
            if row[k] == W:
                last_w = k
            elif row[k] == R and last_w >= 0:
                gaps.append(k - last_w)
    mean_gap = np.mean(gaps) if gaps else 0.0
    return np.array([write_density, read_density, bit_one_rate, gini, mean_gap])


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n


def _featurize_batch(
    configs: list[dict], T: int, n_behavior: int, rng: np.random.Generator
) -> np.ndarray:
    """Compute feature matrix (N, D). Mixes flattened config params + behavior stats."""
    rows = []
    for cfg in configs:
        p = _flatten_config_params(cfg)
        b = _behavior_stats(cfg, T, n_behavior, rng)
        rows.append(np.concatenate([p, b]))
    X = np.stack(rows)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _hdbscan_cluster(features: np.ndarray, min_cluster_size: int) -> list[np.ndarray]:
    """HDBSCAN; returns list of index arrays (noise excluded). Fallback: k-means."""
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(
            min_cluster_size=max(5, min_cluster_size),
            min_samples=max(3, min_cluster_size // 3),
        ).fit_predict(features)
    except Exception as e:  # pragma: no cover
        print(f"[family] HDBSCAN failed ({e}); falling back to k-means")
        labels = _kmeans_silhouette(features)

    groups = [np.where(labels == lbl)[0] for lbl in sorted(set(labels)) if lbl != -1]
    if not groups:
        print("[family] HDBSCAN returned all-noise; falling back to k-means")
        labels = _kmeans_silhouette(features)
        groups = [np.where(labels == lbl)[0] for lbl in sorted(set(labels))]
    return groups


def _kmeans_silhouette(features: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, min(6, len(features))):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(features)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(features, km.labels_)
        if s > best_score:
            best_score, best_k, best_labels = s, k, km.labels_
    return best_labels if best_labels is not None else np.zeros(len(features), dtype=int)


# ---------------------------------------------------------------------------
# Cluster representative
# ---------------------------------------------------------------------------
def _geometric_median(points: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Weiszfeld iteration. Robust against boundary-piled points."""
    x = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - x, axis=1)
        d_safe = np.maximum(d, 1e-10)
        w = 1.0 / d_safe
        x_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def _cluster_representative_config(configs: list[dict]) -> dict:
    """Geometric median over flattened params, re-cast to a valid config."""
    name = configs[0]["name"]
    T = configs[0]["T"]
    if name in ("stationary", "bit_markov", "write_flip"):
        params = np.stack([_flatten_config_params(c) for c in configs])
        med = _geometric_median(params)
        p_w, p_r, bit_p1 = _clip_simplex2(med[0], med[1]), _clip_simplex2(med[1], med[0]), _clip01(med[2])
        return {"name": "stationary", "T": T, "p_w": float(p_w),
                "p_r": float(p_r), "bit_p1": float(bit_p1)}
    if name == "piecewise":
        params = np.stack([_flatten_config_params(c) for c in configs])
        med = _geometric_median(params)
        n_seg = params.shape[1] // 3
        segs = []
        for k in range(n_seg):
            p_w = _clip_simplex2(med[3 * k], med[3 * k + 1])
            p_r = _clip_simplex2(med[3 * k + 1], med[3 * k])
            bit = _clip01(med[3 * k + 2])
            segs.append([k / n_seg, float(p_w), float(p_r), float(bit)])
        return {"name": "piecewise", "T": T, "segments": segs}
    raise ValueError(f"unsupported distribution type {name}")


def _clip_simplex2(a: float, b: float) -> float:
    """Clip a into [0, 1 - b] and ensure a >= 0, with b the other probability."""
    a = max(0.0, min(1.0, float(a)))
    b = max(0.0, min(1.0, float(b)))
    return min(a, max(0.0, 1.0 - b))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Parameter interpolation + pull-back α
# ---------------------------------------------------------------------------
def _lift_to_piecewise(base: Stationary, K: int) -> Piecewise:
    """Convert a Stationary into a Piecewise with K identical segments."""
    segs = [(k / K, base.p_w, base.p_r, base.bit_p1) for k in range(K)]
    return Piecewise(T=base.T, segments=segs)


def interpolate_params(
    base_cfg: dict, adv_cfg: dict, alpha: float
) -> FFLDistribution:
    """Linear interpolation in parameter space. Returns a new FFLDistribution.

    Requires both configs have the same type (after a possible Stationary-
    to-Piecewise lift when adv is Piecewise).
    """
    assert 0.0 <= alpha <= 1.0
    a_name = adv_cfg["name"]

    if a_name == "piecewise":
        # Lift base if needed.
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
                as_[0],  # start_frac — take adv's (identical)
                _clip01((1 - alpha) * bs[1] + alpha * as_[1]),  # p_w
                _clip01((1 - alpha) * bs[2] + alpha * as_[2]),  # p_r
                _clip01((1 - alpha) * bs[3] + alpha * as_[3]),  # bit_p1
            ])
        return Piecewise(T=adv_cfg["T"], segments=[tuple(s) for s in new_segs])

    if a_name in ("stationary", "bit_markov", "write_flip"):
        # Use Stationary for all interp results.
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
    """Generic bisection on α ∈ [0, 1]; type-agnostic. eval_at_alpha(α) ->
    (T_err, lstm_err). Used by pull_back_alpha (parameter interpolation) and
    pull_back_alpha_mixture (mixture-of-generators); same numerics either way.

    Returns (α*, T_err@α*, lstm_err@α*). Picks largest α satisfying both
    skyline-clean (lstm < max_lstm_glitch) and Transformer-still-hard
    (T_err near target_t_glitch). Endpoint cases:
      - LSTM fails at α=1: cannot soften, return α=1 + flag.
      - Transformer too easy at α=1: just return α=1.
      - Transformer too hard at α=0: just return α=0.
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
    ref_glitch: float = 1.0,  # kept for signature stability; unused
    alphas: tuple = (),       # kept for signature stability; unused
) -> tuple[float, float, float]:
    """Bisect α via parameter interpolation (used for ClusterFamily on
    Stationary / Piecewise types). Thin wrapper around _bisect_alpha."""
    def eval_alpha(alpha):
        dist = interpolate_params(base_cfg, adv_cfg, alpha)
        tokens = dist.sample(n_probe, rng)
        t = _eval_glitch(transformer, tokens, batch_size, device)
        l = _eval_glitch(lstm, tokens, batch_size, device) if lstm is not None else 0.0
        return t, l
    return _bisect_alpha(
        eval_alpha,
        max_lstm_glitch=max_lstm_glitch,
        target_t_glitch=target_t_glitch,
        max_iter=max_iter,
        tol=tol,
    )


def pull_back_alpha_mixture(
    base_dist: FFLDistribution,
    adv_dists: list[FFLDistribution],
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
    """Bisect α via mixture-of-generators. `adv_dists` is the list of adv
    variants the family will sample from (e.g. [original, bit-flipped twin]
    for planted decoy). At α=p, each training sequence is drawn from one of
    the adv variants (uniformly) with probability p, else from base_dist.
    Bisection rule unchanged from pull_back_alpha — T_err remains monotone
    in α since each variant's contribution is linear."""
    # Backwards-compat: accept a single FFLDistribution as well.
    if not isinstance(adv_dists, (list, tuple)):
        adv_dists = [adv_dists]
    def eval_alpha(alpha):
        fam = MixtureFamily(base_dist=base_dist, adv_dists=list(adv_dists), alpha=alpha)
        tokens = fam.sample(n_probe, rng)
        t = _eval_glitch(transformer, tokens, batch_size, device)
        l = _eval_glitch(lstm, tokens, batch_size, device) if lstm is not None else 0.0
        return t, l
    return _bisect_alpha(
        eval_alpha,
        max_lstm_glitch=max_lstm_glitch,
        target_t_glitch=target_t_glitch,
        max_iter=max_iter,
        tol=tol,
    )


def _planted_bit_flip_twins(rep_cfg: dict) -> list[dict]:
    """Given a planted config, return the list [original, bit-flipped twin].

    Bit-flip rules per template:
      - decoy:      flip b_decoy
      - distractor: flip b_true
      - disagree:   flip b_last
      - gap:        no bit param → return [original] only

    Per Tier-1 finding: training on a single bit-direction overfit and
    opened the symmetric attack. Including the twin forces the model to
    consult the actual write history, not memorize a bit-bias.
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
# End-to-end extraction
# ---------------------------------------------------------------------------
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
    n_behavior: int = 64,
    seed: int = 0,
) -> list[Family]:
    """Cluster + α-pull-back extraction.

    Threshold floor `min_t_glitch=0.01`: aggressive enough that mid-loop
    rounds (where the adversary returns weaker findings as mechanisms close)
    do not have all candidates filtered out. The previous default 0.5 caused
    the loop to "self-sabotage as it succeeds" — see Phase B postmortem.
    Round 3+ should switch to adaptive `max(0.01, β·max_T_in_log)`.

    Selection: per-axis floor (top-1 from each non-empty axis) + global rank
    fills remaining slots up to `top_k`. Prevents single-axis dominance
    (Phase B saw all 4 K-cap slots collapse to planted PassthroughFamily).

    If `transformer` and `lstm` are not provided, returns PassthroughFamily
    stubs over the top-K highest-fitness configs (legacy behavior for tests).
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

    # Legacy fallback when models aren't available (tests).
    if transformer is None or lstm is None or base_cfg is None:
        valid.sort(key=lambda r: r["fitness"], reverse=True)
        return [
            PassthroughFamily(FFLDistribution.from_dict(r["config"]),
                              name=f"adv_{i:02d}")
            for i, r in enumerate(valid[:top_k])
        ]

    # Group by distribution type. Planted -> MixtureFamily (mixture-of-
    # generators α-bisection because templates are discrete and parameter
    # interpolation is undefined). Stationary / Piecewise -> ClusterFamily
    # (HDBSCAN + geometric median + parameter-interpolation α-bisection).
    groups: dict[str, list[dict]] = {}
    planted_recs: list[dict] = []
    for r in valid:
        name = r["config"]["name"]
        if name == "planted":
            planted_recs.append(r)
        else:
            groups.setdefault(name, []).append(r)

    rng = np.random.default_rng(seed)
    T = base_cfg["T"]
    base_dist = FFLDistribution.from_dict(base_cfg)
    all_families: list[Family] = []

    # ---- Planted: group by template, build MixtureFamily per template ----
    # Each MixtureFamily includes the cluster's representative config AND
    # its bit-flipped twin (for templates with a bit param). This addresses
    # the Phase-B-Tier-1 finding that training on a single bit-direction
    # closed those configs but opened the symmetric attack.
    if planted_recs:
        by_template: dict[str, list[dict]] = {}
        for r in planted_recs:
            tmpl = r["config"]["template"]
            by_template.setdefault(tmpl, []).append(r)
        print(f"[family] planted: {len(planted_recs)} candidates across "
              f"{len(by_template)} template(s): {list(by_template.keys())}")
        for tmpl, recs_t in by_template.items():
            recs_t.sort(key=lambda r: r["fitness"], reverse=True)
            rep_rec = recs_t[0]
            rep_cfg = rep_rec["config"]
            mean_g = float(np.mean([r["T_glitch"] for r in recs_t]))
            # Generate bit-flipped twin for symmetry coverage.
            adv_cfgs = _planted_bit_flip_twins(rep_cfg)
            adv_dists = [FFLDistribution.from_dict(c) for c in adv_cfgs]
            alpha, t_err, l_err = pull_back_alpha_mixture(
                base_dist=base_dist, adv_dists=adv_dists,
                transformer=transformer, lstm=lstm, device=device,
                rng=rng, n_probe=1000, batch_size=64,
                max_lstm_glitch=max_lstm_glitch,
            )
            fam = MixtureFamily(
                base_dist=base_dist,
                adv_dists=adv_dists,
                alpha=alpha,
                cluster_size=len(recs_t),
                cluster_mean_glitch=mean_g,
                rep_config=rep_cfg,
                name=f"planted_{tmpl}_a{alpha:.2f}_n{len(adv_dists)}",
                axis="planted",
            )
            all_families.append(fam)
            print(f"[family]   planted-{tmpl}: n={len(recs_t)} mean_glitch={mean_g:.3f} "
                  f"adv_variants={len(adv_dists)} "
                  f"alpha*={alpha:.2f} T_err@a*={t_err:.3f} lstm@a*={l_err:.4f}")

    # ---- Stationary / Piecewise / etc: HDBSCAN + ClusterFamily ----
    for dist_type, recs_g in groups.items():
        configs = [r["config"] for r in recs_g]
        if len(configs) < 10:
            print(f"[family] skipping group '{dist_type}' (only {len(configs)} valid)")
            continue
        print(f"[family] group '{dist_type}': {len(configs)} candidates; clustering...")
        features = _featurize_batch(configs, T=T, n_behavior=n_behavior, rng=rng)
        min_cs = max(5, len(configs) // 50)
        clusters = _hdbscan_cluster(features, min_cluster_size=min_cs)
        print(f"[family]   -> {len(clusters)} cluster(s) (sizes: {[len(c) for c in clusters]})")

        best_glitch = max(r["T_glitch"] for r in recs_g)

        for c_idx, idx in enumerate(clusters):
            cluster_cfgs = [configs[i] for i in idx]
            cluster_recs = [recs_g[i] for i in idx]
            rep_cfg = _cluster_representative_config(cluster_cfgs)
            mean_g = float(np.mean([r["T_glitch"] for r in cluster_recs]))
            alpha, t_err, l_err = pull_back_alpha(
                base_cfg=base_cfg, adv_cfg=rep_cfg,
                transformer=transformer, lstm=lstm, device=device,
                T=T, rng=rng, ref_glitch=best_glitch,
                max_lstm_glitch=max_lstm_glitch,
            )
            dist = interpolate_params(base_cfg, rep_cfg, alpha)
            fam = ClusterFamily(
                dist=dist,
                alpha=alpha,
                cluster_size=len(idx),
                cluster_mean_glitch=mean_g,
                rep_config=rep_cfg,
                name=f"{dist_type}_c{c_idx:02d}_a{alpha:.2f}",
                axis=dist_type,
            )
            all_families.append(fam)
            print(f"[family]   {dist_type}-c{c_idx}: n={len(idx)} mean_glitch={mean_g:.3f} "
                  f"alpha*={alpha:.2f} T_err@a*={t_err:.3f} lstm@a*={l_err:.4f}")

    # ---- Per-axis floor + global rank fill ----
    return _select_with_axis_floor(all_families, top_k)


def _select_with_axis_floor(families: list[Family], top_k: int) -> list[Family]:
    """Reserve top-1 per non-empty axis; fill remaining slots by global rank.

    This guarantees cross-axis representation in the final mixture: even if
    one axis (typically planted, with fitness ≈ 1.0) dominates the rank,
    each represented axis still contributes at least one family. Beyond the
    floor, remaining slots go by global `(cluster_mean_glitch, cluster_size)`
    rank.

    If the number of distinct non-empty axes exceeds `top_k`, only the
    highest-glitch axes get a floor slot.
    """
    if not families:
        return []

    # Bucket by axis (defensive against missing/empty axis field).
    by_axis: dict[str, list[Family]] = {}
    for f in families:
        by_axis.setdefault(f.axis or "_unknown", []).append(f)

    # Floor: top-1 per axis, ranked by mean_glitch within the axis.
    floors: list[Family] = []
    for axis, fams in by_axis.items():
        fams_sorted = sorted(fams,
                             key=lambda f: (f.cluster_mean_glitch, f.cluster_size),
                             reverse=True)
        floors.append(fams_sorted[0])

    # If too many axes for top_k, keep only the strongest axes (by their
    # representative's glitch).
    floors.sort(key=lambda f: (f.cluster_mean_glitch, f.cluster_size), reverse=True)
    floors = floors[:top_k]
    floor_ids = {id(f) for f in floors}

    # Fill remaining slots by global rank over the rest.
    rest = [f for f in families if id(f) not in floor_ids]
    rest.sort(key=lambda f: (f.cluster_mean_glitch, f.cluster_size), reverse=True)
    final = floors + rest[:max(0, top_k - len(floors))]

    print(f"[family] axis floor+global rank: {len(floors)} floor + "
          f"{len(final) - len(floors)} fill = {len(final)} total")
    print(f"[family]   selection (axis | kind | name | glitch | size):")
    for f in final:
        print(f"[family]     {f.axis or '?':<12} | "
              f"{type(f).__name__:<16} | {f.name:<30} | "
              f"{f.cluster_mean_glitch:.4f} | {f.cluster_size}")
    return final
