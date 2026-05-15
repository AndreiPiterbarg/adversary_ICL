"""Completely fair multi-model evaluation of the four trained FFLM conditions.

The earlier eval (results/flip_flop/full_from_scratch_*/eval_log.jsonl) graded
all four models on Liu's three stationary distributions. That set B (liu_only)
literally on its training distribution and put A (adv_only) on axes it never
trained on, so the comparison was structurally rigged.

This script fixes the comparison. Every distribution is evaluated on all four
trained Transformers + the LSTM skyline simultaneously, with three tiers
distinguishing the kind of exposure asymmetry:

  Tier C - cross-axis untouched:
      Periodic distributions. NO condition trained on the Periodic axis.
      Strictly fair to all four models. The headline tier.

  Tier H - within-axis held-out (asymmetric, labeled):
      bit_markov / write_flip / piecewise / stationary_novel.
      For axes that some conditions trained on, draw specific distributions
      that match no training pool config (canonical-rounded dedup). The axis
      is "in-axis" for some conditions and "out-of-axis" for others; this is
      reported in the summary table so the reader can see which condition's
      training axis the cell tests.

  Tier L - training-overlap controls:
      Liu's three eval distributions, marked as training data for B and
      full_adv. Reported but never folded into the headline summary.

For every distribution we:
  1. Sample N sequences with a fixed seed (same sequences for every model).
  2. Run all five models on those sequences with read-position glitch rate.
  3. Mark the distribution invalid if the LSTM glitch rate exceeds a threshold
     (default 0.5 percent). LSTM-invalid distributions are dropped from
     headline statistics.

Output:
  <out_dir>/per_distribution.jsonl  - one row per (model, distribution) eval
  <out_dir>/summary.md              - the aggregate table

Usage:
  python -m flip_flop.scripts.eval_fair
  python -m flip_flop.scripts.eval_fair --n 4096 --out_dir results/flip_flop/eval_fair
  python -m flip_flop.scripts.eval_fair --tiers C H            # skip Tier L
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.adversary.heldout import HELDOUT_ANTICORR
from flip_flop.adversary.io import load_frozen_model
from flip_flop.eval import evaluate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "flip_flop"

MODELS = {
    "baseline":  (RESULTS / "baseline" / "model_final.pt",
                  RESULTS / "baseline" / "config.yaml"),
    "A_adv":     (RESULTS / "full_from_scratch_adv_only" / "model_final.pt",
                  RESULTS / "full_from_scratch_adv_only" / "config.yaml"),
    "B_liu":     (RESULTS / "full_from_scratch_liu_only" / "model_final.pt",
                  RESULTS / "full_from_scratch_liu_only" / "config.yaml"),
    "full_adv":  (RESULTS / "full_from_scratch_adv" / "model_final.pt",
                  RESULTS / "full_from_scratch_adv" / "config.yaml"),
    "C_regret":  (RESULTS / "full_from_scratch_regret" / "model_final.pt",
                  RESULTS / "full_from_scratch_regret" / "config.yaml"),
    # PLAN_beat_liu_r4 §3 final retrain (pooled descriptor-grid + Liu tails).
    "beat_liu":  (RESULTS / "full_from_scratch_beat_liu" / "model_final.pt",
                  RESULTS / "full_from_scratch_beat_liu" / "config.yaml"),
    "lstm":      (RESULTS / "lstm" / "model_final.pt",
                  RESULTS / "lstm" / "config.yaml"),
}

ADVERSARY_LOGS = {
    "bit_markov": RESULTS / "adversary" / "bitmarkov" / "adversary_log.jsonl",
    "write_flip": RESULTS / "adversary" / "writeflip" / "adversary_log.jsonl",
    "piecewise":  RESULTS / "adversary" / "quick_piecewise" / "adversary_log.jsonl",
}

TRAINING_POOL = RESULTS / "breaking_points" / "quick" / "breaking_points.jsonl"

# Axis-access labels: which conditions' training distribution covered this axis.
# Used to annotate every result row so the reader can see the asymmetry.
AXIS_ACCESS = {
    "periodic":         {"baseline": "out", "A_adv": "out", "B_liu": "out", "full_adv": "out", "C_regret": "out", "beat_liu": "out"},
    "bit_markov":       {"baseline": "out", "A_adv": "in",  "B_liu": "out", "full_adv": "in",  "C_regret": "out", "beat_liu": "out"},
    "write_flip":       {"baseline": "out", "A_adv": "in",  "B_liu": "out", "full_adv": "in",  "C_regret": "out", "beat_liu": "out"},
    "piecewise":        {"baseline": "out", "A_adv": "in",  "B_liu": "out", "full_adv": "in",  "C_regret": "in",  "beat_liu": "out"},
    "stationary_novel": {"baseline": "out", "A_adv": "out", "B_liu": "in",  "full_adv": "in",  "C_regret": "out", "beat_liu": "in"},
    "liu_control":      {"baseline": "out", "A_adv": "out", "B_liu": "in",  "full_adv": "in",  "C_regret": "out", "beat_liu": "in"},
    # PLAN §4 (ii): pre-registered held-out AntiCorrelatedBits. Only the
    # beat-liu pool ever touches the anti_correlated_bits axis (and never
    # these exact frozen params); every other condition is strictly 'out'.
    "heldout_anticorr": {"baseline": "out", "A_adv": "out", "B_liu": "out", "full_adv": "out", "C_regret": "out", "beat_liu": "in"},
}


# ---------------------------------------------------------------------------
# Dedup key matching the one used by pool_breaking_points (so we can filter
# held-out candidates against the actual training pool).
# ---------------------------------------------------------------------------
def _round_floats(o, ndp: int):
    if isinstance(o, float):
        return round(o, ndp)
    if isinstance(o, dict):
        return {k: _round_floats(v, ndp) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_round_floats(x, ndp) for x in o]
    return o


def _dedup_key(cfg: dict, ndp: int = 2) -> str:
    return json.dumps(_round_floats(cfg, ndp), sort_keys=True)


def _training_pool_keys() -> set[str]:
    if not TRAINING_POOL.exists():
        return set()
    keys = set()
    with open(TRAINING_POOL) as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(_dedup_key(json.loads(line)["config"]))
    return keys


# ---------------------------------------------------------------------------
# Distribution builders for each tier.
# ---------------------------------------------------------------------------
def _build_tier_C_periodic(n: int, T: int, seed: int) -> list[dict]:
    """20 Periodic distributions sampled from a fixed prior. NO model trained
    on the periodic axis, so this tier is strictly fair to all four models."""
    rng = np.random.default_rng(seed)
    dists = []
    while len(dists) < n:
        period = int(rng.integers(2, 8))  # period in [2, 7]
        pattern = []
        for _ in range(period):
            # Full-simplex prior: do NOT bias toward FFL(0.8)-ish mixes, or the
            # headline tier covertly favors models trained on stationary/Liu
            # data. p_w, p_r each span [0,1] (reject only if their sum exceeds
            # the simplex); bit_p1 spans the full [0,1].
            p_w = float(rng.uniform(0.0, 1.0))
            p_r = float(rng.uniform(0.0, 1.0))
            if p_w + p_r > 1.0:
                continue
            bp1 = float(rng.uniform(0.0, 1.0))
            pattern.append([p_w, p_r, bp1])
        if len(pattern) != period:
            continue
        # Reject patterns that have zero reads or zero writes across the period
        # - those are degenerate FFL distributions that the LSTM skyline filter
        # would catch anyway, but cheaper to reject up front.
        if sum(p[0] for p in pattern) < 0.05 or sum(p[1] for p in pattern) < 0.05:
            continue
        dists.append({
            "tier": "C",
            "axis": "periodic",
            "name": f"periodic_{len(dists):02d}",
            "config": {"name": "periodic", "T": T, "period": period, "pattern": pattern},
        })
    return dists


def _build_tier_H_axis(axis: str, n: int, T: int, seed: int,
                       training_keys: set[str]) -> list[dict]:
    """Within-axis held-out: pick N records from the adversary log whose configs
    are NOT in the training pool (canonical dedup). Pick stratified by T_glitch
    quartile so easy / medium / hard cases all show up."""
    log = ADVERSARY_LOGS[axis]
    if not log.exists():
        return []
    records = []
    with open(log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("is_valid", True):
                continue
            if r.get("lstm_glitch", 1.0) >= 0.01:
                continue
            if r.get("T_glitch", 0.0) < 0.05:
                continue
            cfg = dict(r["config"])
            cfg["T"] = T
            if _dedup_key(cfg) in training_keys:
                continue
            records.append({"T_glitch": r["T_glitch"], "config": cfg})

    if not records:
        return []

    # Stratify by T_glitch quartile so we test across difficulty.
    records.sort(key=lambda r: r["T_glitch"])
    rng = np.random.default_rng(seed)
    bins = np.array_split(records, 4) if len(records) >= 4 else [records]
    per_bin = max(1, n // len(bins))
    picked: list[dict] = []
    for b in bins:
        if len(b) == 0:
            continue
        idx = rng.choice(len(b), size=min(per_bin, len(b)), replace=False)
        picked.extend(b[i] for i in idx)
    picked = picked[:n]

    dists = []
    for i, rec in enumerate(picked):
        dists.append({
            "tier": "H",
            "axis": axis,
            "name": f"{axis}_holdout_{i:02d}_g{rec['T_glitch']:.2f}",
            "config": rec["config"],
        })
    return dists


def _build_tier_H_stationary_novel(T: int) -> list[dict]:
    """Stationary distributions on (p_i in 0.2..0.95 excluding training pts).
    Symmetric-instruction setup (p_w = p_r = (1 - p_i) / 2), bits uniform."""
    excluded = {0.10, 0.80, 0.98}  # baseline+B's training points
    p_i_grid = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 0.95]
    dists = []
    for i, p_i in enumerate(p_i_grid):
        if p_i in excluded:
            continue
        p_w = (1.0 - p_i) / 2.0
        p_r = (1.0 - p_i) / 2.0
        dists.append({
            "tier": "H",
            "axis": "stationary_novel",
            "name": f"stationary_pi{p_i:.2f}",
            "config": {"name": "stationary", "T": T, "p_w": p_w, "p_r": p_r, "bit_p1": 0.5},
        })
    return dists


def _build_tier_H_heldout_anticorr(T: int) -> list[dict]:
    """PLAN §4 (ii): the pre-registered held-out AntiCorrelatedBits config.
    Frozen in flip_flop/adversary/heldout.py and committed to git BEFORE
    Rung 2. Never fed to the adversary objective. B_liu (trained only on
    Liu's i.i.d. tails) should show non-trivial error here while the LSTM
    stays ~0; a beat-liu model should be ~LSTM. This is the STRETCH axis."""
    cfg = dict(HELDOUT_ANTICORR)
    cfg["T"] = T
    return [{
        "tier": "H", "axis": "heldout_anticorr",
        "name": f"heldout_anticorr_pw{cfg['p_w']:.2f}_rho{cfg['rho']:.1f}",
        "config": cfg,
    }]


def _build_tier_L_liu(T: int) -> list[dict]:
    """Liu's three eval distributions. B and full_adv trained on these."""
    return [
        {"tier": "L", "axis": "liu_control", "name": "liu_ffl_080",
         "config": {"name": "stationary", "T": T, "p_w": 0.1,  "p_r": 0.1,  "bit_p1": 0.5}},
        {"tier": "L", "axis": "liu_control", "name": "liu_ffl_098",
         "config": {"name": "stationary", "T": T, "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.5}},
        {"tier": "L", "axis": "liu_control", "name": "liu_ffl_010",
         "config": {"name": "stationary", "T": T, "p_w": 0.45, "p_r": 0.45, "bit_p1": 0.5}},
    ]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def _eval_one_distribution(dist_entry: dict, models: dict, n: int, T: int,
                           seed: int, device: str, eval_batch: int) -> list[dict]:
    """Sample N sequences once, evaluate every model on the SAME tensor."""
    cfg = dict(dist_entry["config"])
    cfg["T"] = T  # force T in case the log record had a different T
    dist = FFLDistribution.from_dict(cfg)
    rng = np.random.default_rng(seed)
    tokens = dist.sample(n, rng)  # (n, T) LongTensor on CPU
    rows = []
    for model_name, model in models.items():
        result = evaluate_dataset(model, tokens, batch_size=eval_batch, device=device)
        rows.append({
            "tier": dist_entry["tier"],
            "axis": dist_entry["axis"],
            "distribution": dist_entry["name"],
            "config": cfg,
            "model": model_name,
            "axis_access": (AXIS_ACCESS.get(dist_entry["axis"], {}).get(model_name)
                            if model_name != "lstm" else "skyline"),
            "error_rate": result["error_rate"],
            "num_errors": result["num_errors"],
            "num_predictions": result["num_predictions"],
            "loss": result["loss"],
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(rows: list[dict], lstm_threshold: float):
    # Find LSTM-invalid distributions (skyline failed -> distribution suspect).
    lstm_by_dist = {(r["axis"], r["distribution"]): r["error_rate"]
                    for r in rows if r["model"] == "lstm"}
    invalid = {k for k, v in lstm_by_dist.items() if v > lstm_threshold}

    by_cell = defaultdict(list)  # (model, tier, axis) -> [error_rates...]
    for r in rows:
        if r["model"] == "lstm":
            continue
        if (r["axis"], r["distribution"]) in invalid:
            continue
        by_cell[(r["model"], r["tier"], r["axis"])].append(r["error_rate"])

    summary = []
    for (model, tier, axis), errs in sorted(by_cell.items()):
        arr = np.array(errs)
        summary.append({
            "model": model, "tier": tier, "axis": axis,
            "axis_access": AXIS_ACCESS.get(axis, {}).get(model, "unknown"),
            "n": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)) if len(arr) > 1 else float(arr[0]),
            "max": float(arr.max()),
        })
    return summary, invalid


def _write_summary_md(summary: list[dict], invalid: set, n_total_dists: int,
                      args, out_path: Path):
    lines = []
    lines.append("# Fair multi-model FFLM evaluation\n")
    lines.append(f"- N sequences per distribution: {args.n}")
    lines.append(f"- T (sequence length): {args.T}")
    lines.append(f"- Sample seed: {args.seed}")
    lines.append(f"- LSTM-invalid threshold: error_rate > {args.lstm_threshold}")
    lines.append(f"- Distributions evaluated: {n_total_dists} "
                 f"(of which {len(invalid)} dropped on LSTM-validity filter)\n")
    lines.append("## Axis-access legend")
    lines.append("- `in` = axis was present in this condition's training data.")
    lines.append("- `out` = axis was NOT in this condition's training data.")
    lines.append("- Reading a row: low error_rate with `out` access = genuine "
                 "generalization. Low error_rate with `in` access = expected.\n")

    by_tier = defaultdict(list)
    for r in summary:
        by_tier[r["tier"]].append(r)
    tier_names = {
        "C": "Tier C - cross-axis untouched (strictly fair: all conditions are 'out')",
        "H": "Tier H - within-axis held-out (asymmetric; labeled per cell)",
        "L": "Tier L - Liu controls (training overlap for B and full_adv)",
    }
    for tier in ("C", "H", "L"):
        if tier not in by_tier:
            continue
        lines.append(f"\n## {tier_names[tier]}\n")
        lines.append(f"| model | axis | access | n | mean | median | p90 | max |")
        lines.append(f"|---|---|---|---:|---:|---:|---:|---:|")
        for r in sorted(by_tier[tier], key=lambda x: (x["axis"], x["model"])):
            lines.append(f"| {r['model']} | {r['axis']} | {r['axis_access']} "
                         f"| {r['n']} | {r['mean']:.4g} | {r['median']:.4g} "
                         f"| {r['p90']:.4g} | {r['max']:.4g} |")

    lines.append("\n## How to read this\n")
    lines.append("- **Tier C is the headline.** Any model with a low p90 on "
                 "Tier C generalized to a distribution axis it never trained "
                 "on. If full_adv beats B on Tier C, the adversary "
                 "contribution is real.")
    lines.append("- **Tier H** is the per-axis story. A condition with `in` "
                 "access SHOULD do well on that axis - if it does not, "
                 "something is wrong with the training. A condition with `out` "
                 "access doing well is the interesting case (cross-axis "
                 "generalization).")
    lines.append("- **Tier L** numbers can be compared to the original "
                 "eval_log.jsonl rows; they are listed only for continuity "
                 "with the rigged eval and should not be averaged into any "
                 "headline.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", default=str(RESULTS / "eval_fair"))
    p.add_argument("--n", type=int, default=4096,
                   help="Sequences per distribution")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lstm_threshold", type=float, default=0.005,
                   help="Drop distributions where LSTM error_rate exceeds this")
    p.add_argument("--n_tier_C", type=int, default=20)
    p.add_argument("--n_tier_H_per_axis", type=int, default=20)
    p.add_argument("--tiers", nargs="+", default=["C", "H", "L"],
                   choices=["C", "H", "L"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load models -----------------------------------------------------
    print(f"[load] device={device}")
    models = {}
    for name, (ckpt, cfg) in MODELS.items():
        if not ckpt.exists() or not cfg.exists():
            print(f"[load]   [skip] {name}: missing {ckpt if not ckpt.exists() else cfg}")
            continue
        models[name] = load_frozen_model(str(ckpt), str(cfg), device)
        print(f"[load]   {name}")
    if "lstm" not in models:
        raise SystemExit("[fail] LSTM skyline is required but missing "
                          f"({MODELS['lstm'][0]}); cannot run the validity filter")
    if len(models) < 2:
        raise SystemExit("[fail] need >=1 trained model besides the LSTM; "
                          "none of the conditions were found")

    # ---- Assemble eval distributions ------------------------------------
    training_keys = _training_pool_keys()
    print(f"[pool] {len(training_keys)} training-pool keys (for dedup)")

    all_dists: list[dict] = []
    if "C" in args.tiers:
        c = _build_tier_C_periodic(args.n_tier_C, args.T, args.seed)
        all_dists.extend(c)
        print(f"[tierC] {len(c)} Periodic distributions")
    if "H" in args.tiers:
        for axis in ("bit_markov", "write_flip", "piecewise"):
            h = _build_tier_H_axis(axis, args.n_tier_H_per_axis, args.T,
                                   args.seed, training_keys)
            all_dists.extend(h)
            print(f"[tierH] {len(h)} {axis} held-out distributions")
        sn = _build_tier_H_stationary_novel(args.T)
        all_dists.extend(sn)
        print(f"[tierH] {len(sn)} stationary_novel distributions")
        ha = _build_tier_H_heldout_anticorr(args.T)
        all_dists.extend(ha)
        print(f"[tierH] {len(ha)} pre-registered held-out AntiCorrelatedBits")
    if "L" in args.tiers:
        l = _build_tier_L_liu(args.T)
        all_dists.extend(l)
        print(f"[tierL] {len(l)} Liu control distributions")

    print(f"[plan] evaluating {len(all_dists)} distributions x {len(models)} "
          f"models = {len(all_dists) * len(models)} evals")

    # ---- Run -----------------------------------------------------------
    rows: list[dict] = []
    per_dist_path = out_dir / "per_distribution.jsonl"
    with open(per_dist_path, "w") as fh:
        for i, d in enumerate(all_dists):
            r = _eval_one_distribution(d, models, args.n, args.T,
                                       args.seed, device, args.eval_batch_size)
            rows.extend(r)
            for row in r:
                fh.write(json.dumps(row) + "\n")
            lstm_err = next(x["error_rate"] for x in r if x["model"] == "lstm")
            flag = "  [lstm-invalid]" if lstm_err > args.lstm_threshold else ""
            print(f"[{i+1:>3}/{len(all_dists)}] {d['tier']}/{d['axis']:<18} "
                  f"{d['name']:<32}  lstm={lstm_err:.4f}{flag}")
    print(f"[io] wrote {per_dist_path}")

    # ---- Aggregate -----------------------------------------------------
    summary, invalid = _aggregate(rows, args.lstm_threshold)
    summary_jsonl = out_dir / "summary.jsonl"
    with open(summary_jsonl, "w") as fh:
        for r in summary:
            fh.write(json.dumps(r) + "\n")
    print(f"[io] wrote {summary_jsonl}")

    summary_md = out_dir / "summary.md"
    _write_summary_md(summary, invalid, len(all_dists), args, summary_md)
    print(f"[io] wrote {summary_md}")
    print(f"[done] {len(invalid)} distributions dropped on LSTM filter")


if __name__ == "__main__":
    main()
