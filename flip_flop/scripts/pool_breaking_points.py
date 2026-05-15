"""Pool high-fitness records from N adversary logs into one MixedSampler config.

Two pooling modes (PLAN_beat_liu_r4 §2.C):

  "grid" (default) — Quality-Diversity descriptor grid. Every surviving record
  is binned by two free, model-free descriptors:
    * write→read gap p90 : 4 bins  [4,16) [16,64) [64,256) [256,T]
    * read density       : 2 bins  [3,8) [8,∞)
  The single highest-regret distribution per non-empty cell is kept (≤ 8
  cells; capped at --max_families). This structurally guarantees the
  gap-axis extremes (Liu's FFL(0.98)/FFL(0.1) endpoint regimes) survive even
  when an interior degenerate mode has higher individual regret — the
  CVaR/core-set coverage argument. Replaces the old top-K-by-fitness pooling,
  which collapsed every "family" into ~3 near-duplicate corners.

  "top_k" (legacy) — dedup by rounded params, top-K per axis by T_glitch.
  Kept for backward compatibility (run_quick_pipeline passes --top_k_per_axis).

For each input log:
  1. Filter to records with T_glitch >= min_t_glitch, lstm_glitch <
     max_lstm_glitch, read_density >= min_reads_per_seq, is_valid.
  2. Pool by the selected mode.

Output:
  <out_dir>/breaking_points.jsonl  — one record per pooled family
  <out_dir>/sampler.json           — MixedSampler.describe() payload, ready for
                                     dump_dataset.py + run_retrain_from_dataset.py

Each pooled family is a PassthroughFamily wrapping the discovered FFLDistribution
at alpha=1 (the actual breaking distribution). For alpha-pull-back to LSTM-clean
targets, use flip_flop.adversary.family.extract_families_from_adversary_log
with frozen models loaded — that path is preserved and unchanged.

Optionally appends FFL(0.98) sparse and FFL(0.1) dense tails as additional
families (default ON), making the resulting mixture a strict superset of Liu R4
(safety floor: cannot do worse than Liu in expectation).

Usage:
  python -m flip_flop.scripts.pool_breaking_points \
      --logs results/flip_flop/adversary/r1_mixture/adversary_log.jsonl \
             results/flip_flop/adversary/r1_anticorr/adversary_log.jsonl \
      --out_dir results/flip_flop/breaking_points/beat_liu_r1
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

from flip_flop.adversary.distribution import FFLDistribution, Stationary
from flip_flop.adversary.family import PassthroughFamily
from flip_flop.data import R, write_read_gaps


def _read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"log not found: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _filter(records: Iterable[dict], min_t_glitch: float, max_lstm_glitch: float,
            min_reads_per_seq: float = 0.0) -> list[dict]:
    return [
        r for r in records
        if r.get("is_valid", True)
        and r.get("T_glitch", 0.0) >= min_t_glitch
        and r.get("lstm_glitch", 1.0) < max_lstm_glitch
        and r.get("read_density", float("inf")) >= min_reads_per_seq
    ]


def _regret(r: dict) -> float:
    """Score used to rank within a cell. In regret mode the logged `fitness`
    already encodes (min over checkpoints of) T_glitch − LSTM_glitch minus the
    soft hinge — the exact quantity the search optimized. Fall back to the raw
    regret for legacy penalty-mode logs."""
    f = r.get("fitness")
    if f is None or not np.isfinite(f):
        return float(r.get("T_glitch", 0.0)) - float(r.get("lstm_glitch", 0.0))
    return float(f)


# ---------------------------------------------------------------------------
# Descriptor grid (PLAN §2.C)
# ---------------------------------------------------------------------------
def _descriptors(rec: dict, T: int, n_probe: int = 512) -> tuple[float, float]:
    """(gap_p90, read_density) for a record. Uses the logged values when
    present (every fresh log has them); otherwise reconstructs them by
    sampling the config — zero model forward passes either way."""
    gp = rec.get("gap_p90")
    rd = rec.get("read_density")
    if gp is not None and rd is not None:
        return float(gp), float(rd)
    cfg = dict(rec["config"])
    cfg["T"] = T
    toks = FFLDistribution.from_dict(cfg).sample(n_probe, np.random.default_rng(0))
    _, gp2 = write_read_gaps(toks)
    rd2 = float((toks == R).sum(dim=1).float().mean().item())
    return (float(gp) if gp is not None else gp2,
            float(rd) if rd is not None else rd2)


def _gap_bin(gap_p90: float, T: int) -> int:
    edges = [4.0, 16.0, 64.0, 256.0, float(T)]   # 4 bins
    for i in range(len(edges) - 1):
        if gap_p90 < edges[i + 1]:
            return i
    return len(edges) - 2                          # >= T → top bin


def _density_bin(read_density: float) -> int:
    return 0 if read_density < 8.0 else 1          # [3,8) [8,∞)


def _descriptor_grid_pool(records: list[dict], T: int,
                          max_families: int) -> list[dict]:
    """Keep the single highest-regret record per (gap_p90, read_density) cell,
    then cap to the `max_families` highest-regret cells."""
    best_in_cell: dict[tuple[int, int], dict] = {}
    for r in records:
        gp, rd = _descriptors(r, T)
        cell = (_gap_bin(gp, T), _density_bin(rd))
        cur = best_in_cell.get(cell)
        if cur is None or _regret(r) > _regret(cur):
            best_in_cell[cell] = {**r, "gap_p90": gp, "read_density": rd,
                                  "cell": list(cell),
                                  "axis": r["config"]["name"]}
    pooled = sorted(best_in_cell.values(), key=_regret, reverse=True)
    return pooled[:max_families]


# ---------------------------------------------------------------------------
# Legacy top-K-per-axis pooling
# ---------------------------------------------------------------------------
def _round_floats(o, ndp: int):
    if isinstance(o, float):
        return round(o, ndp)
    if isinstance(o, dict):
        return {k: _round_floats(v, ndp) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [_round_floats(x, ndp) for x in o]
    if isinstance(o, tuple):
        return tuple(_round_floats(x, ndp) for x in o)
    return o


def _dedup_key(cfg: dict, ndp: int = 2) -> str:
    return json.dumps(_round_floats(cfg, ndp), sort_keys=True)


def _dedup_top_k_per_axis(records: list[dict], top_k: int, ndp: int = 2) -> list[dict]:
    by_axis: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        axis = r["config"]["name"]
        key = _dedup_key(r["config"], ndp)
        if key not in by_axis[axis] or r["T_glitch"] > by_axis[axis][key]["T_glitch"]:
            by_axis[axis][key] = r
    pooled: list[dict] = []
    for axis, bins in by_axis.items():
        deduped = sorted(bins.values(), key=lambda r: r["T_glitch"], reverse=True)[:top_k]
        for r in deduped:
            pooled.append({**r, "axis": axis})
    return pooled


def _to_passthrough(records: list[dict]) -> list[PassthroughFamily]:
    families = []
    for i, r in enumerate(records):
        dist = FFLDistribution.from_dict(r["config"])
        axis = r.get("axis", r["config"]["name"])
        name = f"{axis}_{i:02d}_g{r['T_glitch']:.2f}"
        families.append(PassthroughFamily(dist=dist, name=name))
    return families


def _liu_tail_families(T: int) -> list[PassthroughFamily]:
    """FFL(0.98) and FFL(0.1) as PassthroughFamily — strict superset of Liu R4
    (which retrained on FFL(0.98))."""
    sparse = Stationary(T=T, p_w=0.01, p_r=0.01, bit_p1=0.5)
    dense = Stationary(T=T, p_w=0.45, p_r=0.45, bit_p1=0.5)
    return [
        PassthroughFamily(dist=sparse, name="liu_ffl_098_sparse"),
        PassthroughFamily(dist=dense, name="liu_ffl_010_dense"),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--logs", nargs="+", required=True,
                   help="Paths to adversary_log.jsonl files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pool_mode", choices=["grid", "top_k"], default="grid",
                   help="grid = QD descriptor grid (PLAN §2.C, default); "
                        "top_k = legacy dedup + top-K per axis")
    p.add_argument("--max_families", type=int, default=8,
                   help="Cap on pooled adversarial families (grid mode)")
    p.add_argument("--top_k_per_axis", type=int, default=5,
                   help="top_k mode only: families kept per axis")
    p.add_argument("--min_t_glitch", type=float, default=0.1,
                   help="Lower bound on T_glitch to count as a breaking point")
    p.add_argument("--max_lstm_glitch", type=float, default=0.01,
                   help="Upper bound on lstm_glitch — keeps the data 'valid' (LSTM still clean)")
    p.add_argument("--min_reads_per_seq", type=float, default=0.0,
                   help="Lower bound on expected reads/sequence — rejects degenerate "
                        "near-zero-read distributions (old logs lacking the field pass through)")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--base_p_i", type=float, default=0.8)
    p.add_argument("--replay_frac", type=float, default=0.5,
                   help="Fraction of training sequences drawn from base FFL(base_p_i)")
    p.add_argument("--include_liu_tails", action="store_true", default=True,
                   help="Append FFL(0.98) sparse and FFL(0.1) dense as extra families (default ON)")
    p.add_argument("--no_liu_tails", dest="include_liu_tails", action="store_false")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_records: list[dict] = []
    for log in args.logs:
        recs = _read_log(log)
        kept = _filter(recs, args.min_t_glitch, args.max_lstm_glitch,
                       args.min_reads_per_seq)
        all_records.extend(kept)
        print(f"[pool] {log}: {len(recs)} total -> {len(kept)} kept "
              f"(T_glitch>={args.min_t_glitch}, lstm_glitch<{args.max_lstm_glitch}, "
              f"reads/seq>={args.min_reads_per_seq})")

    if not all_records:
        raise SystemExit("[fail] no records survived filtering; nothing to pool")

    if args.pool_mode == "grid":
        pooled = _descriptor_grid_pool(all_records, args.T, args.max_families)
        cells = ", ".join(f"({c[0]},{c[1]})" for c in
                          sorted(r["cell"] for r in pooled))
        print(f"[pool] descriptor grid: {len(pooled)} families over "
              f"non-empty (gap_p90 x read_density) cells: {cells}")
    else:
        pooled = _dedup_top_k_per_axis(all_records, args.top_k_per_axis)
        by_axis_count: dict[str, int] = defaultdict(int)
        for r in pooled:
            by_axis_count[r["axis"]] += 1
        print(f"[pool] top_k: dedup + top-{args.top_k_per_axis} per axis -> "
              f"{len(pooled)} families, "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_axis_count.items())))

    families = _to_passthrough(pooled)
    if args.include_liu_tails:
        families.extend(_liu_tail_families(args.T))
        print(f"[pool] +2 Liu R4 tail families (FFL(0.98), FFL(0.1))")

    breaking_points_path = os.path.join(args.out_dir, "breaking_points.jsonl")
    with open(breaking_points_path, "w") as fh:
        for r in pooled:
            fh.write(json.dumps({
                "axis": r.get("axis", r["config"]["name"]),
                "cell": r.get("cell"),
                "T_glitch": r["T_glitch"],
                "T_glitch2": r.get("T_glitch2"),
                "lstm_glitch": r["lstm_glitch"],
                "regret": _regret(r),
                "gap_p90": r.get("gap_p90"),
                "read_density": r.get("read_density"),
                "config": r["config"],
            }) + "\n")
        if args.include_liu_tails:
            fh.write(json.dumps({"axis": "liu_tail", "T_glitch": None, "lstm_glitch": 0.0,
                                 "config": {"name": "stationary", "T": args.T,
                                            "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.5}}) + "\n")
            fh.write(json.dumps({"axis": "liu_tail", "T_glitch": None, "lstm_glitch": 0.0,
                                 "config": {"name": "stationary", "T": args.T,
                                            "p_w": 0.45, "p_r": 0.45, "bit_p1": 0.5}}) + "\n")
    print(f"[pool] wrote {breaking_points_path}")

    sampler_payload = {
        "T": args.T,
        "base_p_i": args.base_p_i,
        "replay_frac": args.replay_frac,
        "families": [fam.to_dict() for fam in families],
    }
    sampler_path = os.path.join(args.out_dir, "sampler.json")
    with open(sampler_path, "w") as fh:
        json.dump(sampler_payload, fh, indent=2)
    print(f"[pool] wrote {sampler_path}")
    n_adv = len(pooled)
    print(f"[pool] {len(families)} total families "
          f"({n_adv} adversarial + {len(families) - n_adv} extra)")


if __name__ == "__main__":
    main()
