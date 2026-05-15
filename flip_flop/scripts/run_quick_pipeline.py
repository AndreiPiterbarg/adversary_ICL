"""End-to-end quick pipeline: adversary on missing axes -> pool -> materialize
dataset -> train from scratch -> eval. ~15-30 minutes on a single GPU.

Each phase reuses an existing entry-point script with a quick_*.yaml override:
  Phase 1 (adversary)  flip_flop.scripts.run_adversary  on quick_adversary_piecewise.yaml
                                                       and quick_adversary_planted.yaml
  Phase 2 (pool)       flip_flop.scripts.pool_breaking_points
  Phase 3 (dataset)    flip_flop.scripts.dump_dataset (32K sequences)
  Phase 4 (train)      flip_flop.scripts.run_retrain_from_dataset on quick_from_scratch_adv.yaml
  Phase 5 (eval)       diff the new model's eval_log.jsonl against the canonical
                       baseline's last full eval row.

The existing bitmarkov + writeflip adversary logs (already on the canonical
baseline; ~3000 evals each) are reused directly — no need to rerun them.

To scale up to publication quality: replace each quick_*.yaml with its
non-quick counterpart and rerun. Flow is identical; only hyperparameters differ.

Usage:
  python -m flip_flop.scripts.run_quick_pipeline
  python -m flip_flop.scripts.run_quick_pipeline --skip_adversary
  python -m flip_flop.scripts.run_quick_pipeline --phases pool dataset train eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "flip_flop"
QUICK_DIR = RESULTS_ROOT / "breaking_points" / "quick"

ADV_LOGS = [
    RESULTS_ROOT / "adversary" / "bitmarkov" / "adversary_log.jsonl",
    RESULTS_ROOT / "adversary" / "writeflip" / "adversary_log.jsonl",
    RESULTS_ROOT / "adversary" / "quick_piecewise" / "adversary_log.jsonl",
    RESULTS_ROOT / "adversary" / "quick_planted_decoy" / "adversary_log.jsonl",
]


def _run(cmd: list[str], desc: str):
    print(f"\n=== {desc} ===")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        raise SystemExit(f"[fail] {desc} (returncode={res.returncode}, {dt:.1f}s)")
    print(f"[ok] {desc} ({dt:.1f}s)")


def phase_adversary(args):
    if args.skip_adversary:
        print("[skip] Phase 1 (adversary) — using existing logs only")
        return
    _run([sys.executable, "-m", "flip_flop.scripts.run_adversary",
          "--config", "flip_flop/configs/quick_adversary_piecewise.yaml"],
         "Phase 1a — adversary: piecewise (quick CMA, ~5-8 min)")
    _run([sys.executable, "-m", "flip_flop.scripts.run_adversary",
          "--config", "flip_flop/configs/quick_adversary_planted.yaml"],
         "Phase 1b — adversary: planted decoy (quick grid, ~1-3 min)")


def phase_pool(args):
    logs_to_use = []
    for p in ADV_LOGS:
        if p.exists():
            logs_to_use.append(str(p))
        else:
            print(f"[warn] missing log {p} — pool will skip this axis")
    if not logs_to_use:
        raise SystemExit("[fail] no adversary logs found; rerun without --skip_adversary")
    _run([sys.executable, "-m", "flip_flop.scripts.pool_breaking_points",
          "--logs", *logs_to_use,
          "--out_dir", str(QUICK_DIR),
          "--top_k_per_axis", "5",
          "--min_t_glitch", "0.1"],
         "Phase 2 — pool breaking points")


def phase_dataset(args):
    _run([sys.executable, "-m", "flip_flop.scripts.dump_dataset",
          "--run_dir", str(QUICK_DIR), "--n", "32000", "--seed", "12345"],
         "Phase 3 — materialize 32K-sequence dataset (~1 min)")


def phase_train(args):
    _run([sys.executable, "-m", "flip_flop.scripts.run_retrain_from_dataset",
          "--config", "flip_flop/configs/quick_from_scratch_adv.yaml",
          "--dataset", str(QUICK_DIR / "dataset.pt")],
         "Phase 4 — train from scratch on the pooled dataset (2K steps, ~5-10 min)")


def _final_full_eval(path: Path) -> dict:
    """Return the last 'full' eval row (subset is null) from a train eval_log."""
    if not path.exists():
        return {}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    full = [r for r in rows if r.get("subset") in (None, 0)]
    return (full or rows)[-1] if rows else {}


def phase_eval(args):
    new_log = RESULTS_ROOT / "quick_from_scratch_adv" / "eval_log.jsonl"
    base_log = RESULTS_ROOT / "baseline" / "eval_log.jsonl"
    new_final = _final_full_eval(new_log)
    base_final = _final_full_eval(base_log)
    if not new_final:
        raise SystemExit(f"[fail] no eval rows in {new_log}")

    print("\n=== Phase 5 — eval comparison (final full-eval rows) ===")
    print(f"{'metric':<20} {'baseline':>12} {'quick_adv':>12} {'delta':>10}")
    for key in ("in_distribution", "sparse_tail", "dense_tail"):
        b = base_final.get(key, {}).get("error_rate")
        n = new_final.get(key, {}).get("error_rate")
        b_s = f"{b:.4f}" if isinstance(b, (int, float)) else "    -   "
        n_s = f"{n:.4f}" if isinstance(n, (int, float)) else "    -   "
        if isinstance(b, (int, float)) and isinstance(n, (int, float)):
            d = n - b
            d_s = ("+" if d >= 0 else "") + f"{d:.4f}"
        else:
            d_s = "    -   "
        print(f"  {key:<18} {b_s:>12} {n_s:>12} {d_s:>10}")
    print()
    print("[note] negative delta on sparse_tail/dense_tail = quick_adv beats baseline.")
    print("[note] in_distribution may regress slightly if replay_frac < 1.0.")
    print("[note] for a fair comparison, scale to baseline.yaml's 10K steps "
          "and 100K-sample sparse eval before drawing conclusions.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--skip_adversary", action="store_true",
                   help="Use whatever adversary logs already exist; skip Phase 1.")
    p.add_argument("--phases", nargs="+",
                   choices=["adversary", "pool", "dataset", "train", "eval"],
                   default=["adversary", "pool", "dataset", "train", "eval"],
                   help="Subset of phases to run, in order.")
    args = p.parse_args()

    phase_fns = {"adversary": phase_adversary, "pool": phase_pool,
                 "dataset": phase_dataset, "train": phase_train,
                 "eval": phase_eval}
    t0 = time.time()
    for phase in args.phases:
        phase_fns[phase](args)
    print(f"\n[done] quick pipeline finished ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
