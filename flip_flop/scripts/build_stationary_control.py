"""PLAN_beat_liu_r4 Sec.2.D -- the scientific control.

Writes a sampler.json whose families are ONLY `stationary` distributions:
Liu's two tails + a fixed 8-point p_i grid spanning the gap axis. Explicitly
ZERO piecewise / periodic / mixture / anti-correlated families.

This isolates the question: does broader *stationary* coverage alone close the
non-stationary segment-drift break, or is the adversary's *non-stationary*
discovery necessary? The grid is FROZEN here (pre-registered) so it cannot be
tuned to fail.

Output is byte-compatible with dump_dataset.py's MixedSampler reader:
  {T, base_p_i, replay_frac, families:[{name,kind:PassthroughFamily,dist:{...}}]}

Usage:
  python -m flip_flop.scripts.build_stationary_control \
      --out_dir results/flip_flop/breaking_points/stationary_control
  python -m flip_flop.scripts.dump_dataset \
      --run_dir results/flip_flop/breaking_points/stationary_control --n 32000
"""
from __future__ import annotations

import argparse
import json
import os

T_DEFAULT = 512

# Pre-registered, frozen. 8 stationary points spanning sparse->dense, with
# symmetric instructions p_w = p_r = (1 - p_i) / 2 and unbiased bits, PLUS the
# two Liu tails. NOT tunable after seeing results.
P_I_GRID = [0.02, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 0.97]
LIU_TAILS = [
    {"name": "stationary", "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.5},  # FFL(0.98)
    {"name": "stationary", "p_w": 0.45, "p_r": 0.45, "bit_p1": 0.5},  # FFL(0.10)
]


def _stationary(p_w: float, p_r: float, T: int) -> dict:
    return {"name": "stationary", "T": T, "p_w": round(p_w, 6),
            "p_r": round(p_r, 6), "bit_p1": 0.5}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    p.add_argument("--base_p_i", type=float, default=0.8)
    p.add_argument("--replay_frac", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    families = []
    for i, t in enumerate(LIU_TAILS):
        families.append({
            "name": f"liu_{i}", "kind": "PassthroughFamily",
            "dist": _stationary(t["p_w"], t["p_r"], args.T),
        })
    for p_i in P_I_GRID:
        pw = pr = (1.0 - p_i) / 2.0
        families.append({
            "name": f"stationary_pi{p_i:.2f}", "kind": "PassthroughFamily",
            "dist": _stationary(pw, pr, args.T),
        })

    payload = {"T": args.T, "base_p_i": args.base_p_i,
               "replay_frac": args.replay_frac, "families": families}
    path = os.path.join(args.out_dir, "sampler.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[control] wrote {path}: {len(families)} stationary families "
          f"(2 Liu tails + {len(P_I_GRID)} grid), replay_frac={args.replay_frac}")
    # Audit trail mirroring pool_breaking_points output.
    bp = os.path.join(args.out_dir, "breaking_points.jsonl")
    with open(bp, "w", encoding="utf-8") as fh:
        for f in families:
            fh.write(json.dumps({"axis": "stationary_control",
                                 "config": f["dist"]}) + "\n")
    print(f"[control] wrote {bp}")


if __name__ == "__main__":
    main()
