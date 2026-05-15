"""PLAN_beat_liu_r4 Sec.3 -- the cheap model-in-the-loop iterative loop.

Fictitious play on the minimax game. Per round k against the current model
M_{k-1} (M_0 = the canonical baseline):

  1. Quick adversary, search space = MixtureTwoStationary U AntiCorrelatedBits,
     regret objective, vs M_{k-1}            (two CMA runs, ~10 min)
  2. Pool = descriptor grid over ALL logs r1..rk + Liu tails (append-only;
     the covering set only grows)             (~3 min)
  3. From-scratch full retrain (10k steps, paper-faithful) on the pooled
     dataset                                  (~2 h)
  4. Eval the round model on the pre-registered held-out axes + LSTM skyline;
     write round_eval.json                    (~10 min)

Stop when round-over-round held-out improvement < --stop_delta or after
--rounds rounds. The PLAN's full pre-registered Sec.4 benchmark (eval_fair
with bootstrap CIs across all conditions) is the FINAL claim, run once after
the loop -- NOT folded into the per-round gate.

PLAN Sec.2.D: pass --transformer2 <seed1 ckpt> to enable the two-checkpoint
robustness filter (a one-time `run_baseline` on baseline_seed1.yaml). It
auto-detects results/flip_flop/baseline_seed1/model_final.pt if present.

Usage (from repo root):
  python -m flip_flop.scripts.run_beat_liu_loop --rounds 2
  python -m flip_flop.scripts.run_beat_liu_loop --rounds 2 --test_run
  python -m flip_flop.scripts.run_beat_liu_loop --phases adversary pool   # partial
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "flip_flop"

BASELINE_CKPT = RESULTS / "baseline" / "model_final.pt"
BASELINE_CFG = RESULTS / "baseline" / "config.yaml"
LSTM_CKPT = RESULTS / "lstm" / "model_final.pt"
LSTM_CFG = REPO_ROOT / "flip_flop" / "configs" / "lstm.yaml"
SEED1_CKPT = RESULTS / "baseline_seed1" / "model_final.pt"
SEED1_CFG = RESULTS / "baseline_seed1" / "config.yaml"

ADV_CONFIGS = {
    "mixture": "flip_flop/configs/quick_adversary_mixture_two_stationary.yaml",
    "anticorr": "flip_flop/configs/quick_adversary_anti_correlated_bits.yaml",
}
RETRAIN_CONFIG = "flip_flop/configs/full_from_scratch_beat_liu.yaml"


def _run(cmd: list[str], desc: str):
    print(f"\n=== {desc} ===\n$ {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        raise SystemExit(f"[fail] {desc} (returncode={res.returncode}, {dt:.1f}s)")
    print(f"[ok] {desc} ({dt:.1f}s)")


def _adv_log(round_dir: Path, fam: str) -> Path:
    return round_dir / f"adv_{fam}" / "adversary_log.jsonl"


def _all_logs_through(loop_dir: Path, k: int) -> list[str]:
    """Append-only: every adversary log from rounds 1..k (the covering set
    only grows -- PLAN Sec.3 step 2)."""
    logs = []
    for r in range(1, k + 1):
        for fam in ADV_CONFIGS:
            p = _adv_log(loop_dir / f"round{r}", fam)
            if p.exists():
                logs.append(str(p))
    return logs


def phase_adversary(loop_dir: Path, k: int, m_prev_ckpt: str, m_prev_cfg: str,
                    t2_ckpt: str, t2_cfg: str, args):
    round_dir = loop_dir / f"round{k}"
    for fam, cfg in ADV_CONFIGS.items():
        out_dir = round_dir / f"adv_{fam}"
        cmd = [sys.executable, "-m", "flip_flop.scripts.run_adversary",
               "--config", cfg, "--out_dir", str(out_dir),
               "--seed", str(args.seed),
               "--transformer_ckpt", m_prev_ckpt,
               "--transformer_cfg", m_prev_cfg]
        if t2_ckpt and t2_cfg:
            cmd += ["--transformer2_ckpt", t2_ckpt, "--transformer2_cfg", t2_cfg]
        if args.test_run:
            cmd.append("--test_run")
        _run(cmd, f"R{k} Phase 1 -- adversary {fam} vs M_{k-1}")


def phase_pool(loop_dir: Path, k: int, args):
    logs = _all_logs_through(loop_dir, k)
    if not logs:
        raise SystemExit(f"[fail] no adversary logs found through round {k}")
    out_dir = loop_dir / f"round{k}" / "breaking_points"
    _run([sys.executable, "-m", "flip_flop.scripts.pool_breaking_points",
          "--logs", *logs, "--out_dir", str(out_dir),
          "--pool_mode", "grid", "--max_families", str(args.max_families),
          "--min_t_glitch", str(args.min_t_glitch),
          "--min_reads_per_seq", "3.0"],
         f"R{k} Phase 2 -- descriptor-grid pool over rounds 1..{k} + Liu tails")


def phase_dataset(loop_dir: Path, k: int, args):
    bp = loop_dir / f"round{k}" / "breaking_points"
    _run([sys.executable, "-m", "flip_flop.scripts.dump_dataset",
          "--run_dir", str(bp), "--n", str(args.dataset_n),
          "--seed", str(12345 + k)],
         f"R{k} Phase 3a -- materialize {args.dataset_n}-seq dataset")


def phase_train(loop_dir: Path, k: int, args):
    bp = loop_dir / f"round{k}" / "breaking_points"
    out_dir = loop_dir / f"round{k}" / "model"
    cmd = [sys.executable, "-m", "flip_flop.scripts.run_retrain_from_dataset",
           "--config", RETRAIN_CONFIG, "--dataset", str(bp / "dataset.pt"),
           "--out_dir", str(out_dir)]
    if args.test_run:
        cmd.append("--test_run")
    _run(cmd, f"R{k} Phase 3b -- from-scratch retrain on pooled dataset")


# ---------------------------------------------------------------------------
# Phase 4: per-round held-out eval + stop gate
# ---------------------------------------------------------------------------
def _eval_round(loop_dir: Path, k: int, args) -> dict:
    """Load the round model + LSTM and score them on the frozen pre-registered
    held-out set. Returns {name: {model:.., lstm:..}, _score: mean-model-err}."""
    from flip_flop.adversary.distribution import FFLDistribution
    from flip_flop.adversary.heldout import heldout_set
    from flip_flop.adversary.io import load_frozen_model
    from flip_flop.eval import evaluate_dataset

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model_dir = loop_dir / f"round{k}" / "model"
    model = load_frozen_model(str(model_dir / "model_final.pt"),
                              str(model_dir / "config.yaml"), device)
    lstm = load_frozen_model(str(LSTM_CKPT), str(LSTM_CFG), device)

    n = 512 if args.test_run else args.eval_n
    out: dict = {"round": k}
    model_errs = []
    for name, cfg in heldout_set(T=512).items():
        toks = FFLDistribution.from_dict(cfg).sample(n, np.random.default_rng(args.eval_seed))
        m = evaluate_dataset(model, toks, batch_size=64, device=device)["error_rate"]
        l = evaluate_dataset(lstm, toks, batch_size=64, device=device)["error_rate"]
        out[name] = {"model": m, "lstm": l}
        model_errs.append(m)
        print(f"  [R{k} heldout] {name:20s} model={m:.5f}  lstm={l:.5f}")
    out["_score"] = float(np.mean(model_errs))
    return out


def phase_eval(loop_dir: Path, k: int, args) -> dict:
    res = _eval_round(loop_dir, k, args)
    path = loop_dir / f"round{k}" / "round_eval.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[ok] R{k} Phase 4 -- wrote {path} (held-out mean model err={res['_score']:.5f})")
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--loop_dir", default=str(RESULTS / "beat_liu_loop"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--start_round", type=int, default=1,
                   help="Resume from this round (uses round{n-1}/model as M_{k-1})")
    p.add_argument("--phases", nargs="+",
                   choices=["adversary", "pool", "dataset", "train", "eval"],
                   default=["adversary", "pool", "dataset", "train", "eval"])
    p.add_argument("--max_families", type=int, default=8)
    p.add_argument("--min_t_glitch", type=float, default=0.05)
    p.add_argument("--dataset_n", type=int, default=160000,
                   help="Sequences materialized per round (=train_steps*batch)")
    p.add_argument("--eval_n", type=int, default=16384,
                   help="Sequences per held-out distribution in the round gate")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--stop_delta", type=float, default=0.005,
                   help="Stop if round-over-round held-out improvement < this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transformer2", default=None,
                   help="Second-seed ckpt for the PLAN Sec.2.D filter "
                        "(default: auto-detect baseline_seed1)")
    p.add_argument("--device", default="auto")
    p.add_argument("--test_run", action="store_true")
    args = p.parse_args()

    loop_dir = Path(args.loop_dir)
    loop_dir.mkdir(parents=True, exist_ok=True)

    if not BASELINE_CKPT.exists():
        raise SystemExit(f"[fail] missing M_0 baseline: {BASELINE_CKPT}")

    # PLAN Sec.2.D second checkpoint: fixed across all rounds (the seed-1
    # baseline, NOT the per-round model -- it filters non-transferable exploits).
    t2_ckpt = args.transformer2 or (str(SEED1_CKPT) if SEED1_CKPT.exists() else "")
    t2_cfg = str(SEED1_CFG) if t2_ckpt and SEED1_CFG.exists() else ""
    if t2_ckpt and t2_cfg:
        print(f"[loop] two-checkpoint filter ON: {t2_ckpt}")
    else:
        print("[loop] two-checkpoint filter OFF (no baseline_seed1; Rung 2 mode)")

    phase_fns = {"adversary": phase_adversary, "pool": phase_pool,
                 "dataset": phase_dataset, "train": phase_train, "eval": phase_eval}

    t0 = time.time()
    prev_score = None
    for k in range(args.start_round, args.rounds + 1):
        (loop_dir / f"round{k}").mkdir(parents=True, exist_ok=True)
        if k == 1:
            m_prev_ckpt, m_prev_cfg = str(BASELINE_CKPT), str(BASELINE_CFG)
        else:
            prev_model = loop_dir / f"round{k-1}" / "model"
            m_prev_ckpt = str(prev_model / "model_final.pt")
            m_prev_cfg = str(prev_model / "config.yaml")
            if not Path(m_prev_ckpt).exists():
                raise SystemExit(f"[fail] round {k}: M_{k-1} missing ({m_prev_ckpt}); "
                                 f"run round {k-1} first")
        print(f"\n########## ROUND {k}/{args.rounds}  (M_{k-1} = {m_prev_ckpt}) ##########")

        for phase in args.phases:
            if phase == "adversary":
                phase_adversary(loop_dir, k, m_prev_ckpt, m_prev_cfg,
                                t2_ckpt, t2_cfg, args)
            elif phase == "eval":
                res = phase_eval(loop_dir, k, args)
                score = res["_score"]
                if prev_score is not None:
                    delta = prev_score - score
                    print(f"[loop] round {k}: held-out score {score:.5f} "
                          f"(prev {prev_score:.5f}, improvement {delta:+.5f})")
                    if delta < args.stop_delta:
                        print(f"[loop] STOP: improvement {delta:+.5f} "
                              f"< stop_delta {args.stop_delta}")
                        prev_score = score
                        break
                prev_score = score
            else:
                phase_fns[phase](loop_dir, k, args)
        else:
            continue
        break

    print(f"\n[done] beat-liu loop finished ({time.time() - t0:.1f}s total). "
          f"Final pre-registered Sec.4 claim: run eval_fair separately.")


if __name__ == "__main__":
    main()
