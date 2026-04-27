"""Evaluate a model on saved adversary top_k.jsonl from prior rounds.

Tier-1 diagnostic: does retraining preserve previous-round robustness?
Cheap (~30 sec / 5000 sequences) test. Avoids spending H100 cycles on a
fresh adversary search just to answer "did we forget?".

Usage:
    python -m flip_flop.scripts.eval_on_saved_topk \
        --model results/flip_flop/retrain/round_2/model_final.pt \
        --topk results_phase_b/results/flip_flop/adversary_r1/piecewise/top_k.jsonl:r1_piecewise \
        --topk results_phase_b/results/flip_flop/adversary_r1/stationary/top_k.jsonl:r1_stationary \
        --topk results_phase_b/results/flip_flop/adversary_r1/planted_decoy/top_k.jsonl:r1_planted \
        --n 5000 \
        --out results/flip_flop/retrain/round_2/saved_topk_eval.json
"""
import argparse
import json
import os

import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.adversary.io import load_frozen_model
from flip_flop.eval import evaluate_dataset


def _eval_topk_file(model, lstm, path: str, n: int, batch_size: int,
                    device: str, rng: np.random.Generator) -> dict:
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    per_config = []
    for i, r in enumerate(recs):
        # top_k.jsonl entries from search.py have "config" field;
        # final_eval.jsonl entries have it too.
        cfg = r.get("config")
        if cfg is None:
            raise ValueError(f"record {i} in {path} has no 'config' field")
        dist = FFLDistribution.from_dict(cfg)
        tokens = dist.sample(n, rng)
        t_err = evaluate_dataset(model, tokens,
                                 batch_size=batch_size, device=device)["error_rate"]
        l_err = (evaluate_dataset(lstm, tokens, batch_size=batch_size, device=device)["error_rate"]
                 if lstm is not None else None)
        per_config.append({
            "rank": i + 1,
            "T_glitch": float(t_err),
            "lstm_glitch": float(l_err) if l_err is not None else None,
            "search_fitness": r.get("fitness", r.get("search_fitness")),
            "search_T_glitch": r.get("T_glitch"),
        })
    glitches = [c["T_glitch"] for c in per_config]
    return {
        "n_configs": len(per_config),
        "n_per_config": n,
        "T_glitch_mean": float(np.mean(glitches)) if glitches else 0.0,
        "T_glitch_max": float(max(glitches)) if glitches else 0.0,
        "T_glitch_min": float(min(glitches)) if glitches else 0.0,
        "T_glitch_top5_mean": float(np.mean(sorted(glitches, reverse=True)[:5]))
                              if glitches else 0.0,
        "lstm_glitch_max": float(max(c["lstm_glitch"] or 0 for c in per_config))
                           if per_config else 0.0,
        "per_config": per_config,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model_final.pt")
    p.add_argument("--model_cfg", default="flip_flop/configs/baseline.yaml",
                   help="Architecture YAML for the model (defaults to baseline shape).")
    p.add_argument("--lstm", default="results/flip_flop/lstm/model_final.pt")
    p.add_argument("--lstm_cfg", default="flip_flop/configs/lstm.yaml")
    p.add_argument("--no_lstm", action="store_true",
                   help="Skip LSTM evaluation (faster).")
    p.add_argument("--topk", action="append", required=True,
                   help="One or more 'path:label' pairs. Repeat the flag.")
    p.add_argument("--n", type=int, default=5000,
                   help="Sequences sampled per config.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="JSON path to write the full summary.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_frozen_model(args.model, args.model_cfg, device)
    lstm = None if args.no_lstm else load_frozen_model(args.lstm, args.lstm_cfg, device)

    rng = np.random.default_rng(args.seed)
    summary = {
        "model": args.model,
        "lstm": None if args.no_lstm else args.lstm,
        "n_per_config": args.n,
        "results": {},
    }
    for spec in args.topk:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path = spec
            label = os.path.basename(os.path.dirname(path)) or path
        print(f"\n=== {label}  ({path}) ===")
        if not os.path.exists(path):
            print(f"  [skip] file not found")
            summary["results"][label] = {"error": "file not found", "path": path}
            continue
        out = _eval_topk_file(model, lstm, path, args.n, args.batch_size, device, rng)
        summary["results"][label] = out
        print(f"  n_configs={out['n_configs']}  "
              f"mean={out['T_glitch_mean']:.4f}  max={out['T_glitch_max']:.4f}  "
              f"top-5 mean={out['T_glitch_top5_mean']:.4f}  "
              f"lstm_max={out['lstm_glitch_max']:.4f}")
        for c in out["per_config"][:5]:
            print(f"    rank {c['rank']:>3}: T={c['T_glitch']:.4f}  "
                  f"lstm={c['lstm_glitch']}  (search={c.get('search_T_glitch')})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary -> {args.out}")


if __name__ == "__main__":
    main()
