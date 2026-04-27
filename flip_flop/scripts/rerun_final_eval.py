"""Re-run final evaluation on an already-completed adversary search.

Useful when `top_k.jsonl` is saved but `final_eval.jsonl` is missing or the
original run was killed mid-final-eval. Skips the search phase entirely.

Usage:
    python -m flip_flop.scripts.rerun_final_eval \
        --out_dir results/flip_flop/adversary/stationary \
        --n 20000 --n_seeds 3
"""
import argparse
import json
import os

import numpy as np
import torch

from flip_flop.adversary.io import dump_final_eval, load_frozen_model
from flip_flop.adversary.run import AdversaryConfig
from flip_flop.adversary.search import EvalResult


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="Directory containing top_k.jsonl + config.yaml from the original run.")
    p.add_argument("--n", type=int, default=20_000,
                   help="Sequences per seed for final evaluation.")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    cfg_path = os.path.join(args.out_dir, "config.yaml")
    cfg = AdversaryConfig.from_yaml(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device != "auto":
        device = cfg.device

    transformer = load_frozen_model(cfg.transformer_ckpt, cfg.transformer_cfg, device)
    lstm = load_frozen_model(cfg.lstm_ckpt, cfg.lstm_cfg, device) if cfg.use_lstm else None

    top_path = os.path.join(args.out_dir, "top_k.jsonl")
    with open(top_path) as f:
        top = [EvalResult(**json.loads(line)) for line in f]
    print(f"[rerun] loaded top-{len(top)} from {top_path}")

    dump_final_eval(
        top, transformer, lstm,
        n=args.n, batch_size=args.batch_size, device=device,
        n_seeds=args.n_seeds, out_dir=args.out_dir,
        lambda_lstm=cfg.lambda_lstm, lstm_tolerance=cfg.lstm_tolerance,
    )


if __name__ == "__main__":
    main()
