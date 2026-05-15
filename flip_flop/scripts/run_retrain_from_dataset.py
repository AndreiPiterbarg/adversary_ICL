"""Retrain a fresh model on a dataset.pt that a prior training run dumped.

Loads the saved (N, T) token tensor, wraps it in FixedDatasetSampler, and
trains using the standard train() loop. By default trains from scratch
(no init_from_ckpt), so the question this answers is: "if a new model
only ever saw the data this prior run was exposed to, what happens?"

Usage:
    python -m flip_flop.scripts.run_retrain_from_dataset \
        --config flip_flop/configs/retrain_from_dataset.yaml \
        --dataset results/flip_flop/retrain/<run>/dataset.pt
    python -m flip_flop.scripts.run_retrain_from_dataset \
        --config flip_flop/configs/retrain_from_dataset.yaml \
        --dataset path/to/dataset.pt --test_run
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from flip_flop.adversary.mixture_sampler import FixedDatasetSampler
from flip_flop.train import TrainConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Path to a dataset.pt produced by a run with cfg.dump_dataset_n > 0")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    sub = raw.get("retrain_from_dataset", {})

    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    cfg.init_from_ckpt = sub.get("init_from_ckpt", "")
    if args.test_run:
        cfg.train_steps = 50
        cfg.decay_end_step = 51
        cfg.warmup_steps = 5
        cfg.eval_every = 25
        cfg.save_every = 0
        cfg.eval_in_n = 64
        cfg.eval_sparse_n = 128
        cfg.eval_dense_n = 64

    if not os.path.exists(args.dataset):
        raise SystemExit(f"dataset not found: {args.dataset}")
    payload = torch.load(args.dataset, map_location="cpu")
    tokens = payload["tokens"]
    print(f"[from_dataset] loaded {tuple(tokens.shape)} from {args.dataset}")
    if "describe" in payload and payload["describe"] is not None:
        print(f"[from_dataset] source sampler: {json.dumps(payload['describe'])[:300]}")

    if tokens.shape[1] != cfg.seq_len:
        raise SystemExit(
            f"dataset T={tokens.shape[1]} does not match cfg.seq_len={cfg.seq_len}"
        )

    sampler = FixedDatasetSampler(tokens, source=os.path.abspath(args.dataset))
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "sampler.json"), "w") as f:
        json.dump(sampler.describe(), f, indent=2)

    result = train(cfg, sampler=sampler)
    print(f"[done] {result}")


if __name__ == "__main__":
    main()
