"""Tier-A retrain: continue-train baseline on FFL(0.8) + adversary-found families.

Parameterized by --config (one yaml per axis). Loads the baseline transformer
+ LSTM, calls extract_families_from_adversary_log on the configured log,
wraps families in MixedSampler, saves sampler.json, then runs train().

If extract_families returns []: aborts the axis with exit 2 (no retrain).

Usage:
    python -m flip_flop.scripts.run_retrain_tierA \
        --config flip_flop/configs/retrain_tierA_bitmarkov.yaml
    python -m flip_flop.scripts.run_retrain_tierA \
        --config flip_flop/configs/retrain_tierA_writeflip.yaml --test_run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import yaml

from flip_flop.adversary.family import extract_families_from_adversary_log
from flip_flop.adversary.mixture_sampler import MixedSampler
from flip_flop.model import build_model
from flip_flop.train import TrainConfig, train


def _load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
    """Loader that handles both flat and nested config yamls."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    cfg = SimpleNamespace(**flat)
    model = build_model(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[io] loaded {cfg.family} from {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--lstm_ckpt", default="results/flip_flop/lstm/model_final.pt")
    parser.add_argument("--lstm_cfg", default="results/flip_flop/lstm/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    retrain_cfg = raw.get("retrain", {})

    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.test_run:
        cfg.train_steps = 50
        cfg.decay_end_step = 51
        cfg.warmup_steps = 5
        cfg.eval_every = 25
        cfg.save_every = 0
        cfg.eval_in_n = 64
        cfg.eval_sparse_n = 128
        cfg.eval_dense_n = 64

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.device == "auto" else cfg.device

    init_ckpt = retrain_cfg.get("init_from_ckpt", cfg.init_from_ckpt)
    cfg.init_from_ckpt = init_ckpt
    pre_transformer = _load_frozen_model(
        init_ckpt, "results/flip_flop/baseline/config.yaml", device)
    pre_lstm = None
    if os.path.exists(args.lstm_ckpt) and os.path.exists(args.lstm_cfg):
        pre_lstm = _load_frozen_model(args.lstm_ckpt, args.lstm_cfg, device)
    else:
        print(f"[retrain] no LSTM at {args.lstm_ckpt}; pull-back will skip LSTM check")

    base_p_i = retrain_cfg.get("base_p_i", 0.8)
    base_cfg = {"name": "stationary", "T": cfg.seq_len,
                "p_w": (1 - base_p_i) / 2, "p_r": (1 - base_p_i) / 2,
                "bit_p1": 0.5}

    sources = retrain_cfg.get("family_sources", [])
    assert len(sources) == 1, "tierA retrain expects exactly one family_source"
    src = sources[0]
    log_path = src["log"]
    top_k = int(src.get("top_k", 5))
    print(f"[retrain] extracting families from {log_path} (top_k={top_k})")

    families = extract_families_from_adversary_log(
        log_path, base_cfg=base_cfg,
        transformer=pre_transformer, lstm=pre_lstm,
        device=device, top_k=top_k,
    )
    if not families:
        print(f"[retrain] no families extracted from {log_path}; aborting axis")
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "no_families.txt"), "w") as f:
            f.write(f"extract_families_from_adversary_log returned [] for {log_path}\n")
        sys.exit(2)
    print(f"[retrain] extracted {len(families)} families")
    for i, fam in enumerate(families):
        print(f"  family[{i:02d}] {fam.name}: {type(fam).__name__}")

    sampler = MixedSampler(
        T=cfg.seq_len, base_p_i=base_p_i,
        families=families,
        replay_frac=retrain_cfg.get("replay_frac", 0.7),
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "sampler.json"), "w") as f:
        json.dump(sampler.describe(), f, indent=2)
    print(f"[retrain] wrote {os.path.join(cfg.out_dir, 'sampler.json')}")

    # Free pre-retrain model graphs to save GPU mem before training.
    del pre_transformer
    if pre_lstm is not None:
        del pre_lstm
    if device == "cuda":
        torch.cuda.empty_cache()

    result = train(cfg, sampler=sampler)
    print(f"[done] {result}")


if __name__ == "__main__":
    main()
