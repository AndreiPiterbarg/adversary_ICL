"""Run one round of the cumulative retrain loop.

Workflow:
  1. Load families from each adversary log listed in cfg.family_sources.
  2. Build a MixedSampler (50% base / 50% uniform-over-families, per sequence).
  3. Continue training the Transformer from cfg.init_from_ckpt for
     cfg.train_steps at cfg.lr (LOW — default 3e-5).
  4. Evaluate on a 5-way battery:
       - FFL(0.8) in-distribution (forgetting metric)
       - FFL(0.98) sparse tail (paper)
       - FFL(0.1) dense tail (paper)
       - Fresh samples from each family (robustness — transformer + LSTM)

Multi-round orchestration is manual: pull results, extract new adversary,
edit the yaml's init_from_ckpt + family_sources, re-run.

Usage:
    python -m flip_flop.scripts.run_retrain --config flip_flop/configs/retrain.yaml
    python -m flip_flop.scripts.run_retrain --config ... --test_run
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from flip_flop.adversary.family import (Family, _select_with_axis_floor,
                                         extract_families_from_adversary_log)
from flip_flop.adversary.io import load_frozen_model
from flip_flop.adversary.mixture_sampler import MixedSampler
from flip_flop.data import make_eval_dataset
from flip_flop.eval import evaluate_dataset
from flip_flop.train import TrainConfig, train

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "retrain.yaml",
)


def _load_families(
    family_sources: list[dict],
    base_cfg: dict,
    transformer,
    lstm,
    device: str,
    global_top_k: int = 5,
) -> list[Family]:
    """Extract families from each source, then apply per-axis floor + global
    rank ACROSS sources (not per-source). This is what the Phase C plan calls
    for: axis representation is enforced at the final mixture, not per log.

    Each source's `top_k` is treated as a *source cap* (max candidates from
    that source pre-aggregation). After all sources, `_select_with_axis_floor`
    picks `global_top_k` total with the per-axis floor.
    """
    pooled: list[Family] = []
    for src in family_sources:
        pooled.extend(extract_families_from_adversary_log(
            src["log"],
            base_cfg=base_cfg,
            transformer=transformer,
            lstm=lstm,
            device=device,
            top_k=int(src.get("top_k", 10)),  # source cap (pre-aggregation)
        ))
    assert pooled, "family_sources produced zero families"
    # Global axis-floor + rank across all sources combined.
    return _select_with_axis_floor(pooled, top_k=global_top_k)


def _eval_battery(
    model, lstm, families: list[Family], cfg: TrainConfig, device: str,
    n_family: int = 5000,
):
    """Return dict of eval metrics on 5 targets."""
    results = {}

    # 1-3. Fixed FFL(p) test sets (paper's three).
    for name, p_i, n in [
        ("ffl_0.8_in",    cfg.eval_in_p_i,     cfg.eval_in_n),
        ("ffl_0.98_sparse", cfg.eval_sparse_p_i, cfg.eval_sparse_n),
        ("ffl_0.1_dense", cfg.eval_dense_p_i,   cfg.eval_dense_n),
    ]:
        ds = make_eval_dataset(p_i, n, cfg.seq_len, seed=cfg.eval_seed)
        res = evaluate_dataset(model, ds, batch_size=cfg.eval_batch_size, device=device)
        results[name] = {"error_rate": res["error_rate"],
                         "num_predictions": res["num_predictions"]}

    # 4. Each family: Transformer + LSTM glitch rate on fresh samples.
    rng = np.random.default_rng(cfg.eval_seed + 100)
    for i, fam in enumerate(families):
        tokens = fam.sample(n_family, rng)
        t_res = evaluate_dataset(model, tokens, batch_size=cfg.eval_batch_size, device=device)
        if lstm is not None:
            l_res = evaluate_dataset(lstm, tokens, batch_size=cfg.eval_batch_size, device=device)
            lstm_err = l_res["error_rate"]
        else:
            lstm_err = None
        results[f"family_{i:02d}"] = {
            "name": fam.name,
            "family_dict": fam.to_dict(),
            "T_glitch": t_res["error_rate"],
            "lstm_glitch": lstm_err,
            "num_predictions": t_res["num_predictions"],
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--lstm_ckpt", default="results/flip_flop/lstm/model_final.pt")
    parser.add_argument("--lstm_cfg", default="flip_flop/configs/lstm.yaml")
    args = parser.parse_args()

    # Load the full yaml so we can get `retrain:` section separately.
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    retrain_cfg = raw.get("retrain", {})

    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.test_run:
        cfg.train_steps = 50
        cfg.decay_end_step = 51
        cfg.warmup_steps = 5
        cfg.eval_in_n = 64
        cfg.eval_sparse_n = 128
        cfg.eval_dense_n = 64
        cfg.eval_every = 25
        cfg.save_every = 0

    # Load pre-retrain models for family extraction (pull-back α requires both).
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    pre_transformer = load_frozen_model(
        retrain_cfg.get("init_from_ckpt", cfg.init_from_ckpt),
        "flip_flop/configs/baseline.yaml",
        device,
    )
    pre_lstm = load_frozen_model(args.lstm_ckpt, args.lstm_cfg, device) \
        if os.path.exists(args.lstm_ckpt) else None
    base_cfg = {"name": "stationary", "T": cfg.seq_len,
                "p_w": (1 - retrain_cfg.get("base_p_i", 0.8)) / 2,
                "p_r": (1 - retrain_cfg.get("base_p_i", 0.8)) / 2,
                "bit_p1": 0.5}

    families = _load_families(
        retrain_cfg["family_sources"], base_cfg, pre_transformer, pre_lstm, device,
    )
    print(f"[retrain] loaded {len(families)} families from {len(retrain_cfg['family_sources'])} sources")
    for i, f in enumerate(families):
        print(f"  family[{i:02d}] {f.name}: {type(f).__name__}")

    # Build MixedSampler.
    sampler = MixedSampler(
        T=cfg.seq_len,
        base_p_i=retrain_cfg.get("base_p_i", 0.8),
        families=families,
        replay_frac=retrain_cfg.get("replay_frac", 0.5),
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "sampler.json"), "w") as f:
        json.dump(sampler.describe(), f, indent=2)

    # Continue training.
    result = train(cfg, sampler=sampler)
    print(f"[retrain] train done: last_loss={result['last_loss']:.4f}")

    # Eval battery.
    ckpt = os.path.join(cfg.out_dir, "model_final.pt")
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    from flip_flop.model import build_model
    model = build_model(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()

    lstm = None
    if args.lstm_ckpt and os.path.exists(args.lstm_ckpt):
        lstm = load_frozen_model(args.lstm_ckpt, args.lstm_cfg, device)

    battery = _eval_battery(model, lstm, families, cfg, device,
                            n_family=128 if args.test_run else 5000)
    with open(os.path.join(cfg.out_dir, "eval_battery.json"), "w") as f:
        json.dump(battery, f, indent=2)

    print()
    print("===== EVAL BATTERY =====")
    for name, r in battery.items():
        if name.startswith("ffl_"):
            print(f"  {name:<22}  err={r['error_rate']:.5f}  N={r['num_predictions']}")
        else:
            lstm_str = f"{r['lstm_glitch']:.5f}" if r['lstm_glitch'] is not None else "   -   "
            print(f"  {name:<22}  T_glitch={r['T_glitch']:.5f}  lstm={lstm_str}  ({r['name']})")
    print("=========================")


if __name__ == "__main__":
    main()
