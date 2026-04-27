"""Load frozen models and persist adversary outputs."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ..model import build_model
from .distribution import FFLDistribution
from .objective import seed_averaged_fitness
from .search import EvalResult


def _load_cfg(cfg_path: str) -> SimpleNamespace:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    return SimpleNamespace(**flat)


def load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
    """Build a model from its training config yaml, load weights, freeze."""
    cfg = _load_cfg(cfg_path)
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


def save_top_k(results: list[EvalResult], k: int, out_dir: str, filename: str = "top_k.jsonl"):
    """Write the top-K results by fitness to JSONL."""
    os.makedirs(out_dir, exist_ok=True)
    valid = [r for r in results if r.is_valid]
    valid.sort(key=lambda r: r.fitness, reverse=True)
    top = valid[:k]
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        for r in top:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"[io] saved top-{len(top)} to {path}")
    return top


def dump_final_eval(
    top: list[EvalResult],
    transformer,
    lstm,
    *,
    n: int,
    batch_size: int,
    device: str,
    n_seeds: int = 3,
    out_dir: str,
    lambda_lstm: float = 10.0,
    lstm_tolerance: float = 1e-3,
):
    """Re-evaluate top-K at larger N with seed averaging, write final_eval.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_eval.jsonl")
    with open(path, "w") as f:
        for i, r in enumerate(top):
            dist = FFLDistribution.from_dict(r.config)
            fr = seed_averaged_fitness(
                dist, transformer, lstm,
                n=n, batch_size=batch_size, device=device, n_seeds=n_seeds,
                lambda_lstm=lambda_lstm, lstm_tolerance=lstm_tolerance,
            )
            print(f"  [final_eval {i + 1}/{len(top)}] "
                  f"fit={fr.fitness:.4f} T_glitch={fr.T_glitch:.4f} "
                  f"lstm={fr.lstm_glitch:.4f}")
            out = {
                "config": r.config,
                "descriptor": r.descriptor,
                "search_fitness": r.fitness,
                "final_fitness": fr.fitness,
                "final_T_glitch": fr.T_glitch,
                "final_lstm_glitch": fr.lstm_glitch,
                "n_samples": fr.n_samples,
            }
            f.write(json.dumps(out) + "\n")
            f.flush()
    print(f"[io] wrote final_eval to {path}")
