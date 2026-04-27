"""Adversary entry point: parse config, load frozen models, run search."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
import yaml

from . import distribution as dist_mod
from .io import dump_final_eval, load_frozen_model, save_top_k
from .objective import fitness
from .search import (EvalResult, PiecewiseEncoder, cma_search, grid_search,
                     save_checkpoint)


@dataclass
class AdversaryConfig:
    # strategy
    strategy: str = "grid"               # "grid" | "cma" | "planted"
    # frozen models
    transformer_ckpt: str = "results/flip_flop/baseline/model_final.pt"
    transformer_cfg: str = "flip_flop/configs/baseline.yaml"
    lstm_ckpt: str = "results/flip_flop/lstm/model_final.pt"
    lstm_cfg: str = "flip_flop/configs/lstm.yaml"
    use_lstm: bool = True
    # distribution
    dist_name: str = "stationary"
    T: int = 512
    # grid / planted
    param_grid: dict = field(default_factory=dict)
    template: str = "gap"
    filler_p_i: float = 1.0
    # cma
    K_segments: int = 4
    budget: int = 3000
    pop_size: int = 16
    sigma_init: float = 0.3
    num_restarts: int = 3
    # search / eval
    search_n: int = 2000
    final_eval_n: int = 100_000
    eval_batch_size: int = 64
    n_final_seeds: int = 3
    lambda_lstm: float = 10.0
    lstm_tolerance: float = 1e-3
    # output
    out_dir: str = "results/flip_flop/adversary/run"
    top_k: int = 25
    seed: int = 0
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "AdversaryConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat: dict[str, Any] = {}
        for v in raw.values():
            if isinstance(v, dict):
                flat.update(v)
        # Map yaml keys to dataclass field names (yaml may use nested-friendly names).
        # Accept both {strategy: {type: grid}} and flat {strategy: grid}.
        if "type" in flat and "strategy" not in flat:
            flat["strategy"] = flat.pop("type")
        elif isinstance(flat.get("strategy"), dict):
            flat["strategy"] = flat["strategy"]["type"]
        if "name" in flat and "dist_name" not in flat:
            flat["dist_name"] = flat.pop("name")
        # Drop unknown keys rather than erroring, to keep yaml schemas loose.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        flat = {k: v for k, v in flat.items() if k in known}
        return cls(**flat)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _planted_factory(cfg: AdversaryConfig):
    """Returns a factory that builds Planted dists from a {template, params} dict."""
    def factory(params: dict) -> dist_mod.Planted:
        return dist_mod.Planted(
            T=cfg.T, template=cfg.template, filler_p_i=cfg.filler_p_i,
            params=params,
        )
    return factory


def _generic_factory(cfg: AdversaryConfig):
    """Returns a factory that builds dists of cfg.dist_name from a params dict."""
    cls = dist_mod.REGISTRY[cfg.dist_name]
    def factory(params: dict):
        return cls(T=cfg.T, **params)
    return factory


def run_adversary(cfg: AdversaryConfig):
    device = _resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

    transformer = load_frozen_model(cfg.transformer_ckpt, cfg.transformer_cfg, device)
    lstm = load_frozen_model(cfg.lstm_ckpt, cfg.lstm_cfg, device) if cfg.use_lstm else None

    rng = np.random.default_rng(cfg.seed)
    objective = partial(
        fitness,
        transformer=transformer, lstm=lstm,
        n=cfg.search_n, batch_size=cfg.eval_batch_size, device=device,
        rng=rng, lambda_lstm=cfg.lambda_lstm, lstm_tolerance=cfg.lstm_tolerance,
    )

    if cfg.strategy == "grid":
        results = grid_search(_generic_factory(cfg), cfg.param_grid, objective, cfg.out_dir)
    elif cfg.strategy == "planted":
        results = grid_search(_planted_factory(cfg), cfg.param_grid, objective, cfg.out_dir)
    elif cfg.strategy == "cma":
        encoder = PiecewiseEncoder(T=cfg.T, K=cfg.K_segments)
        results = cma_search(
            encoder, objective, cfg.out_dir,
            budget=cfg.budget, pop_size=cfg.pop_size, sigma_init=cfg.sigma_init,
            num_restarts=cfg.num_restarts, seed=cfg.seed,
        )
    else:
        raise ValueError(f"unknown strategy {cfg.strategy!r}")

    save_checkpoint(results, cfg.out_dir)
    top = save_top_k(results, cfg.top_k, cfg.out_dir)
    dump_final_eval(
        top, transformer, lstm,
        n=cfg.final_eval_n, batch_size=cfg.eval_batch_size, device=device,
        n_seeds=cfg.n_final_seeds, out_dir=cfg.out_dir,
        lambda_lstm=cfg.lambda_lstm, lstm_tolerance=cfg.lstm_tolerance,
    )
    return {"n_candidates": len(results),
            "best_search_fitness": max((r.fitness for r in results if r.is_valid),
                                       default=float("nan"))}
