"""CMA-ES adversary search over the WriteFlipRate axis (axis A2).

Search space (5 dims):
  x[0:3] -> softmax-3 -> (p_w, p_r, p_i)         # simplex
  x[3]   -> sigmoid    -> bit_p1    in [0, 1]
  x[4]   -> sigmoid    -> flip_rate in [0, 1]    # P(new_write_bit != stored)

Usage:
    python -m flip_flop.scripts.run_adversary_writeflip \
        --config flip_flop/configs/adversary_writeflip.yaml
    python -m flip_flop.scripts.run_adversary_writeflip \
        --config flip_flop/configs/adversary_writeflip.yaml --test_run
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from flip_flop.adversary.distribution import WriteFlipRate
from flip_flop.adversary.objective import fitness
from flip_flop.adversary.search import _sigmoid, _softmax3, cma_search
from flip_flop.model import build_model


def _load_yaml_flat(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
    flat = _load_yaml_flat(cfg_path)
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


class WriteFlipEncoder:
    """Real-vector <-> WriteFlipRate distribution params (5 dims)."""

    def __init__(self, T: int):
        self.T = T
        self.n_dims = 5

    def decode(self, x: np.ndarray) -> WriteFlipRate:
        x = np.asarray(x)
        simplex = _softmax3(x[0:3])
        p_w, p_r, _p_i = simplex
        bit_p1 = float(_sigmoid(np.array([x[3]]))[0])
        flip_rate = float(_sigmoid(np.array([x[4]]))[0])
        return WriteFlipRate(T=self.T, p_w=float(p_w), p_r=float(p_r),
                             bit_p1=bit_p1, flip_rate=flip_rate)

    def random_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.n_dims) * 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    flat = _load_yaml_flat(args.config)
    out_dir = args.out_dir or flat["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(flat, f, sort_keys=False)

    seed = int(flat.get("seed", 0))
    T = int(flat.get("T", 512))
    n = int(flat.get("n", 2000))
    batch_size = int(flat.get("batch_size", 64))
    budget = int(flat.get("budget", 3000))
    pop_size = int(flat.get("pop_size", 16))
    num_restarts = int(flat.get("num_restarts", 3))
    sigma_init = float(flat.get("sigma_init", 0.3))
    lambda_lstm = float(flat.get("lambda_lstm", 10.0))
    lstm_tolerance = float(flat.get("lstm_tolerance", 1e-3))

    if args.test_run:
        budget = 50
        num_restarts = 1
        n = 64
        batch_size = 16
        pop_size = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[adversary-writeflip] device={device} T={T} n={n} budget={budget} "
          f"pop={pop_size} restarts={num_restarts}")

    transformer = _load_frozen_model(
        flat["transformer_ckpt"], flat["transformer_cfg"], device)
    lstm = None
    lstm_ckpt = flat.get("lstm_ckpt")
    lstm_cfg = flat.get("lstm_cfg")
    if lstm_ckpt and lstm_cfg and os.path.exists(lstm_ckpt) and os.path.exists(lstm_cfg):
        lstm = _load_frozen_model(lstm_ckpt, lstm_cfg, device)
    else:
        print("[adversary-writeflip] no LSTM checkpoint; running without LSTM penalty")

    rng = np.random.default_rng(seed)
    objective = partial(
        fitness,
        transformer=transformer, lstm=lstm,
        n=n, batch_size=batch_size, device=device, rng=rng,
        lambda_lstm=lambda_lstm, lstm_tolerance=lstm_tolerance,
    )

    encoder = WriteFlipEncoder(T=T)
    results = cma_search(
        encoder, objective, out_dir,
        budget=budget, pop_size=pop_size, sigma_init=sigma_init,
        num_restarts=num_restarts, seed=seed,
    )

    valid = [r for r in results if r.is_valid]
    valid.sort(key=lambda r: r.fitness, reverse=True)
    best_fit = valid[0].fitness if valid else float("nan")
    best_t = valid[0].T_glitch if valid else float("nan")
    print(f"[done] n_candidates={len(results)} best_fitness={best_fit:.4e} "
          f"best_T_glitch={best_t:.4e}")


if __name__ == "__main__":
    main()
