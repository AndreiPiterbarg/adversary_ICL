"""Materialize a dataset.pt from a finished training run's directory.

Reads <run_dir>/sampler.json (if present) or <run_dir>/config.yaml (else),
reconstructs the same per-sequence sampling distribution, draws --n
sequences with a deterministic RNG, and saves them as a (N, T) LongTensor
to <run_dir>/dataset.pt (overridable with --out).

Designed to run after any training script (run_baseline, run_retrain,
run_retrain_tierA, ...) without modifying it. The resulting dataset.pt
plugs into run_retrain_from_dataset.py.

Sampling logic mirrors MixedSampler exactly:
  * with prob replay_frac: FFL(base_p_i)
  * else: uniform pick over the families list
For each family kind:
  * ClusterFamily / PassthroughFamily: sample directly from `dist`
  * MixtureFamily: with prob (1-alpha) base_dist, else uniform over adv_dists

Usage:
    python -m flip_flop.scripts.dump_dataset --run_dir results/flip_flop/retrain/tierA_bitmarkov --n 32000
    python -m flip_flop.scripts.dump_dataset --run_dir results/flip_flop/baseline --n 10000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import yaml

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.data import sample_ffl


def _sample_family(fam_d: dict, n: int, rng: np.random.Generator) -> torch.LongTensor:
    """Draw n sequences from a single family entry of sampler.json."""
    kind = fam_d["kind"]
    if kind in ("ClusterFamily", "PassthroughFamily"):
        return FFLDistribution.from_dict(fam_d["dist"]).sample(n, rng)
    if kind == "MixtureFamily":
        base = FFLDistribution.from_dict(fam_d["base_dist"])
        advs = [FFLDistribution.from_dict(d) for d in fam_d["adv_dists"]]
        alpha = float(fam_d["alpha"])
        is_adv = rng.random(n) < alpha
        n_adv = int(is_adv.sum())
        n_base = n - n_adv
        out = torch.empty(n, base.T, dtype=torch.long)
        if n_base > 0:
            out[~is_adv] = base.sample(n_base, rng)
        if n_adv > 0:
            adv_idx = np.where(is_adv)[0]
            v_per = rng.integers(0, len(advs), size=n_adv)
            for vi, vd in enumerate(advs):
                m = v_per == vi
                if not m.any():
                    continue
                out[adv_idx[m]] = vd.sample(int(m.sum()), rng)
        return out
    raise ValueError(f"unknown family kind in sampler.json: {kind!r}")


def _sample_mixed(spec: dict, n: int, rng: np.random.Generator) -> torch.LongTensor:
    """MixedSampler-equivalent: replay_frac × FFL(base_p_i) ∪ uniform-over-families."""
    T = int(spec["T"])
    base_p_i = float(spec["base_p_i"])
    replay_frac = float(spec["replay_frac"])
    families = spec["families"]
    assert families, "sampler.json families list is empty"

    is_base = rng.random(n) < replay_frac
    n_base = int(is_base.sum())
    n_adv = n - n_base
    out = torch.empty(n, T, dtype=torch.long)

    if n_base > 0:
        out[is_base] = sample_ffl(T, base_p_i, n_base, rng)
    if n_adv > 0:
        fam_idx = rng.integers(0, len(families), size=n_adv)
        adv_pos = np.where(~is_base)[0]
        for f_i, fam_d in enumerate(families):
            mask = fam_idx == f_i
            if not mask.any():
                continue
            n_f = int(mask.sum())
            out[adv_pos[mask]] = _sample_family(fam_d, n_f, rng)
    return out


def _sample_ffl_only(cfg_yaml_path: str, n: int, rng: np.random.Generator) -> tuple:
    """Fallback for runs without sampler.json (e.g., baseline): use config.yaml's
    train_p_i and seq_len. Returns (tokens, describe_dict)."""
    with open(cfg_yaml_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat.update(raw)
            break
    T = int(flat["seq_len"])
    p_i = float(flat["train_p_i"])
    tokens = sample_ffl(T, p_i, n, rng)
    return tokens, {"kind": "FFLSampler", "T": T, "p_i": p_i}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Training run directory (must contain sampler.json or config.yaml)")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of sequences to materialize")
    parser.add_argument("--out", default=None,
                        help="Output .pt path (default: <run_dir>/dataset.pt)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    sampler_json = os.path.join(args.run_dir, "sampler.json")
    config_yaml = os.path.join(args.run_dir, "config.yaml")
    rng = np.random.default_rng(args.seed)

    if os.path.exists(sampler_json):
        with open(sampler_json) as f:
            spec = json.load(f)
        print(f"[dump] reading {sampler_json}: T={spec['T']} "
              f"replay_frac={spec['replay_frac']} n_families={len(spec['families'])}")
        # Chunk to keep peak memory bounded.
        chunks = []
        n_left = int(args.n)
        while n_left > 0:
            bs = min(1024, n_left)
            chunks.append(_sample_mixed(spec, bs, rng))
            n_left -= bs
        tokens = torch.cat(chunks, dim=0)
        describe = {"kind": "MixedSampler", "T": spec["T"],
                    "base_p_i": spec["base_p_i"],
                    "replay_frac": spec["replay_frac"],
                    "n_families": len(spec["families"])}
    elif os.path.exists(config_yaml):
        print(f"[dump] no sampler.json; falling back to FFL from {config_yaml}")
        tokens, describe = _sample_ffl_only(config_yaml, args.n, rng)
    else:
        raise SystemExit(
            f"neither sampler.json nor config.yaml found in {args.run_dir}"
        )

    out_path = args.out or os.path.join(args.run_dir, "dataset.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {"tokens": tokens, "describe": describe,
         "seq_len": int(tokens.shape[1]), "seed": args.seed,
         "source_run_dir": os.path.abspath(args.run_dir)},
        out_path,
    )
    print(f"[dump] wrote {tuple(tokens.shape)} to {out_path}")


if __name__ == "__main__":
    main()
