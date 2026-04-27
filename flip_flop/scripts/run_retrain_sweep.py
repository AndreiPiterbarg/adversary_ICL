"""Sweep retrain: train one model on the union of every family any prior
retrain saw.

Reads each existing sampler.json (the source of truth for what was actually
trained on per axis), reconstructs the Family objects exactly via
`family_from_dict`, concatenates them, and trains one model on the union
with one shared replay_frac. No re-clustering, no re-bisection — the sweep
dataset is exactly the families those retrains saw.

The merged `sampler.json` is augmented with a per-family `source` field
(the path of the sampler.json it came from) so provenance is preserved.

Usage:
    python -m flip_flop.scripts.run_retrain_sweep \
        --config flip_flop/configs/retrain_sweep.yaml
    python -m flip_flop.scripts.run_retrain_sweep \
        --config flip_flop/configs/retrain_sweep.yaml --test_run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

from flip_flop.adversary.family import Family, family_from_dict
from flip_flop.adversary.mixture_sampler import MixedSampler
from flip_flop.train import TrainConfig, train


def _load_families_from_sampler_json(path: str, T: int) -> list[tuple[Family, dict]]:
    """Reconstruct families from a saved sampler.json. Returns
    (family, raw_dict) pairs so the caller can re-emit provenance."""
    with open(path) as f:
        d = json.load(f)
    src_T = d.get("T")
    if src_T is not None and src_T != T:
        raise ValueError(
            f"sampler.json {path}: T={src_T} != sweep T={T} (sequence-length mismatch)"
        )
    out: list[tuple[Family, dict]] = []
    for fam_d in d.get("families", []):
        out.append((family_from_dict(fam_d), fam_d))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--test_run", action="store_true")
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

    init_ckpt = retrain_cfg.get("init_from_ckpt", cfg.init_from_ckpt)
    cfg.init_from_ckpt = init_ckpt

    sampler_sources = retrain_cfg.get("sampler_sources", [])
    assert sampler_sources, "retrain_sweep needs at least one sampler.json in sampler_sources"

    base_p_i = retrain_cfg.get("base_p_i", 0.8)
    replay_frac = retrain_cfg.get("replay_frac", 0.7)

    all_families: list[Family] = []
    provenance: list[dict] = []
    for src in sampler_sources:
        if not os.path.exists(src):
            print(f"[sweep] missing sampler.json: {src}; aborting")
            sys.exit(2)
        loaded = _load_families_from_sampler_json(src, T=cfg.seq_len)
        print(f"[sweep] {src}: loaded {len(loaded)} families")
        for fam, fam_d in loaded:
            all_families.append(fam)
            entry = dict(fam_d)
            entry["source"] = src
            provenance.append(entry)

    assert all_families, "no families collected across sources"
    print(f"[sweep] total: {len(all_families)} families from {len(sampler_sources)} source(s)")
    for i, fam in enumerate(all_families):
        print(f"  family[{i:02d}] axis={fam.axis or '?':<12} "
              f"{type(fam).__name__:<16} {fam.name}")

    sampler = MixedSampler(
        T=cfg.seq_len, base_p_i=base_p_i,
        families=all_families, replay_frac=replay_frac,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    merged = sampler.describe()
    merged["sources"] = list(sampler_sources)
    merged["families"] = provenance
    with open(os.path.join(cfg.out_dir, "sampler.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[sweep] wrote {os.path.join(cfg.out_dir, 'sampler.json')}")

    result = train(cfg, sampler=sampler)
    print(f"[done] {result}")


if __name__ == "__main__":
    main()
