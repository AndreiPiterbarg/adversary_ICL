"""Decisive experiment: evaluate Liu R4's model on R2-redone-v2 adversarial families.

Question: does R4's hand-crafted stationary mixture also close our adversary's
piecewise + planted families, or does it miss them?

If R4 also nukes them -> our automated method just rederives R4.
If R4 is still high on piecewise -> the piecewise discovery is the contribution.

Reports: glitch rate per family, for {baseline, R2-redone-v2, R4}, with seeds
disjoint from training/selection.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.eval import evaluate_dataset
from flip_flop.model import build_model
import yaml
from types import SimpleNamespace


def load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
    """Loader that handles both flat and nested config yamls."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
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

SAMPLER_JSON = "results_tier1_v2/results/flip_flop/retrain/round_2_redone/sampler.json"

MODELS = {
    "baseline":     ("results/flip_flop/baseline/model_final.pt",
                     "results/flip_flop/baseline/config.yaml"),
    "R2-redone-v2": ("results_tier1_v2/results/flip_flop/retrain/round_2_redone/model_final.pt",
                     "results_tier1_v2/results/flip_flop/retrain/round_2_redone/config.yaml"),
    "R4":           ("results/flip_flop/liu_r4/model_final.pt",
                     "results/flip_flop/liu_r4/config.yaml"),
}

N_SAMPLES = 10000
EVAL_SEED = 9999  # disjoint from training (seed=0,100,200) and selection (1,2,3)
BATCH_SIZE = 64


def sample_from_family(fam_dict: dict, n: int, rng: np.random.Generator) -> torch.LongTensor:
    """Reconstruct a family from its saved dict and sample n sequences."""
    kind = fam_dict["kind"]
    if kind == "ClusterFamily":
        dist = FFLDistribution.from_dict(fam_dict["dist"])
        return dist.sample(n, rng)
    if kind == "MixtureFamily":
        base = FFLDistribution.from_dict(fam_dict["base_dist"])
        advs = [FFLDistribution.from_dict(d) for d in fam_dict["adv_dists"]]
        alpha = fam_dict["alpha"]
        is_adv = rng.random(n) < alpha
        n_adv = int(is_adv.sum())
        n_base = n - n_adv
        T = base.T
        out = torch.empty(n, T, dtype=torch.long)
        if n_base > 0:
            out[~is_adv] = base.sample(n_base, rng)
        if n_adv > 0:
            adv_idx = np.where(is_adv)[0]
            variant = rng.integers(0, len(advs), size=n_adv)
            for v_i, v_dist in enumerate(advs):
                m = variant == v_i
                k = int(m.sum())
                if k:
                    out[adv_idx[m]] = v_dist.sample(k, rng)
        return out
    raise ValueError(f"unknown family kind: {kind}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    with open(SAMPLER_JSON) as f:
        sampler_spec = json.load(f)
    families = sampler_spec["families"]
    print(f"[eval] {len(families)} families from {SAMPLER_JSON}")

    # Pre-sample all families ONCE so every model sees identical sequences.
    rng = np.random.default_rng(EVAL_SEED)
    fam_tokens = {}
    for fam in families:
        toks = sample_from_family(fam, N_SAMPLES, rng)
        fam_tokens[fam["name"]] = toks
        print(f"  sampled {N_SAMPLES} seqs from {fam['name']} (axis={fam['axis']})")

    # Load each model and evaluate.
    results: dict[str, dict[str, float]] = {}
    for model_name, (ckpt, cfg) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"[eval] SKIP {model_name}: missing {ckpt}")
            continue
        print(f"\n[eval] === {model_name} ===")
        model = load_frozen_model(ckpt, cfg, device)
        results[model_name] = {}
        for fam in families:
            toks = fam_tokens[fam["name"]].to(device)
            res = evaluate_dataset(model, toks, batch_size=BATCH_SIZE, device=device)
            err = res["error_rate"]
            results[model_name][fam["name"]] = err
            print(f"  {fam['name']:<32} glitch={err:.4%} "
                  f"({res['num_errors']}/{res['num_predictions']})")
        # free
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Pretty table.
    print("\n" + "=" * 90)
    print(f"{'family':<35} | {'axis':<10} | " + " | ".join(f"{m:<14}" for m in results))
    print("-" * 90)
    for fam in families:
        row = f"{fam['name']:<35} | {fam['axis']:<10} | "
        row += " | ".join(f"{results[m].get(fam['name'], float('nan')):>13.4%}"
                          for m in results)
        print(row)
    print("=" * 90)

    out_path = "results/flip_flop/liu_r4/r4_vs_families.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_samples": N_SAMPLES,
            "eval_seed": EVAL_SEED,
            "sampler_source": SAMPLER_JSON,
            "results": results,
            "family_axes": {fam["name"]: fam["axis"] for fam in families},
        }, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
