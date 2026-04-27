"""Decisive evaluation: per axis, eval baseline / R4_seed0 / tierA_<axis> on
the tierA retrain's families (saved in sampler.json).

For each axis (bitmarkov, writeflip):
  1. Load results/flip_flop/retrain/tierA_<axis>/sampler.json
  2. Sample N=10000 sequences per family with eval_seed=12345
  3. Score baseline, R4_seed0, tierA_<axis> on identical sequences
  4. Save results/flip_flop/tierA_results/<axis>_vs_r4.json

If a per-axis tierA dir is missing (no breakthrough or skipped), the axis is
skipped with a logged note.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.eval import evaluate_dataset
from flip_flop.model import build_model


N_SAMPLES = 10000
EVAL_SEED = 12345
BATCH_SIZE = 64

AXES = [
    ("bitmarkov", "results/flip_flop/retrain/tierA_bitmarkov"),
    ("writeflip", "results/flip_flop/retrain/tierA_writeflip"),
]
OUT_DIR = "results/flip_flop/tierA_results"
BASELINE = ("results/flip_flop/baseline/model_final.pt",
            "results/flip_flop/baseline/config.yaml")
R4_SEED0 = ("results/flip_flop/liu_r4/model_final.pt",
            "results/flip_flop/liu_r4/config.yaml")


def load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
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


def sample_from_family(fam_dict: dict, n: int, rng: np.random.Generator) -> torch.LongTensor:
    """Reconstruct a family from its saved dict and sample n sequences."""
    kind = fam_dict["kind"]
    if kind == "ClusterFamily":
        dist = FFLDistribution.from_dict(fam_dict["dist"])
        return dist.sample(n, rng)
    if kind == "PassthroughFamily":
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


def evaluate_axis(axis_name: str, retrain_dir: str, device: str) -> dict | None:
    sampler_path = os.path.join(retrain_dir, "sampler.json")
    if not os.path.exists(sampler_path):
        print(f"[eval] SKIP axis '{axis_name}': sampler missing at {sampler_path}")
        return None
    with open(sampler_path) as f:
        spec = json.load(f)
    families = spec["families"]
    print(f"\n[eval] === axis={axis_name}: {len(families)} families from {sampler_path} ===")

    rng = np.random.default_rng(EVAL_SEED)
    fam_tokens: dict[str, torch.Tensor] = {}
    for fam in families:
        fam_tokens[fam["name"]] = sample_from_family(fam, N_SAMPLES, rng)
        print(f"  sampled {N_SAMPLES} sequences from {fam['name']} "
              f"(axis={fam.get('axis', '')})")

    tierA_ckpt = os.path.join(retrain_dir, "model_final.pt")
    tierA_cfg = os.path.join(retrain_dir, "config.yaml")
    models = {
        "baseline": BASELINE,
        "R4_seed0": R4_SEED0,
        f"tierA_{axis_name}": (tierA_ckpt, tierA_cfg),
    }

    results: dict[str, dict[str, float]] = {}
    for mname, (ckpt, cfgp) in models.items():
        if not os.path.exists(ckpt):
            print(f"[eval] SKIP model {mname}: missing {ckpt}")
            continue
        if not os.path.exists(cfgp):
            print(f"[eval] SKIP model {mname}: missing {cfgp}")
            continue
        print(f"\n[eval] -- {mname} --")
        model = load_frozen_model(ckpt, cfgp, device)
        results[mname] = {}
        for fam in families:
            toks = fam_tokens[fam["name"]].to(device)
            res = evaluate_dataset(model, toks, batch_size=BATCH_SIZE, device=device)
            err = res["error_rate"]
            results[mname][fam["name"]] = err
            print(f"  {fam['name']:<40} glitch={err:.4%} "
                  f"({res['num_errors']}/{res['num_predictions']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {
        "axis": axis_name,
        "n_samples": N_SAMPLES,
        "eval_seed": EVAL_SEED,
        "batch_size": BATCH_SIZE,
        "families": families,
        "results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{axis_name}_vs_r4.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {out_path}")

    print("\n" + "=" * 100)
    cols = list(results.keys())
    print(f"{'family':<42} | " + " | ".join(f"{m:<18}" for m in cols))
    print("-" * 100)
    for fam in families:
        row = f"{fam['name']:<42} | "
        row += " | ".join(
            f"{results[m].get(fam['name'], float('nan')):>17.4%}"
            for m in cols
        )
        print(row)
    print("=" * 100)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for axis_name, retrain_dir in AXES:
        evaluate_axis(axis_name, retrain_dir, device)


if __name__ == "__main__":
    main()
