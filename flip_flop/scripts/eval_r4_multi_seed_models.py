"""Step 2B eval: evaluate the two new R4 models (seed=1, seed=2) on the
5 R2-v2 families, plus re-run R4 seed=0 for an apples-to-apples row.

Reuses load_frozen_model + sample_from_family from eval_r4_on_families.

Outputs under results/flip_flop/liu_r4/diagnostic_step2b/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from flip_flop.eval import evaluate_dataset
from flip_flop.scripts.eval_r4_on_families import (
    SAMPLER_JSON,
    load_frozen_model,
    sample_from_family,
)

OUT_DIR = "results/flip_flop/liu_r4/diagnostic_step2b"
N_SAMPLES = 10000
EVAL_SEED = 9999  # same as the original eval_r4_on_families (matches CONTEXT table)
BATCH_SIZE = 64

MODELS = {
    "R4_seed0": ("results/flip_flop/liu_r4/model_final.pt",
                 "results/flip_flop/liu_r4/config.yaml"),
    "R4_seed1": ("results/flip_flop/liu_r4_seed1/model_final.pt",
                 "results/flip_flop/liu_r4_seed1/config.yaml"),
    "R4_seed2": ("results/flip_flop/liu_r4_seed2/model_final.pt",
                 "results/flip_flop/liu_r4_seed2/config.yaml"),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step2b] device={device} n_samples={N_SAMPLES} eval_seed={EVAL_SEED}")

    with open(SAMPLER_JSON) as f:
        sampler_spec = json.load(f)
    families = sampler_spec["families"]
    fam_names = [fam["name"] for fam in families]

    # Pre-sample once so all three R4 seeds see identical sequences.
    rng = np.random.default_rng(EVAL_SEED)
    fam_tokens = {}
    for fam in families:
        toks = sample_from_family(fam, N_SAMPLES, rng)
        fam_tokens[fam["name"]] = toks
        print(f"  sampled {N_SAMPLES} seqs from {fam['name']} (axis={fam['axis']})")

    # Sanity: check that each new R4 model is at ~0% on standard FFL battery.
    # We rely on the eval_log.jsonl that train.py wrote at step 10000 — read those.
    print("\n[step2b] reading standard FFL battery from eval_log.jsonl ...")
    battery = {}
    for model_name, (ckpt, cfg) in MODELS.items():
        log_path = os.path.join(os.path.dirname(ckpt), "eval_log.jsonl")
        last = None
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    last = json.loads(line)
        if last is None:
            print(f"  {model_name}: NO eval_log found")
            battery[model_name] = None
            continue
        battery[model_name] = {
            "step": last["step"],
            "in_distribution": last["in_distribution"]["error_rate"],
            "sparse_tail": last["sparse_tail"]["error_rate"],
            "dense_tail": last["dense_tail"]["error_rate"],
        }
        print(f"  {model_name} step={last['step']}: "
              f"in={last['in_distribution']['error_rate']:.4%} "
              f"sparse(0.98)={last['sparse_tail']['error_rate']:.4%} "
              f"dense(0.1)={last['dense_tail']['error_rate']:.4%}")

    # Family eval per model.
    results = {}
    for model_name, (ckpt, cfg) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"[step2b] SKIP {model_name}: missing {ckpt}")
            continue
        print(f"\n[step2b] === {model_name} ===")
        model = load_frozen_model(ckpt, cfg, device)
        results[model_name] = {}
        for fam in families:
            toks = fam_tokens[fam["name"]].to(device)
            res = evaluate_dataset(model, toks, batch_size=BATCH_SIZE, device=device)
            err = res["error_rate"]
            results[model_name][fam["name"]] = err
            print(f"  {fam['name']:<32} glitch={err:.4%} "
                  f"({res['num_errors']}/{res['num_predictions']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {
        "n_samples": N_SAMPLES,
        "eval_seed": EVAL_SEED,
        "sampler_source": SAMPLER_JSON,
        "results": results,
        "ffl_battery": battery,
        "family_axes": {fam["name"]: fam["axis"] for fam in families},
    }
    with open(os.path.join(OUT_DIR, "r4_multi_seed_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary text + pass criterion.
    lines = []
    lines.append(f"Step 2B R4 multi-seed eval (n={N_SAMPLES}, eval_seed={EVAL_SEED})")
    lines.append("")
    lines.append("Standard FFL battery (from each model's eval_log.jsonl, step=10000):")
    for m, b in battery.items():
        if b is None:
            lines.append(f"  {m}: MISSING")
            continue
        lines.append(f"  {m}: in={b['in_distribution']:.4%} "
                     f"sparse(0.98)={b['sparse_tail']:.4%} "
                     f"dense(0.1)={b['dense_tail']:.4%}")
    lines.append("")
    header = f"{'family':<35} | " + " | ".join(f"{m:<14}" for m in results)
    lines.append(header)
    lines.append("-" * len(header))
    for fn in fam_names:
        row = f"{fn:<35} | " + " | ".join(
            f"{results[m].get(fn, float('nan')):>13.4%}" for m in results
        )
        lines.append(row)
    lines.append("")
    # Pass criterion: all three R4 seeds glitch >= 0.5% on piecewise_c00 AND
    # ~0% on the other piecewise families.
    target = "piecewise_c00_a1.00"
    other_pieces = ["piecewise_c01_a1.00", "piecewise_c02_a1.00"]
    if all(m in results for m in MODELS):
        c00_vals = [results[m][target] for m in MODELS]
        all_above = all(v >= 0.005 for v in c00_vals)
        others_clean = all(
            results[m][p] < 0.005 for m in MODELS for p in other_pieces
        )
        lines.append("PASS CRITERION (Step 2B):")
        for m in MODELS:
            lines.append(f"  {m} on {target}: {results[m][target]:.4%}")
        lines.append(f"  All three R4 seeds >= 0.5% on piecewise_c00? {all_above}")
        lines.append(f"  All three R4 seeds < 0.5% on c01 and c02?    {others_clean}")
        lines.append(f"  PASS: {all_above and others_clean}")
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[step2b] wrote {OUT_DIR}/r4_multi_seed_results.json + summary.txt")


if __name__ == "__main__":
    main()
