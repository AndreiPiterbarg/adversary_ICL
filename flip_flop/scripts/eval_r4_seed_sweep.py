"""Step 2A: data-seed robustness sweep on the existing 3 trained models.

Re-runs eval_r4_on_families's loader+sampler+evaluator with N=50000 per family
and 5 different EVAL_SEEDs in {7001..7005}. Reports mean +/- std per (model,
family) pair across the 5 seeds.

Outputs under results/flip_flop/liu_r4/diagnostic_step2a/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from flip_flop.eval import evaluate_dataset
from flip_flop.scripts.eval_r4_on_families import (
    MODELS,
    SAMPLER_JSON,
    load_frozen_model,
    sample_from_family,
)

OUT_DIR = "results/flip_flop/liu_r4/diagnostic_step2a"
N_SAMPLES = 50000
EVAL_SEEDS = (7001, 7002, 7003, 7004, 7005)
BATCH_SIZE = 64


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step2a] device={device} n_samples={N_SAMPLES} seeds={EVAL_SEEDS}")

    with open(SAMPLER_JSON) as f:
        sampler_spec = json.load(f)
    families = sampler_spec["families"]
    fam_names = [fam["name"] for fam in families]

    # results[model][family] = list of glitch values across seeds
    results: dict[str, dict[str, list[float]]] = {
        m: {fn: [] for fn in fam_names} for m in MODELS
    }

    # Load each model once, then iterate seeds inner-loop. (Models are big and
    # disjoint; we want to avoid loading 5x per model.)
    for model_name, (ckpt, cfg_path) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"[step2a] SKIP {model_name}: missing {ckpt}")
            continue
        print(f"\n[step2a] === model={model_name} ===")
        model = load_frozen_model(ckpt, cfg_path, device)
        for seed in EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            print(f"  seed={seed}")
            for fam in families:
                toks = sample_from_family(fam, N_SAMPLES, rng).to(device)
                res = evaluate_dataset(model, toks, batch_size=BATCH_SIZE, device=device)
                err = res["error_rate"]
                results[model_name][fam["name"]].append(err)
                print(f"    {fam['name']:<32} glitch={err:.4%} "
                      f"({res['num_errors']}/{res['num_predictions']})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary stats: mean / std / min / max per (model, family).
    summary = {m: {fn: {} for fn in fam_names} for m in results}
    for m, by_fam in results.items():
        for fn, vals in by_fam.items():
            arr = np.array(vals, dtype=float)
            summary[m][fn] = {
                "values": [float(v) for v in vals],
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    out = {
        "n_samples": N_SAMPLES,
        "eval_seeds": list(EVAL_SEEDS),
        "sampler_source": SAMPLER_JSON,
        "summary": summary,
    }
    with open(os.path.join(OUT_DIR, "seed_sweep_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Pretty txt summary + pass-criterion.
    lines = []
    lines.append(f"Step 2A data-seed sweep (n={N_SAMPLES}, seeds={list(EVAL_SEEDS)})")
    lines.append("")
    header = f"{'family':<35} | " + " | ".join(f"{m:<25}" for m in summary)
    lines.append(header)
    lines.append("-" * len(header))
    for fn in fam_names:
        row = f"{fn:<35} | "
        cells = []
        for m in summary:
            s = summary[m][fn]
            cells.append(f"{s['mean']:>8.4%} ± {s['std']:>7.4%}")
        row += " | ".join(f"{c:<25}" for c in cells)
        lines.append(row)
    lines.append("")
    # Pass criterion (R4, piecewise_c00):
    # mean_R4 >= 5 * std_R4  AND  mean_R4 >= 10 * mean_R2v2
    target = "piecewise_c00_a1.00"
    if "R4" in summary and target in summary["R4"]:
        s_r4 = summary["R4"][target]
        s_r2 = summary["R2-redone-v2"][target] if "R2-redone-v2" in summary else None
        cond1 = s_r4["std"] > 0 and s_r4["mean"] >= 5 * s_r4["std"]
        cond2 = (s_r2 is not None
                 and s_r2["mean"] > 0
                 and s_r4["mean"] >= 10 * s_r2["mean"])
        cond_zero_std = s_r4["std"] == 0 and s_r4["mean"] > 0
        lines.append(f"PASS CRITERION (piecewise_c00):")
        lines.append(f"  R4 mean = {s_r4['mean']:.4%}, std = {s_r4['std']:.4%}")
        if s_r2 is not None:
            lines.append(f"  R2-v2 mean = {s_r2['mean']:.4%}")
        lines.append(f"  R4 mean >= 5 * std? {cond1 or cond_zero_std} "
                     f"(or std==0 and mean>0)")
        if s_r2 is not None:
            ratio = s_r4["mean"] / max(s_r2["mean"], 1e-12)
            lines.append(f"  R4 mean >= 10 * R2-v2 mean? {cond2} (ratio={ratio:.1f}x)")
        ok = (cond1 or cond_zero_std) and cond2
        lines.append(f"  PASS: {ok}")
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[step2a] wrote {OUT_DIR}/seed_sweep_results.json and summary.txt")


if __name__ == "__main__":
    main()
