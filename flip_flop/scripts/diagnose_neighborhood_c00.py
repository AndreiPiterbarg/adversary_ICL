"""Step 3: neighborhood escalation around piecewise_c00.

Per-segment jitter of (p_w, p_r, bit_p1) one segment at a time:
  - p_w by multiplicative factors {0.8, 0.9, 1.0, 1.1, 1.2}
  - p_r by multiplicative factors {0.8, 0.9, 1.0, 1.1, 1.2}
  - bit_p1 by additive deltas {-0.1, 0.0, +0.1}

K=4 segments x 75 jitters/seg = 300 candidates. Capped at MAX_CONFIGS=100 by
random subsample (was 200 in v1; reduced for speed). N=5000 seqs/config (was
10000); still ~50K reads per config so the noise floor is ~0.04% — well below
our 0.5% threshold.

Per config: build dist, sample, evaluate ALL 3 models, free CPU tensor before
moving on. (v1 held all 200x10000 token tensors in CPU RAM = ~8 GB which
caused swapping on this 16 GB box.)

CRITICAL VALIDITY CHECK: each jittered config produces a valid FFL
distribution (verified by walking the read-determinism on first 100 sequences).

Outputs under results/flip_flop/liu_r4/diagnostic_step3/.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution, Piecewise
from flip_flop.adversary.family import _clip_simplex2, _clip01
from flip_flop.data import W, R, ZERO, ONE
from flip_flop.eval import evaluate_dataset
from flip_flop.scripts.eval_r4_on_families import MODELS, SAMPLER_JSON, load_frozen_model

OUT_DIR = "results/flip_flop/liu_r4/diagnostic_step3"
N_SAMPLES = 2000
BATCH_SIZE = 64
MAX_CONFIGS = 60
TARGET_FAMILY = "piecewise_c00_a1.00"
PW_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
PR_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
BIT_DELTAS = [-0.1, 0.0, +0.1]


def jittered_segments(orig_segs, seg_idx, pw_f, pr_f, bit_d):
    new_segs = [list(s) for s in orig_segs]
    sf, p_w, p_r, b_p1 = orig_segs[seg_idx]
    new_pw_raw = p_w * pw_f
    new_pr_raw = p_r * pr_f
    new_pw = _clip_simplex2(new_pw_raw, new_pr_raw)
    new_pr = _clip_simplex2(new_pr_raw, new_pw)
    new_bit = _clip01(b_p1 + bit_d)
    new_segs[seg_idx] = [sf, new_pw, new_pr, new_bit]
    return new_segs


def assert_validity(tokens, n_check=100):
    """Walk read-determinism on first n_check sequences."""
    x = tokens.numpy() if isinstance(tokens, torch.Tensor) else tokens
    inst = x[:, 0::2]
    data = x[:, 1::2] - ZERO
    if not (inst[:, 0] == W).all():
        return False
    if not (inst[:, -1] == R).all():
        return False
    for row in range(min(x.shape[0], n_check)):
        last_w = data[row, 0]
        for k in range(1, inst.shape[1]):
            if inst[row, k] == W:
                last_w = data[row, k]
            elif inst[row, k] == R:
                if data[row, k] != last_w:
                    return False
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step3] device={device} N={N_SAMPLES} batch={BATCH_SIZE} max_configs={MAX_CONFIGS}")

    with open(SAMPLER_JSON) as f:
        sampler_spec = json.load(f)
    target_fam = next(f for f in sampler_spec["families"] if f["name"] == TARGET_FAMILY)
    rep_segs = target_fam["rep_config"]["segments"]
    T_full = target_fam["rep_config"]["T"]
    K = len(rep_segs)
    print(f"[step3] target={TARGET_FAMILY}, K={K} segments, T={T_full}")

    # Build full grid then random-subsample.
    grid = [(seg_idx, pw_f, pr_f, bit_d)
            for seg_idx in range(K)
            for pw_f in PW_FACTORS
            for pr_f in PR_FACTORS
            for bit_d in BIT_DELTAS]
    print(f"[step3] full grid = {len(grid)} configs; capping at {MAX_CONFIGS}")
    if len(grid) > MAX_CONFIGS:
        rng = random.Random(0)
        rng.shuffle(grid)
        grid = grid[:MAX_CONFIGS]

    # Load all 3 models once.
    print("\n[step3] loading models ...")
    loaded = {}
    for model_name, (ckpt, cfg_path) in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"  SKIP {model_name}: missing {ckpt}")
            continue
        loaded[model_name] = load_frozen_model(ckpt, cfg_path, device)

    # Loop: build, sample, eval-all-3, free.
    rows = []
    invalid_count = 0
    t0 = time.time()
    for idx, (seg_idx, pw_f, pr_f, bit_d) in enumerate(grid):
        new_segs = jittered_segments(rep_segs, seg_idx, pw_f, pr_f, bit_d)
        try:
            dist = Piecewise(T=T_full, segments=[tuple(s) for s in new_segs])
        except AssertionError as e:
            print(f"  [{idx:>3d}/{len(grid)}] seg={seg_idx} INVALID(build): {e}")
            invalid_count += 1
            continue
        rng_np = np.random.default_rng(6000 + idx)
        toks = dist.sample(N_SAMPLES, rng_np)
        ok = assert_validity(toks, n_check=100)
        if not ok:
            print(f"  [{idx:>3d}/{len(grid)}] seg={seg_idx} INVALID(validity)")
            invalid_count += 1
            del toks
            continue

        # Eval on all 3 loaded models.
        toks_gpu = toks.to(device)
        per_model = {}
        for m, model in loaded.items():
            res = evaluate_dataset(model, toks_gpu, batch_size=BATCH_SIZE, device=device)
            per_model[f"{m}_glitch"] = float(res["error_rate"])

        rows.append({
            "idx": idx, "seg_idx": seg_idx,
            "pw_f": pw_f, "pr_f": pr_f, "bit_d": bit_d,
            "segments": new_segs,
            **per_model,
        })

        # progress print every config (cheap; already running per-config)
        elapsed = time.time() - t0
        eta = (elapsed / (idx + 1)) * (len(grid) - idx - 1)
        if True:  # always print to monitor
            line = f"  [{idx+1:>3d}/{len(grid)}] seg={seg_idx} pwf={pw_f} prf={pr_f} bitd={bit_d:+.1f}"
            for m in loaded:
                line += f" {m}={per_model[f'{m}_glitch']:.3%}"
            line += f" t={elapsed:.0f}s eta={eta:.0f}s"
            print(line)

        del toks_gpu, toks
        if device == "cuda":
            torch.cuda.empty_cache()

    threshold = 0.005
    if not rows:
        print("[step3] no valid rows!")
        return

    fractions = {}
    for m in loaded:
        key = f"{m}_glitch"
        vals = [r[key] for r in rows]
        fractions[m] = float(np.mean([v >= threshold for v in vals]))

    out = {
        "n_samples": N_SAMPLES,
        "max_configs": MAX_CONFIGS,
        "n_grid": len(grid),
        "n_valid": len(rows),
        "n_invalid": invalid_count,
        "threshold": threshold,
        "fractions_above_threshold": fractions,
        "rows": rows,
    }
    with open(os.path.join(OUT_DIR, "neighborhood_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append(f"Step 3 neighborhood escalation around {TARGET_FAMILY}")
    lines.append(f"  n_samples={N_SAMPLES}, n_valid={len(rows)}, n_invalid={invalid_count}, "
                 f"threshold={threshold:.0%}")
    lines.append("")
    lines.append("Fraction of valid jittered configs with glitch >= 0.5%:")
    for m, frac in fractions.items():
        n_above = int(frac * len(rows))
        lines.append(f"  {m:<14}: {frac:>7.2%} ({n_above}/{len(rows)})")
    lines.append("")
    if "R4" in fractions and "R2-redone-v2" in fractions:
        if fractions["R2-redone-v2"] > 0:
            ratio = fractions["R4"] / fractions["R2-redone-v2"]
            lines.append(f"  R4 / R2-v2 ratio = {ratio:.1f}x")
        else:
            lines.append(f"  R4 / R2-v2 ratio = inf (R2-v2 fraction = 0)")
        is_region = fractions["R4"] >= 0.5 and fractions["R2-redone-v2"] <= 0.1
        lines.append(f"  Region (R4 high, R2-v2 low) verdict: {is_region}")
    lines.append("")
    lines.append("Per-segment-jittered breakdown:")
    for seg_i in range(K):
        per_seg = [r for r in rows if r["seg_idx"] == seg_i]
        if not per_seg:
            continue
        means = {m: float(np.mean([r[f"{m}_glitch"] for r in per_seg])) for m in loaded}
        line = f"  seg {seg_i}: n={len(per_seg)}"
        for m, v in means.items():
            line += f"  {m}={v:.4%}"
        lines.append(line)
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "neighborhood_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[step3] wrote {OUT_DIR}/neighborhood_results.json + neighborhood_summary.txt")


if __name__ == "__main__":
    main()
