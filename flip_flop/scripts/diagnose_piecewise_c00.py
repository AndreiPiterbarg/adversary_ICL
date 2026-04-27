"""Step 1: diagnose what makes piecewise_c00 differentially hard for R4.

1a. Print side-by-side segment structure of piecewise c00 / c01 / c02 from
    sampler.json; compute implied p_i = 1 - p_w - p_r per segment.
1b. Compute per-position glitch rate of R4(seed=0) on 10000 fresh samples
    from each piecewise family (seed=8888). Per-position rate is masked to
    positions where the PRECEDING token is 'r' (paper's read mask).
    Sanity check: weighted aggregate ~= scalar evaluate_dataset() error_rate.
1c. Plot per-position curves with c00 segment boundaries overlaid.

Outputs under results/flip_flop/liu_r4/diagnostic_step1/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from types import SimpleNamespace

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.data import R
from flip_flop.eval import evaluate_dataset
from flip_flop.model import build_model

OUT_DIR = "results/flip_flop/liu_r4/diagnostic_step1"
SAMPLER_JSON = "results_tier1_v2/results/flip_flop/retrain/round_2_redone/sampler.json"
R4_CKPT = "results/flip_flop/liu_r4/model_final.pt"
R4_CFG = "results/flip_flop/liu_r4/config.yaml"
EVAL_SEED = 8888
N = 10000
BATCH_SIZE = 64

PIECEWISE_NAMES = ["piecewise_c00_a1.00", "piecewise_c01_a1.00", "piecewise_c02_a1.00"]


def load_model(ckpt_path: str, cfg_path: str, device: str):
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
    return model


@torch.no_grad()
def per_position_glitch(model, tokens: torch.LongTensor, batch_size: int, device: str):
    """Returns (T-1,) np arrays: errors_per_pos, reads_per_pos.

    Index t corresponds to logit position t (predicting token t+1) where
    tokens[:, t] == R.
    """
    T_minus_1 = tokens.size(1) - 1
    errors_per_pos = np.zeros(T_minus_1, dtype=np.int64)
    reads_per_pos = np.zeros(T_minus_1, dtype=np.int64)

    for i in range(0, tokens.size(0), batch_size):
        batch = tokens[i : i + batch_size].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_targets = batch[:, 1:]
        mask = batch[:, :-1] == R
        preds = shift_logits.argmax(dim=-1)
        err = (preds != shift_targets) & mask
        errors_per_pos += err.sum(dim=0).cpu().numpy().astype(np.int64)
        reads_per_pos += mask.sum(dim=0).cpu().numpy().astype(np.int64)

    return errors_per_pos, reads_per_pos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[step1] device={device}")

    with open(SAMPLER_JSON) as f:
        sampler_spec = json.load(f)
    families = {fam["name"]: fam for fam in sampler_spec["families"]}

    # ---- 1a. Side-by-side segment structure ----
    config_summary = {}
    print("\n" + "=" * 100)
    print("1a. Piecewise family segment structures")
    print("=" * 100)
    for fname in PIECEWISE_NAMES:
        fam = families[fname]
        rep = fam["rep_config"]
        segs = rep["segments"]
        rows = []
        for s in segs:
            sf, p_w, p_r, b_p1 = s
            p_i = 1.0 - p_w - p_r
            rows.append({"start_frac": float(sf), "p_w": float(p_w),
                         "p_r": float(p_r), "p_i": float(p_i),
                         "bit_p1": float(b_p1)})
        config_summary[fname] = {
            "n_segments": len(segs),
            "segments": rows,
        }
        print(f"\n{fname}:")
        print(f"  {'start':>8}  {'p_w':>8}  {'p_r':>8}  {'p_i':>8}  {'bit_p1':>8}")
        for r in rows:
            print(f"  {r['start_frac']:>8.3f}  {r['p_w']:>8.4f}  "
                  f"{r['p_r']:>8.4f}  {r['p_i']:>8.4f}  {r['bit_p1']:>8.4f}")

    # ---- 1b. Per-position glitch on R4 seed=0 ----
    print("\n" + "=" * 100)
    print("1b. Per-position glitch rate (R4 seed=0, 10000 fresh seqs, seed=8888)")
    print("=" * 100)
    rng = np.random.default_rng(EVAL_SEED)
    fam_tokens = {}
    for fname in PIECEWISE_NAMES:
        dist = FFLDistribution.from_dict(families[fname]["dist"])
        toks = dist.sample(N, rng)
        fam_tokens[fname] = toks
        print(f"  sampled {N} seqs from {fname}")

    model = load_model(R4_CKPT, R4_CFG, device)
    print("[step1] loaded R4 seed=0")

    sanity_lines = []
    for fname in PIECEWISE_NAMES:
        toks = fam_tokens[fname]
        # Per-position
        errs, reads = per_position_glitch(model, toks, BATCH_SIZE, device)
        rate = np.where(reads > 0, errs / np.maximum(reads, 1), np.nan)
        np.save(os.path.join(OUT_DIR, f"per_position_glitch_{fname}.npy"), rate)
        # also save (errs, reads) so the weighted-mean check is reproducible
        np.save(os.path.join(OUT_DIR, f"per_position_errs_{fname}.npy"), errs)
        np.save(os.path.join(OUT_DIR, f"per_position_reads_{fname}.npy"), reads)
        # Scalar via evaluate_dataset (paper's mask)
        res = evaluate_dataset(model, toks, batch_size=BATCH_SIZE, device=device)
        weighted = errs.sum() / max(reads.sum(), 1)
        line = (f"  {fname}: scalar={res['error_rate']:.6f} "
                f"weighted_per_pos={weighted:.6f} "
                f"abs_diff={abs(res['error_rate']-weighted):.2e} "
                f"(num_errors={errs.sum()}, num_reads={reads.sum()})")
        print(line)
        sanity_lines.append(line)
        # Sanity assertion (within fp tolerance)
        assert abs(res["error_rate"] - weighted) < 1e-9, f"sanity failed for {fname}"

    # ---- Save 1a JSON ----
    with open(os.path.join(OUT_DIR, "piecewise_configs.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    # ---- 1c. Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"piecewise_c00_a1.00": "#d62728",
              "piecewise_c01_a1.00": "#1f77b4",
              "piecewise_c02_a1.00": "#2ca02c"}
    T_minus_1 = None
    for fname in PIECEWISE_NAMES:
        rate = np.load(os.path.join(OUT_DIR, f"per_position_glitch_{fname}.npy"))
        T_minus_1 = len(rate)
        # Only positions with reads contribute. Replace NaN with 0 for plotting.
        rate_p = np.nan_to_num(rate, nan=0.0)
        # Smooth with a 9-pt moving avg for legibility (raw can be noisy at low read counts).
        win = 9
        kernel = np.ones(win) / win
        rate_s = np.convolve(rate_p, kernel, mode="same")
        ax.plot(rate_s, label=fname.replace("_a1.00", ""), color=colors[fname], linewidth=1.4)
    # c00 segment boundaries (multiplied by T=512; logit pos is 0..510 over T=512 tokens).
    # rate index is over t = 0..T-2 mapping to read instructions at token positions t.
    c00_segs = config_summary["piecewise_c00_a1.00"]["segments"]
    T_full = 512
    for s in c00_segs[1:]:  # skip start_frac=0 (left edge)
        x = s["start_frac"] * T_full
        ax.axvline(x, linestyle="--", color="grey", alpha=0.5, linewidth=1)
    ax.set_xlabel("token position t (read mask: tokens[:,t] == r)")
    ax.set_ylabel("R4 glitch rate (9-pt smoothed)")
    ax.set_title("Per-position glitch rate of R4 seed=0 on piecewise families")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_full - 1)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "per_position_glitch.png")
    fig.savefig(plot_path, dpi=150)
    print(f"[step1] wrote {plot_path}")

    # ---- 1d. Where are R4's errors on c00 concentrated? ----
    rate_c00 = np.load(os.path.join(OUT_DIR, f"per_position_glitch_{PIECEWISE_NAMES[0]}.npy"))
    rate_c00_p = np.nan_to_num(rate_c00, nan=0.0)
    # Per-segment glitch (using read counts as weights)
    errs00 = np.load(os.path.join(OUT_DIR, f"per_position_errs_{PIECEWISE_NAMES[0]}.npy"))
    reads00 = np.load(os.path.join(OUT_DIR, f"per_position_reads_{PIECEWISE_NAMES[0]}.npy"))
    seg_summary = []
    n_inst = T_full // 2
    for i, seg in enumerate(c00_segs):
        s_inst = int(round(seg["start_frac"] * n_inst))
        e_inst = int(round(c00_segs[i + 1]["start_frac"] * n_inst)) if i + 1 < len(c00_segs) else n_inst
        # token index of the instruction occupies position 2*k; reads at 2*k.
        # Per-position arrays index t in [0, T-2] = [0, 510]; reads happen at even t.
        s_tok, e_tok = 2 * s_inst, 2 * e_inst  # half-open
        e_tok = min(e_tok, T_full - 1)
        seg_errs = int(errs00[s_tok:e_tok].sum())
        seg_reads = int(reads00[s_tok:e_tok].sum())
        seg_rate = seg_errs / max(seg_reads, 1)
        seg_summary.append({
            "seg_idx": i,
            "start_frac": seg["start_frac"],
            "p_w": seg["p_w"], "p_r": seg["p_r"], "p_i": seg["p_i"],
            "bit_p1": seg["bit_p1"],
            "tok_range": [s_tok, e_tok],
            "errs": seg_errs, "reads": seg_reads,
            "seg_glitch": seg_rate,
        })

    # ---- summary ----
    with open(os.path.join(OUT_DIR, "per_position_glitch_summary.txt"), "w") as f:
        f.write("Step 1b sanity check: weighted-per-position vs scalar evaluate_dataset\n")
        f.write("(should be exactly equal modulo float ordering)\n\n")
        for line in sanity_lines:
            f.write(line + "\n")
        f.write("\nStep 1d: per-segment glitch on piecewise_c00 (R4 seed=0, n=10000)\n")
        f.write(f"  {'seg':>3}  {'tok_range':>14}  {'p_w':>7}  {'p_r':>7}  {'p_i':>7}  "
                f"{'bit_p1':>7}  {'reads':>7}  {'errs':>6}  {'glitch':>9}\n")
        for s in seg_summary:
            f.write(f"  {s['seg_idx']:>3}  [{s['tok_range'][0]:>3},{s['tok_range'][1]:>3})  "
                    f"{s['p_w']:>7.4f}  {s['p_r']:>7.4f}  {s['p_i']:>7.4f}  "
                    f"{s['bit_p1']:>7.4f}  {s['reads']:>7d}  {s['errs']:>6d}  "
                    f"{s['seg_glitch']:>9.4%}\n")

    print(f"[step1] wrote per_position_glitch_summary.txt")
    print(f"[step1] segment-by-segment glitch on c00:")
    for s in seg_summary:
        print(f"  seg {s['seg_idx']} tok[{s['tok_range'][0]},{s['tok_range'][1]}) "
              f"p_w={s['p_w']:.3f} p_r={s['p_r']:.3f} p_i={s['p_i']:.3f} bit_p1={s['bit_p1']:.3f} "
              f"reads={s['reads']} errs={s['errs']} -> {s['seg_glitch']:.4%}")


if __name__ == "__main__":
    main()
