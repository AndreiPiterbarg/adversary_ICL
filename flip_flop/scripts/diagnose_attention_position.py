"""Per-(layer, head) attention-position diagnostic for a trained FFLM.

At each read position t, compute attention mass on three regions:
  mass_correct  = attn[t, t*+1] where t* is the most-recent W before t
                  (Liu et al. Fig 16b: ideal attention target)
  mass_early    = sum(attn[t, 0:8])  (paper Prop 4 / Fig 16d: drift target)
  mass_recent_r = attn[t, most-recent R before t]
  mass_other    = 1 - the above three

Aggregated per (layer, head) and (layer, head, segment), split CORRECT vs
WRONG by argmax(logits[t]) vs tokens[t+1] (same as flip_flop/eval.py).

Headline: on WRONG predictions in segment 3 of piecewise_c00, which
(layer, head) puts the most mass on the early window?
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.data import R, W, sample_ffl
from flip_flop.model import build_model

PIECEWISE_SAMPLER_JSON = "results_tier1_v2/results/flip_flop/retrain/round_2_redone/sampler.json"
PIECEWISE_C00_NAME = "piecewise_c00_a1.00"
EARLY_WINDOW = 8
SPLIT_NAMES = ("correct", "wrong")
REGION_NAMES = ("mass_correct", "mass_early", "mass_recent_r", "mass_other")


def load_model(ckpt_path, cfg_path, device, dtype):
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for k, v in raw.items():
        flat.update(v) if isinstance(v, dict) else flat.__setitem__(k, v)
    cfg = SimpleNamespace(**flat)
    model = build_model(cfg).to(device).to(dtype)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg


def build_family(family, T=512):
    """Return (sampler_fn, seg_token_starts) where the latter has length n_seg+1."""
    if family == "piecewise_c00":
        if not os.path.exists(PIECEWISE_SAMPLER_JSON):
            raise FileNotFoundError(f"missing: {PIECEWISE_SAMPLER_JSON}")
        with open(PIECEWISE_SAMPLER_JSON) as f:
            fams = {x["name"]: x for x in json.load(f)["families"]}
        if PIECEWISE_C00_NAME not in fams:
            raise KeyError(f"{PIECEWISE_C00_NAME} not in {PIECEWISE_SAMPLER_JSON}")
        dist = FFLDistribution.from_dict(fams[PIECEWISE_C00_NAME]["dist"])
        n_inst = dist.T // 2
        starts = [int(round(s[0] * n_inst)) for s in dist.segments]
        return (lambda n, rng: dist.sample(n, rng)), [2 * s for s in starts] + [dist.T]
    if family == "ffl_098":
        return (lambda n, rng: sample_ffl(T, 0.98, n, rng)), [0, T]
    if family == "ffl_080":
        return (lambda n, rng: sample_ffl(T, 0.80, n, rng)), [0, T]
    raise ValueError(f"unknown family {family!r}")


def compute_targets(tokens_np):
    """Per-(b, t): (most-recent W before t) + 1 and most-recent R before t,
    or -1 when none. Output shape (B, T) for both."""
    B, T = tokens_np.shape
    is_w, is_r = tokens_np == W, tokens_np == R
    tstar_plus1 = np.full((B, T), -1, dtype=np.int64)
    tr = np.full((B, T), -1, dtype=np.int64)
    last_w = np.full(B, -1, dtype=np.int64)
    last_r = np.full(B, -1, dtype=np.int64)
    for t in range(T):
        vw = last_w >= 0
        tstar_plus1[vw, t] = last_w[vw] + 1
        vr = last_r >= 0
        tr[vr, t] = last_r[vr]
        last_w = np.where(is_w[:, t], t, last_w)
        last_r = np.where(is_r[:, t], t, last_r)
    return tstar_plus1, tr


@torch.no_grad()
def aggregate_batches(model, sampler, *, n, batch_size, n_layer, n_head,
                      seg_token_starts, device, seed):
    """Forward each batch with output_attentions=True; accumulate region-mass
    sums per (split, seg, layer, head). Returns (sums, counts) with shapes
    (2, n_seg, n_layer, n_head, 4) and (2, n_seg, n_layer, n_head)."""
    n_seg = len(seg_token_starts) - 1
    sums = np.zeros((2, n_seg, n_layer, n_head, 4), dtype=np.float64)
    counts = np.zeros((2, n_seg, n_layer, n_head), dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_done = 0
    while n_done < n:
        b = min(batch_size, n - n_done)
        tokens = sampler(b, rng)
        T = tokens.size(1)
        tokens_np = tokens.numpy()
        tstar_plus1, tr_arr = compute_targets(tokens_np)
        # HF GPT2LMHeadModel forward: .logits (B,T,V), .attentions tuple of
        # n_layer (B,H,T,T) post-softmax row-stochastic over the last dim.
        out = model.model(input_ids=tokens.to(device), output_attentions=True)
        preds_np = out.logits[:, :-1, :].float().argmax(dim=-1).cpu().numpy()
        correct_full = np.zeros((b, T), dtype=bool)
        correct_full[:, :-1] = (preds_np == tokens_np[:, 1:])
        is_read = tokens_np == R
        is_read[:, T - 1] = False
        idx_b, idx_t = np.where(is_read)
        if idx_b.size:
            tp1 = tstar_plus1[idx_b, idx_t]
            trv = tr_arr[idx_b, idx_t]
            split_idx = np.where(correct_full[idx_b, idx_t], 0, 1).astype(np.int64)
            seg_idx = np.clip(
                np.searchsorted(seg_token_starts, idx_t, side="right") - 1,
                0, n_seg - 1)
            combo = split_idx * n_seg + seg_idx
            ar = np.arange(idx_b.size)
            safe_tp1 = np.clip(tp1, 0, T - 1)
            safe_trv = np.clip(trv, 0, T - 1)
            flat_sums = sums.reshape(2 * n_seg, n_layer, n_head, 4)
            flat_counts = counts.reshape(2 * n_seg, n_layer, n_head)
            for layer_idx in range(n_layer):
                A_np = out.attentions[layer_idx].float().cpu().numpy()
                for h in range(n_head):
                    rows = A_np[idx_b, h, idx_t, :]
                    mc = np.where(tp1 >= 0, rows[ar, safe_tp1], 0.0)
                    me = rows[:, :EARLY_WINDOW].sum(axis=1)
                    mr = np.where(trv >= 0, rows[ar, safe_trv], 0.0)
                    masses = np.stack([mc, me, mr, 1.0 - mc - me - mr], axis=1)
                    for c in range(2 * n_seg):
                        sel = combo == c
                        if sel.any():
                            flat_sums[c, layer_idx, h] += masses[sel].sum(axis=0)
                            flat_counts[c, layer_idx, h] += sel.sum()
                del A_np
        del out
        n_done += b
        print(f"  [aggregate] {n_done}/{n}")
    return sums, counts


def means_from_sums(sums, counts):
    cb = counts[..., None].astype(np.float64)
    return np.where(cb > 0, sums / np.maximum(cb, 1.0), np.nan)


def head_means(sums, counts, l, h):
    c = int(counts[l, h])
    if c == 0:
        return {r: None for r in REGION_NAMES} | {"n_reads": 0}
    d = {r: float(sums[l, h, k] / c) for k, r in enumerate(REGION_NAMES)}
    d["n_reads"] = c
    return d


def make_summary_json(sums, counts, n_layer, n_head):
    flat_sums = sums.sum(axis=(0, 1))
    flat_counts = counts.sum(axis=(0, 1))
    return {f"layer_{l}_head_{h}": head_means(flat_sums, flat_counts, l, h)
            for l in range(n_layer) for h in range(n_head)}


def make_wrong_vs_right_json(sums, counts, n_layer, n_head, n_seg):
    means = means_from_sums(sums, counts)
    out = {}
    for s, sname in enumerate(SPLIT_NAMES):
        for sg in range(n_seg):
            heads = {}
            for l in range(n_layer):
                for h in range(n_head):
                    if counts[s, sg, l, h] == 0:
                        heads[f"layer_{l}_head_{h}"] = {r: None for r in REGION_NAMES}
                    else:
                        heads[f"layer_{l}_head_{h}"] = {
                            r: float(means[s, sg, l, h, k])
                            for k, r in enumerate(REGION_NAMES)}
            out[f"{sname}_seg{sg}"] = {
                "n_reads": int(counts[s, sg, 0, 0]),
                "heads": heads,
            }
    return out


def make_summary_md(sums, counts, n_layer, n_head, n_seg, *,
                    family, ckpt_path, n_total):
    """Rank heads by mass_early on WRONG in the LAST segment (segment 3 for
    piecewise_c00; segment 0 for stationary families)."""
    means = means_from_sums(sums, counts)
    target = n_seg - 1
    wi = SPLIT_NAMES.index("wrong")
    ek = REGION_NAMES.index("mass_early")
    ck = REGION_NAMES.index("mass_correct")
    flat = []
    for l in range(n_layer):
        for h in range(n_head):
            nw = int(counts[wi, target, l, h])
            if nw:
                flat.append((l, h,
                             float(means[wi, target, l, h, ek]),
                             float(means[wi, target, l, h, ck]),
                             nw))
    flat.sort(key=lambda r: r[2], reverse=True)
    n_wrong = int(counts[wi, target, 0, 0])
    n_corr = int(counts[SPLIT_NAMES.index("correct"), target, 0, 0])
    lines = [
        "# Attention-position diagnostic", "",
        f"- ckpt: `{ckpt_path}`",
        f"- family: `{family}`",
        f"- sequences: {n_total}",
        f"- segments: {n_seg} (target = segment {target})",
        f"- WRONG reads in segment {target}: {n_wrong}",
        f"- CORRECT reads in segment {target}: {n_corr}",
        "",
        f"## Top-3 heads by mean mass_early on WRONG predictions in segment {target}",
        "",
        f"Paper drift prediction: Prop 4 / Fig 16d (early writes dominate at large t).",
        "",
        "| rank | layer | head | mass_early | mass_correct | n_wrong |",
        "| ---: | ----: | ---: | ---------: | -----------: | ------: |",
    ]
    for rank, (l, h, me, mc, nw) in enumerate(flat[:3], start=1):
        lines.append(f"| {rank} | {l} | {h} | {me:.4f} | {mc:.4f} | {nw} |")
    lines.append("")
    if flat:
        l, h, me, mc, nw = flat[0]
        lines.append(
            f"**Headline:** on WRONG predictions in segment {target}, "
            f"layer {l} head {h} puts the most mass on the early window: "
            f"mass_early={me:.4f} (vs mass_correct={mc:.4f}, n_wrong={nw}).")
    else:
        lines.append("**Headline:** no WRONG predictions in target segment.")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True,
                   help="path to model_final.pt; config.yaml is read from same dir")
    p.add_argument("--family", default="piecewise_c00",
                   choices=["piecewise_c00", "ffl_098", "ffl_080"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default=None,
                   help="default: <ckpt_dir>/attention_position/<family>")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260427)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    out_dir = args.out or os.path.join(ckpt_dir, "attention_position", args.family)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[diag] ckpt={args.ckpt} family={args.family} n={args.n} "
          f"batch_size={args.batch_size} device={device} out={out_dir}")

    model, cfg = load_model(args.ckpt, cfg_path, device, torch.float32)
    n_layer, n_head, T = int(cfg.n_layer), int(cfg.n_head), int(cfg.n_positions)
    sampler, seg_token_starts = build_family(args.family, T=T)
    n_seg = len(seg_token_starts) - 1
    print(f"[diag] L={n_layer} H={n_head} T={T} seg_starts={seg_token_starts}")

    sums, counts = aggregate_batches(
        model, sampler, n=args.n, batch_size=args.batch_size,
        n_layer=n_layer, n_head=n_head,
        seg_token_starts=seg_token_starts, device=device, seed=args.seed)

    summary = make_summary_json(sums, counts, n_layer, n_head)
    wvr = make_wrong_vs_right_json(sums, counts, n_layer, n_head, n_seg)
    md = make_summary_md(sums, counts, n_layer, n_head, n_seg,
                         family=args.family, ckpt_path=args.ckpt, n_total=args.n)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "wrong_vs_right.json"), "w") as f:
        json.dump(wvr, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md)
    print(f"[diag] wrote summary.json wrong_vs_right.json summary.md to {out_dir}")
    print()
    print(md)


if __name__ == "__main__":
    main()
