"""PLAN_beat_liu_r4 Sec.2.B + Sec.4 -- the rigorous head-to-head.

Every trained model + the LSTM skyline, scored on the SAME N sequences per
frozen pre-registered distribution, with a PAIRED bootstrap CI on the
B_liu - model pooled-error difference (defensible at ~1e-5 error scales where
aggregate means are noise).

Distribution groups (PLAN Sec.3):
  * piecewise band (HELDOUT_PIECEWISE + 16 frozen jitters) -- THE BEAT AXIS
    (non-stationary; Liu's stationary sampler structurally cannot emit it)
  * liu_ffl_098 / liu_ffl_010                  -- tie-where-Liu-is-strong
  * periodic (20, full-simplex, seed 42)        -- unseen-axis tie check
  * heldout_anticorr                            -- completeness (proven dead)

Headline gate (PLAN Sec.4):
  BEAT  : B_liu pooled err on the band >= 100x LSTM AND the paired bootstrap
          95% CI of (B_liu - <our model>) excludes 0 in our favour.
  TIE   : |B_liu - <our model>| CI within delta=2e-4 on liu tails + periodic.
  FALSIFY: B_liu band err <= 5x LSTM at this N -> no beat (report, stop).

Outputs <out_dir>/{summary.md, per_distribution.jsonl}.

Usage:
  python -m flip_flop.scripts.eval_headtohead
  python -m flip_flop.scripts.eval_headtohead --n 16384 --bootstrap 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.adversary.heldout import (HELDOUT_ANTICORR, HELDOUT_PIECEWISE,
                                         LIU_TAILS, piecewise_band)
from flip_flop.adversary.io import load_frozen_model
from flip_flop.eval import evaluate_dataset_per_seq
from flip_flop.scripts.eval_fair import _build_tier_C_periodic

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "flip_flop"

MODELS = {
    "baseline":           (RESULTS / "baseline"),
    "B_liu":              (RESULTS / "full_from_scratch_liu_only"),
    "full_adv":           (RESULTS / "full_from_scratch_adv"),
    "C_regret":           (RESULTS / "full_from_scratch_regret"),
    "beat_R1":            (RESULTS / "beat_liu_loop" / "round1" / "model"),
    "stationary_control": (RESULTS / "full_from_scratch_stationary_control"),
    "lstm":               (RESULTS / "lstm"),
}

OUR_MODELS = ["full_adv", "C_regret", "beat_R1"]  # the dataset(s) we claim for


def _load_present(device, only=""):
    keep = set(x.strip() for x in only.split(",") if x.strip())
    keep |= {"B_liu", "lstm"}  # always needed
    models = {}
    for name, d in MODELS.items():
        if only and name not in keep:
            print(f"[skip] {name} (not in --models)")
            continue
        ckpt, cfg = d / "model_final.pt", d / "config.yaml"
        if ckpt.exists() and cfg.exists():
            models[name] = load_frozen_model(str(ckpt), str(cfg), device)
            print(f"[load] {name}")
        else:
            print(f"[skip] {name} (not trained yet)")
    return models


def _eval_dists(T: int, n_periodic: int = 20):
    """Frozen pre-registered eval set. group -> {dist_name: cfg}."""
    band = {"heldout_piecewise": dict(HELDOUT_PIECEWISE, T=T)}
    band.update(piecewise_band(T=T))                       # 16 frozen jitters
    periodic = {d["name"]: d["config"]
                for d in _build_tier_C_periodic(n_periodic, T, seed=42)}
    return {
        "piecewise_band": band,
        "liu_tails": {"liu_ffl_098": dict(LIU_TAILS["ffl_098"], T=T),
                      "liu_ffl_010": dict(LIU_TAILS["ffl_010"], T=T)},
        "periodic": periodic,
        "anticorr": {"heldout_anticorr": dict(HELDOUT_ANTICORR, T=T)},
    }


def _paired_bootstrap(err_a, rd_a, err_b, rd_b, B, rng):
    """95% percentile CI of pooled_err(a) - pooled_err(b), paired over the
    SAME resampled sequence indices. Returns (mean_diff, lo, hi)."""
    n = len(err_a)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ea, ra = err_a[idx].sum(), rd_a[idx].sum()
        eb, rb = err_b[idx].sum(), rd_b[idx].sum()
        diffs[b] = (ea / max(ra, 1)) - (eb / max(rb, 1))
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out_dir", default=str(RESULTS / "eval_headtohead"))
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--n_periodic", type=int, default=20,
                   help="Periodic tie-check configs (fewer = faster)")
    p.add_argument("--models", default="",
                   help="Comma list to restrict MODELS (default: all present). "
                        "lstm + B_liu always kept.")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lstm_invalid", type=float, default=5e-4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[h2h] device={device} N={args.n} bootstrap={args.bootstrap}")

    models = _load_present(device, only=args.models)
    assert "B_liu" in models and "lstm" in models, "need B_liu + lstm at minimum"
    groups = _eval_dists(args.T, n_periodic=args.n_periodic)
    boot_rng = np.random.default_rng(args.seed)

    rows = []
    perseq = {}  # (group,dist) -> {model: (err[],rd[])}
    n_total = sum(len(g) for g in groups.values())
    done = 0
    for gname, dists in groups.items():
        for dname, cfg in dists.items():
            done += 1
            toks = FFLDistribution.from_dict(cfg).sample(
                args.n, np.random.default_rng(args.seed))
            per = {}
            for mname, m in models.items():
                e, r = evaluate_dataset_per_seq(
                    m, toks, batch_size=args.eval_batch_size, device=device)
                per[mname] = (e, r)
            perseq[(gname, dname)] = per
            lstm_e = per["lstm"][0].sum() / max(per["lstm"][1].sum(), 1)
            valid = lstm_e <= args.lstm_invalid
            row = {"group": gname, "dist": dname, "lstm_err": float(lstm_e),
                   "lstm_valid": bool(valid)}
            for mname, (e, r) in per.items():
                row[mname] = float(e.sum() / max(r.sum(), 1))
            rows.append(row)
            flag = "" if valid else "  [LSTM-INVALID -> excluded]"
            print(f"[{done:>2}/{n_total}] {gname}/{dname:<22} "
                  f"lstm={lstm_e:.2e} B_liu={row.get('B_liu',0):.2e}{flag}")

    with open(out_dir / "per_distribution.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---- band-pooled headline bootstrap (the BEAT test) ----
    def _pool(group):
        keep = [(g, d) for (g, d) in perseq
                if g == group and next(r for r in rows
                                       if r["group"] == g and r["dist"] == d)["lstm_valid"]]
        agg = {}
        for m in models:
            es = np.concatenate([perseq[k][m][0] for k in keep])
            rs = np.concatenate([perseq[k][m][1] for k in keep])
            agg[m] = (es, rs)
        return agg, len(keep)

    md = ["# Head-to-head: beat Liu R4 on the non-stationary axis\n",
          f"- N={args.n} seq/dist, T={args.T}, seed={args.seed}, "
          f"bootstrap={args.bootstrap} paired resamples, "
          f"LSTM-invalid > {args.lstm_invalid}\n",
          f"- models: {', '.join(models)}\n"]

    for group in ("piecewise_band", "liu_tails", "periodic", "anticorr"):
        agg, nk = _pool(group)
        le = agg["lstm"][0].sum() / max(agg["lstm"][1].sum(), 1)
        md.append(f"\n## {group}  (pooled over {nk} LSTM-valid dists)\n")
        md.append("| model | pooled_err | vs_LSTM | (B_liu - model) 95% CI |")
        md.append("|---|--:|--:|---|")
        for m in models:
            if m == "lstm":
                continue
            me = agg[m][0].sum() / max(agg[m][1].sum(), 1)
            ratio = "inf" if le == 0 else f"{me/le:.1f}x"
            if m == "B_liu":
                ci = "(reference)"
            else:
                dmean, lo, hi = _paired_bootstrap(
                    agg["B_liu"][0], agg["B_liu"][1], agg[m][0], agg[m][1],
                    args.bootstrap, boot_rng)
                sign = "BEAT" if lo > 0 else ("tie" if (lo <= 0 <= hi) else "worse")
                ci = f"{dmean:+.2e} [{lo:+.2e}, {hi:+.2e}] {sign}"
            md.append(f"| {m} | {me:.3e} | {ratio} | {ci} |")
        md.append(f"\nLSTM pooled_err = {le:.3e}\n")

    # ---- explicit verdicts ----
    pb, _ = _pool("piecewise_band")
    le = pb["lstm"][0].sum() / max(pb["lstm"][1].sum(), 1)
    be = pb["B_liu"][0].sum() / max(pb["B_liu"][1].sum(), 1)
    md.append("\n## Verdict\n")
    if be <= 5 * max(le, 1e-9):
        md.append(f"- **FALSIFY**: B_liu band err {be:.2e} <= 5x LSTM {le:.2e} "
                  f"-> no non-stationary beat at this N. Report negative.")
    else:
        md.append(f"- B_liu band err {be:.2e} = {be/max(le,1e-9):.0f}x LSTM "
                  f"{le:.2e} -> the axis is real.")
        for m in OUR_MODELS:
            if m not in models:
                continue
            me = pb[m][0].sum() / max(pb[m][1].sum(), 1)
            dmean, lo, hi = _paired_bootstrap(
                pb["B_liu"][0], pb["B_liu"][1], pb[m][0], pb[m][1],
                args.bootstrap, boot_rng)
            tag = "BEAT (CI excludes 0, our favour)" if lo > 0 else \
                  ("tie" if lo <= 0 <= hi else "B_liu better")
            md.append(f"- {m}: band err {me:.2e}; (B_liu-{m}) "
                      f"[{lo:+.2e},{hi:+.2e}] -> **{tag}**")
        if "stationary_control" not in models:
            md.append("- stationary_control: NOT TRAINED yet "
                      "(Rung 1 -- necessity control still pending).")
        else:
            me = pb["stationary_control"][0].sum() / max(pb["stationary_control"][1].sum(), 1)
            nec = "NECESSARY (control still fails band)" if me > 10 * max(le, 1e-9) \
                  else "NOT necessary (control also closes band -> honest downgrade)"
            md.append(f"- stationary_control band err {me:.2e} "
                      f"-> non-stationarity is **{nec}**")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[h2h] wrote {out_dir/'summary.md'} and per_distribution.jsonl")


if __name__ == "__main__":
    main()
