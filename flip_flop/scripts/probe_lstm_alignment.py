"""Stages 1-3 of the LSTM-alignment study.

Stage 1: linear probe at each Transformer layer for the LSTM's c_t,
         with Hewitt-Liang selectivity control.
Stage 2: SVD rank analysis of the probe weights at the best layer.
Stage 3: causal subspace ablation -> change in glitch rate, vs random null.

Citations live in the docstrings of flip_flop/analysis/{probe,ablation}.py.

Usage:
  python -m flip_flop.scripts.probe_lstm_alignment
  python -m flip_flop.scripts.probe_lstm_alignment --test_run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from types import SimpleNamespace

import yaml

from flip_flop.analysis.ablation import random_orthonormal, subspace_ablation
from flip_flop.model import build_model


def load_frozen_model(ckpt_path: str, cfg_path: str, device: str):
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    flat = {}
    for v in raw.values():
        if isinstance(v, dict):
            flat.update(v)
    cfg = SimpleNamespace(**(flat if flat else raw))
    model = build_model(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
from flip_flop.analysis.lstm_state import extract_lstm_states
from flip_flop.analysis.probe import (
    fit_and_score,
    rank_curve,
    shuffle_within_sequence,
    smallest_rank_meeting,
)
from flip_flop.analysis.transformer_state import extract_transformer_hidden_states
from flip_flop.data import make_eval_dataset
from flip_flop.eval import evaluate_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "flip_flop"

MODELS = {
    "baseline": RESULTS_ROOT / "baseline",
    "tierA_bitmarkov": RESULTS_ROOT / "retrain" / "tierA_bitmarkov",
    "tierA_writeflip": RESULTS_ROOT / "retrain" / "tierA_writeflip",
}
LSTM_DIR = RESULTS_ROOT / "lstm"


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def flatten_for_probe(x: torch.Tensor) -> torch.Tensor:
    # (B, L, d) -> (B*L, d), float32
    return x.reshape(-1, x.size(-1)).to(torch.float32)


def split_train_eval(B_total: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx = np.arange(B_total)
    rng.shuffle(idx)
    n_train = int(round(train_frac * B_total))
    return torch.from_numpy(idx[:n_train]), torch.from_numpy(idx[n_train:])


def gather_sequences(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, idx)


def run_probe_stage(
    transformer,
    c_states: torch.Tensor,
    tokens: torch.Tensor,
    device: str,
    seed: int,
    train_frac: float,
    ridge: float,
    probe_device: str,
) -> dict:
    """Stage 1: per-layer probe + selectivity. Returns dict with per-layer R^2s and the W at the best layer."""
    hidden_states = extract_transformer_hidden_states(transformer, tokens, device, batch_size=32, to_cpu=True)
    n_layers_inc_emb = len(hidden_states)
    B, L, _ = hidden_states[0].shape

    train_idx, eval_idx = split_train_eval(B, train_frac, seed)
    Y_train_seq = gather_sequences(c_states.cpu(), train_idx)
    Y_eval_seq = gather_sequences(c_states.cpu(), eval_idx)
    Y_train = flatten_for_probe(Y_train_seq)
    Y_eval = flatten_for_probe(Y_eval_seq)

    gen = torch.Generator().manual_seed(seed)
    Y_train_ctrl = shuffle_within_sequence(Y_train, n_seq=Y_train_seq.size(0), seq_len=L, generator=gen)
    Y_eval_ctrl = shuffle_within_sequence(Y_eval, n_seq=Y_eval_seq.size(0), seq_len=L, generator=gen)

    layer_results = []
    fits = []
    for layer in range(n_layers_inc_emb):
        Xs = hidden_states[layer]
        X_train = flatten_for_probe(gather_sequences(Xs, train_idx)).to(probe_device)
        X_eval = flatten_for_probe(gather_sequences(Xs, eval_idx)).to(probe_device)

        Y_tr = Y_train.to(probe_device)
        Y_ev = Y_eval.to(probe_device)
        Y_tr_c = Y_train_ctrl.to(probe_device)
        Y_ev_c = Y_eval_ctrl.to(probe_device)

        real = fit_and_score(X_train, Y_tr, X_eval, Y_ev, ridge=ridge)
        ctrl = fit_and_score(X_train, Y_tr_c, X_eval, Y_ev_c, ridge=ridge)
        layer_results.append({
            "layer": layer,
            "r2_real": real.r2,
            "r2_control": ctrl.r2,
            "selectivity": real.r2 - ctrl.r2,
        })
        fits.append(real)
        print(f"    layer {layer}: r2={real.r2:.4f}  ctrl={ctrl.r2:.4f}  sel={real.r2 - ctrl.r2:+.4f}")

    best_layer = max(range(n_layers_inc_emb), key=lambda i: layer_results[i]["r2_real"])
    return {
        "layer_results": layer_results,
        "best_layer": best_layer,
        "best_fit": fits[best_layer],
        "X_eval_best": flatten_for_probe(gather_sequences(hidden_states[best_layer], eval_idx)).to(probe_device),
        "Y_eval": Y_eval.to(probe_device),
        "n_train_seq": int(train_idx.numel()),
        "n_eval_seq": int(eval_idx.numel()),
    }


def run_rank_stage(probe_out: dict, fraction: float = 0.9) -> dict:
    """Stage 2: SVD rank analysis at the best layer."""
    fit = probe_out["best_fit"]
    r2_curve, U = rank_curve(fit.W, probe_out["X_eval_best"], probe_out["Y_eval"], fit.mean_x, fit.mean_y)
    k_star = smallest_rank_meeting(r2_curve, fit.r2, fraction=fraction)
    return {
        "best_layer": probe_out["best_layer"],
        "r2_full": fit.r2,
        "r2_per_rank": r2_curve,
        "rank_k_for_fraction": k_star,
        "fraction": fraction,
        "U": U.detach().cpu(),
        "mean_x": fit.mean_x.detach().cpu(),
    }


def run_ablation_stage(
    transformer,
    rank_out: dict,
    eval_sets: dict[str, torch.Tensor],
    device: str,
    eval_batch_size: int,
    seed: int,
) -> dict:
    """Stage 3: ablate the rank-k input subspace at the best layer; compare to random null."""
    layer = rank_out["best_layer"]
    k = rank_out["rank_k_for_fraction"]
    U = rank_out["U"][:, :k].to(device)
    mean_x = rank_out["mean_x"].to(device)
    d = U.size(0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    U_random = random_orthonormal(d, k, device="cpu", dtype=torch.float32, generator=gen).to(device)

    out: dict[str, dict] = {"layer": layer, "rank_k": k, "per_set": {}}
    for set_name, tokens in eval_sets.items():
        baseline_eval = evaluate_dataset(transformer, tokens, batch_size=eval_batch_size, device=device)
        per_mode = {"baseline_no_ablation": baseline_eval}
        for mode in ("mean", "zero", "resample"):
            with subspace_ablation(transformer, layer=layer, U=U, mean_x_for_proj=mean_x, mode=mode):
                per_mode[f"probe_subspace_{mode}"] = evaluate_dataset(
                    transformer, tokens, batch_size=eval_batch_size, device=device
                )
        with subspace_ablation(transformer, layer=layer, U=U_random, mean_x_for_proj=mean_x, mode="mean"):
            per_mode["random_subspace_mean_null"] = evaluate_dataset(
                transformer, tokens, batch_size=eval_batch_size, device=device
            )
        out["per_set"][set_name] = per_mode
        print(f"  [{set_name}] base={baseline_eval['error_rate']:.4f} "
              f"mean={per_mode['probe_subspace_mean']['error_rate']:.4f} "
              f"zero={per_mode['probe_subspace_zero']['error_rate']:.4f} "
              f"resamp={per_mode['probe_subspace_resample']['error_rate']:.4f} "
              f"rand={per_mode['random_subspace_mean_null']['error_rate']:.4f}")
    return out


def write_report(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = ["# LSTM-alignment probe report", ""]
    lines.append(f"- Probe data: FFL(p_i={report['probe_p_i']}), n={report['n_seq']}")
    lines.append(f"- Selectivity: real R^2 - control (within-sequence shuffled c_t) R^2")
    lines.append(f"- Rank-k chosen: smallest k s.t. R^2(rank-k) >= {report['rank_fraction']} * R^2(full)")
    lines.append("")

    for model_name, m in report["models"].items():
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("### Stage 1 - per-layer probe (R^2 of c_t from residual stream)")
        lines.append("")
        lines.append("| layer | R^2 (real) | R^2 (control) | selectivity |")
        lines.append("|---:|---:|---:|---:|")
        for row in m["stage1"]["layer_results"]:
            lines.append(f"| {row['layer']} | {row['r2_real']:.4f} | {row['r2_control']:.4f} | {row['selectivity']:+.4f} |")
        lines.append("")
        lines.append(f"Best layer: **{m['stage1']['best_layer']}** (R^2 = {m['stage2']['r2_full']:.4f})")
        lines.append("")
        lines.append("### Stage 2 - rank analysis at best layer")
        lines.append("")
        r2s = m["stage2"]["r2_per_rank"]
        lines.append(f"- R^2 per rank (k=1..{len(r2s)}): "
                     + ", ".join(f"k{k+1}={r:.3f}" for k, r in enumerate(r2s[: min(16, len(r2s))]))
                     + (" ..." if len(r2s) > 16 else ""))
        lines.append(f"- Smallest k for {int(100*m['stage2']['fraction'])}% of full R^2: **k = {m['stage2']['rank_k_for_fraction']}**")
        lines.append("")
        lines.append("### Stage 3 - causal subspace ablation (glitch rate)")
        lines.append("")
        for set_name, s3 in m["stage3"]["per_set"].items():
            lines.append(f"#### {set_name}")
            lines.append("")
            lines.append("| ablation | error_rate | num_errors / num_predictions |")
            lines.append("|:--|---:|:--|")
            for mode_name, e in s3.items():
                lines.append(f"| {mode_name} | {e['error_rate']:.4f} | {e['num_errors']} / {e['num_predictions']} |")
            lines.append("")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def maybe_plot(report: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, m in report["models"].items():
        rows = m["stage1"]["layer_results"]
        xs = [r["layer"] for r in rows]
        ys = [r["r2_real"] for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{model_name} (real)")
        ys_c = [r["r2_control"] for r in rows]
        ax.plot(xs, ys_c, linestyle="--", alpha=0.5, label=f"{model_name} (control)")
    ax.set_xlabel("Transformer layer (0 = embedding output)")
    ax.set_ylabel("R^2 of LSTM c_t (variance-weighted)")
    ax.set_title("Per-layer probe R^2 with selectivity controls")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_r2_per_layer.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_p_i", type=float, default=0.8)
    parser.add_argument("--n_seq", type=int, default=192, help="sequences for probe (train + eval)")
    parser.add_argument("--train_frac", type=float, default=0.75)
    parser.add_argument("--abl_n", type=int, default=512, help="sequences per eval set for ablation")
    parser.add_argument("--abl_p_is", type=float, nargs="+", default=[0.8, 0.98, 0.1])
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--rank_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--probe_device", type=str, default="auto",
                        help="device for closed-form ridge solve; 'auto' = same as model device")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "results" / "analysis" / "lstm_alignment"))
    parser.add_argument("--models", type=str, nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--test_run", action="store_true",
                        help="tiny n for smoke testing (n_seq=32, abl_n=32, FFL(0.8) only)")
    args = parser.parse_args()

    if args.test_run:
        args.n_seq = 32
        args.abl_n = 32
        args.abl_p_is = [0.8]
        args.eval_batch_size = 16

    device = resolve_device(args.device)
    probe_device = device if args.probe_device == "auto" else args.probe_device
    out_dir = Path(args.out_dir)
    print(f"[setup] device={device}  probe_device={probe_device}  out_dir={out_dir}")

    cfg = next(iter(MODELS.values())) / "config.yaml"
    with open(cfg) as f:
        seq_len = yaml.safe_load(f)["seq_len"]
    T_tokens = 2 * seq_len

    print(f"[stage 0] sampling probe data: FFL({args.probe_p_i}), n={args.n_seq}, T_tokens={T_tokens}")
    probe_tokens = make_eval_dataset(args.probe_p_i, args.n_seq, seq_len, args.seed)

    print(f"[stage 0] loading LSTM and extracting c_t per position...")
    lstm = load_frozen_model(str(LSTM_DIR / "model_final.pt"), str(LSTM_DIR / "config.yaml"), device)
    _, c_states = extract_lstm_states(lstm, probe_tokens, device)
    c_states = c_states.cpu()
    print(f"[stage 0] c_states shape: {tuple(c_states.shape)}")
    del lstm
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[stage 0] sampling ablation eval sets...")
    eval_sets: dict[str, torch.Tensor] = {}
    for p_i in args.abl_p_is:
        key = f"FFL_{p_i:g}"
        eval_sets[key] = make_eval_dataset(p_i, args.abl_n, seq_len, args.seed + 100 + int(p_i * 100))

    report = {
        "probe_p_i": args.probe_p_i,
        "n_seq": args.n_seq,
        "train_frac": args.train_frac,
        "ridge": args.ridge,
        "rank_fraction": args.rank_fraction,
        "abl_n": args.abl_n,
        "abl_p_is": args.abl_p_is,
        "seed": args.seed,
        "models": {},
    }

    for name in args.models:
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
        ckpt_dir = MODELS[name]
        print(f"\n[{name}] loading from {ckpt_dir}")
        model = load_frozen_model(str(ckpt_dir / "model_final.pt"), str(ckpt_dir / "config.yaml"), device)

        print(f"[{name}] Stage 1: per-layer probe...")
        probe_out = run_probe_stage(
            model, c_states, probe_tokens, device,
            seed=args.seed, train_frac=args.train_frac, ridge=args.ridge,
            probe_device=probe_device,
        )

        print(f"[{name}] Stage 2: rank analysis at layer {probe_out['best_layer']}...")
        rank_out = run_rank_stage(probe_out, fraction=args.rank_fraction)
        print(f"[{name}]   rank for {int(100*args.rank_fraction)}% R^2: k = {rank_out['rank_k_for_fraction']}")

        print(f"[{name}] Stage 3: causal ablation at layer {rank_out['best_layer']}, rank {rank_out['rank_k_for_fraction']}...")
        abl_out = run_ablation_stage(
            model, rank_out, eval_sets,
            device=device, eval_batch_size=args.eval_batch_size, seed=args.seed,
        )

        report["models"][name] = {
            "stage1": {
                "layer_results": probe_out["layer_results"],
                "best_layer": probe_out["best_layer"],
                "n_train_seq": probe_out["n_train_seq"],
                "n_eval_seq": probe_out["n_eval_seq"],
            },
            "stage2": {
                "best_layer": rank_out["best_layer"],
                "r2_full": rank_out["r2_full"],
                "r2_per_rank": rank_out["r2_per_rank"],
                "rank_k_for_fraction": rank_out["rank_k_for_fraction"],
                "fraction": rank_out["fraction"],
            },
            "stage3": abl_out,
        }

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[done] writing report to {out_dir}")
    write_report(report, out_dir)
    maybe_plot(report, out_dir)


if __name__ == "__main__":
    main()
