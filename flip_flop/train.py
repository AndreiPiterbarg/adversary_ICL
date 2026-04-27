"""Training loop for FFLM.

Matches Section 4 / Appendix B.2 of Liu et al. 2023 for the baseline sweep:
  * AdamW, (beta1, beta2) = (0.9, 0.999), weight_decay = 0.1
  * learning rate 3e-4, 50-step linear warmup, linear decay to 0 at step 10001
  * batch size 16, 10000 steps (~81.9M training tokens at T = 512)
  * Train on FFL(0.8); evaluate FFL(0.8) / FFL(0.98) / FFL(0.1).

Entry point: flip_flop/scripts/run_baseline.py.
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from .data import make_eval_dataset, sample_ffl
from .eval import clean_loss, evaluate_dataset
from .model import build_model


@dataclass
class TrainConfig:
    # model
    family: str = "gpt2"
    vocab_size: int = 5
    n_positions: int = 512
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    # lstm-only
    hidden_size: int = 128
    num_layers: int = 1
    # data
    seq_len: int = 512
    train_p_i: float = 0.8
    eval_in_p_i: float = 0.8
    eval_sparse_p_i: float = 0.98
    eval_dense_p_i: float = 0.1
    eval_in_n: int = 1000
    eval_sparse_n: int = 100_000
    eval_dense_n: int = 3000
    eval_seed: int = 1
    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 50
    train_steps: int = 10_000
    decay_end_step: int = 10_001  # LR reaches 0 here
    batch_size: int = 16
    grad_clip: float = 1.0  # matches HF / x-transformers default; prevents divergence spikes
    # logging / i/o
    seed: int = 0
    eval_every: int = 500
    training_eval_subset: int = 1000  # paper: "first 1% of (ii)" during training
    log_every: int = 50
    save_every: int = 2000
    eval_batch_size: int = 64
    out_dir: str = "results/flip_flop/baseline"
    device: str = "auto"
    # optional: continue training from an existing checkpoint (used by retrain loop)
    init_from_ckpt: str = ""
    # Balanced selection (Fix 2 + Step 4: three-tail score). Active only when
    # sampler has .families. Score per eval step:
    #   score = mean(family_glitch)
    #         + lambda_in * max(0, ffl_in_glitch  - baseline_in)
    #         + lambda_98 * max(0, ffl_98_glitch  - baseline_98)
    #         + lambda_01 * max(0, ffl_01_glitch  - baseline_01)
    # Lower is better. model_final.pt = argmin(score) across eval steps.
    # Per-tail lambdas sized to that test set's empirical noise floor:
    # σ_emp = √(p̂(1−p̂)/N).  Round-0 baseline values:
    #   FFL(0.8): N=26523, p̂≈0.000  → σ≈0  → lambda_in=5 (degenerate; conservative)
    #   FFL(0.98):N=10000, p̂≈0.060  → σ≈0.0024 → lambda_98≈0.7  (1pp ≈ 4σ)
    #   FFL(0.1): N=3000,  p̂≈0.016  → σ≈0.0023 → lambda_01≈0.7  (1pp ≈ 4σ)
    # Hard cap fires if ANY of the three tails exceeds baseline + 0.005.
    selection_enabled: bool = True
    lambda_in: float = 5.0
    lambda_98: float = 0.7
    lambda_01: float = 0.7
    # legacy single-lambda field kept for backwards-compat with older yamls;
    # if set, overrides lambda_in (the historical behavior).
    lambda_penalty: float = -1.0          # -1.0 = unset; use lambda_in
    baseline_in_dist_glitch: float = 0.0  # spec'd from baseline run
    baseline_98_glitch: float = 0.0599
    baseline_01_glitch: float = 0.0157
    in_dist_hard_cap: float = 0.005       # 0.5pp above baseline → terminate
    tail_98_hard_cap: float = 0.005       # 0.5pp on FFL(0.98)
    tail_01_hard_cap: float = 0.005       # 0.5pp on FFL(0.1)
    family_eval_n: int = 2000             # sequences per family per eval step
    plateau_window: int = 5               # consecutive evals to check
    plateau_tol: float = 0.005            # min score-range over window to continue
    # Memorization detector (Step 5): if any of the first
    # `memorize_check_evals` eval steps shows min(per_family_glitch) == 0,
    # suppress plateau halt until step >= memorize_warmup_steps.
    memorize_check_evals: int = 2
    memorize_warmup_steps: int = 800

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        flat = {k: v for k, v in flat.items() if k in known}
        return cls(**flat)


def _make_lr_lambda(warmup, decay_end):
    """Linear warmup then linear decay; step `decay_end` maps to 0."""
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (decay_end - step) / max(1, decay_end - warmup))
    return lr_lambda


def _run_eval(model, eval_sets, cfg, device, step, fh, subset=None):
    out = {"step": step, "subset": subset}
    for name, ds in eval_sets.items():
        view = ds if subset is None else ds[: min(subset, len(ds))]
        res = evaluate_dataset(model, view, batch_size=cfg.eval_batch_size, device=device)
        out[name] = res
        print(
            f"  eval@{step} {name}: err={res['error_rate']:.3e} "
            f"({res['num_errors']}/{res['num_predictions']}) loss={res['loss']:.4f}"
        )
    fh.write(json.dumps(out) + "\n")
    fh.flush()
    return out


def train(cfg, sampler=None):
    """Train the model. If `sampler` is provided (a callable `(batch_size, rng) ->
    LongTensor`), it replaces the default FFL(cfg.train_p_i) sampler — used by
    the retrain loop to inject MixedSampler. If `cfg.init_from_ckpt` is set,
    the model weights are loaded from that path before training (continue-train).
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

    model = build_model(cfg).to(device)
    if cfg.init_from_ckpt:
        sd = torch.load(cfg.init_from_ckpt, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        print(f"[flip_flop] loaded weights from {cfg.init_from_ckpt}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[flip_flop] model={model.name} params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(cfg.warmup_steps, cfg.decay_end_step)
    )

    print("[flip_flop] building eval datasets...")
    eval_sets = {
        "in_distribution": make_eval_dataset(cfg.eval_in_p_i, cfg.eval_in_n, cfg.seq_len, cfg.eval_seed),
        "sparse_tail": make_eval_dataset(cfg.eval_sparse_p_i, cfg.eval_sparse_n, cfg.seq_len, cfg.eval_seed + 1),
        "dense_tail": make_eval_dataset(cfg.eval_dense_p_i, cfg.eval_dense_n, cfg.seq_len, cfg.eval_seed + 2),
    }
    train_rng = np.random.default_rng(cfg.seed + 100)

    log_fh = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a")
    eval_fh = open(os.path.join(cfg.out_dir, "eval_log.jsonl"), "a")

    model.train()
    t0 = time.time()
    last_loss = float("nan")

    if sampler is None:
        def sampler(bs, rng):
            return sample_ffl(cfg.seq_len, cfg.train_p_i, bs, rng)

    # Balanced selection setup (Fix 2). Active iff the provided sampler
    # exposes .families (only MixedSampler does); otherwise we fall back to
    # last-step model_final.pt as before.
    families = getattr(sampler, "families", None)
    selection_active = cfg.selection_enabled and families is not None and len(families) > 0
    selection_log_fh = open(os.path.join(cfg.out_dir, "selection_log.jsonl"), "a") \
        if selection_active else None
    fam_rng = np.random.default_rng(cfg.seed + 200)
    best_score = float("inf")
    best_step = -1
    score_window: list[float] = []
    halt_reason = "completed"

    # Resolve effective lambdas (legacy: lambda_penalty > 0 overrides lambda_in)
    lambda_in_eff = cfg.lambda_penalty if cfg.lambda_penalty > 0 else cfg.lambda_in
    lambda_98_eff = cfg.lambda_98
    lambda_01_eff = cfg.lambda_01

    def _eval_and_score(step_idx: int):
        """Run full eval, eval families, compute three-tail balanced score,
        save best checkpoint. Returns (ffl_in, ffl_98, ffl_01, fam_mean, score)."""
        nonlocal best_score, best_step
        # Standard FFL eval (writes to eval_log.jsonl)
        out = _run_eval(model, eval_sets, cfg, device, step_idx, eval_fh,
                        subset=cfg.training_eval_subset)
        ffl_in_err = out["in_distribution"]["error_rate"]
        ffl_98_err = out["sparse_tail"]["error_rate"]
        ffl_01_err = out["dense_tail"]["error_rate"]
        if not selection_active:
            return ffl_in_err, ffl_98_err, ffl_01_err, None, None
        # Family eval: fresh sample from each family, compute mean glitch
        from .eval import evaluate_dataset
        fam_glitches = []
        for fam in families:
            tokens = fam.sample(cfg.family_eval_n, fam_rng).to(device)
            res = evaluate_dataset(model, tokens, batch_size=cfg.eval_batch_size, device=device)
            fam_glitches.append(res["error_rate"])
        fam_mean = float(np.mean(fam_glitches))
        # Three-tail balanced score; lower is better.
        score = (
            fam_mean
            + lambda_in_eff * max(0.0, ffl_in_err - cfg.baseline_in_dist_glitch)
            + lambda_98_eff * max(0.0, ffl_98_err - cfg.baseline_98_glitch)
            + lambda_01_eff * max(0.0, ffl_01_err - cfg.baseline_01_glitch)
        )
        rec = {
            "step": step_idx,
            "ffl_in_glitch": ffl_in_err,
            "ffl_98_glitch": ffl_98_err,
            "ffl_01_glitch": ffl_01_err,
            "family_mean_glitch": fam_mean,
            "family_glitches": fam_glitches,
            "score": score,
            "lambda_in": lambda_in_eff,
            "lambda_98": lambda_98_eff,
            "lambda_01": lambda_01_eff,
            "baseline_in_dist_glitch": cfg.baseline_in_dist_glitch,
            "baseline_98_glitch": cfg.baseline_98_glitch,
            "baseline_01_glitch": cfg.baseline_01_glitch,
        }
        selection_log_fh.write(json.dumps(rec) + "\n")
        selection_log_fh.flush()
        is_best = score < best_score
        if is_best:
            best_score = score
            best_step = step_idx
            torch.save({"step": step_idx, "model_state_dict": model.state_dict()},
                       os.path.join(cfg.out_dir, "model_final.pt"))
            print(f"  [select] new best @ step {step_idx}: score={score:.4f} "
                  f"(fam={fam_mean:.4f}, in={ffl_in_err:.4f}, "
                  f"98={ffl_98_err:.4f}, 01={ffl_01_err:.4f})")
        return ffl_in_err, ffl_98_err, ffl_01_err, fam_mean, score

    try:
        for step in range(cfg.train_steps):
            tokens = sampler(cfg.batch_size, train_rng).to(device)
            loss = clean_loss(model(tokens), tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.item())
            if step % cfg.log_every == 0:
                rec = {"step": step, "loss": last_loss,
                       "lr": float(scheduler.get_last_lr()[0]),
                       "elapsed": time.time() - t0}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                print(f"[step {step:>6d}] loss={last_loss:.4f} lr={rec['lr']:.2e} t={rec['elapsed']:.1f}s")

            if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                ffl_in_err, ffl_98_err, ffl_01_err, fam_mean, score = _eval_and_score(step + 1)
                model.train()
                # Hard caps: any of three tails exceeding baseline + cap → halt.
                # Only after warmup_steps to avoid tripping on early stochastic noise.
                if selection_active and step + 1 >= cfg.warmup_steps:
                    cap_violations = []
                    if ffl_in_err > cfg.baseline_in_dist_glitch + cfg.in_dist_hard_cap:
                        cap_violations.append(f"FFL(0.8): {ffl_in_err:.4f} > {cfg.baseline_in_dist_glitch + cfg.in_dist_hard_cap:.4f}")
                    if ffl_98_err > cfg.baseline_98_glitch + cfg.tail_98_hard_cap:
                        cap_violations.append(f"FFL(0.98): {ffl_98_err:.4f} > {cfg.baseline_98_glitch + cfg.tail_98_hard_cap:.4f}")
                    if ffl_01_err > cfg.baseline_01_glitch + cfg.tail_01_hard_cap:
                        cap_violations.append(f"FFL(0.1): {ffl_01_err:.4f} > {cfg.baseline_01_glitch + cfg.tail_01_hard_cap:.4f}")
                    if cap_violations:
                        print(f"  [hard cap] {'; '.join(cap_violations)}; halt")
                        halt_reason = "tail_hard_cap"
                        break
                # Plateau: stop if score-range over window is below tolerance.
                # Memorization detector (Step 5): if any of the first
                # `memorize_check_evals` eval steps showed min(per_family)==0,
                # suspect memorization — defer plateau halt until step >=
                # `memorize_warmup_steps`.
                if selection_active and score is not None:
                    score_window.append(score)
                    if len(score_window) > cfg.plateau_window:
                        score_window.pop(0)
                    if len(score_window) == cfg.plateau_window:
                        # Read the last `memorize_check_evals` family-glitch
                        # records from the selection log to check for early-zero.
                        suspect_memorize = False
                        try:
                            with open(os.path.join(cfg.out_dir, "selection_log.jsonl")) as sf:
                                early_recs = [json.loads(ln) for ln in sf.readlines()[:cfg.memorize_check_evals]]
                            suspect_memorize = any(
                                min(r.get("family_glitches", [1.0])) == 0.0
                                for r in early_recs
                            )
                        except Exception:
                            suspect_memorize = False

                        if suspect_memorize and step + 1 < cfg.memorize_warmup_steps:
                            print(f"  [plateau-defer] suspected memorization "
                                  f"(early eval saw family_glitch=0); deferring halt "
                                  f"until step {cfg.memorize_warmup_steps}")
                        elif max(score_window) - min(score_window) < cfg.plateau_tol:
                            print(f"  [plateau] score range over last "
                                  f"{cfg.plateau_window} evals < {cfg.plateau_tol}; halt")
                            halt_reason = "plateau"
                            break

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                torch.save(
                    {"step": step + 1, "model_state_dict": model.state_dict()},
                    os.path.join(cfg.out_dir, "state.pt"),
                )

        # End-of-training eval (also a candidate for best-by-score).
        if selection_active:
            ffl_in_err, ffl_98_err, ffl_01_err, fam_mean, score = \
                _eval_and_score(min(step + 1, cfg.train_steps))
        else:
            ffl_in_err = ffl_98_err = ffl_01_err = fam_mean = score = None
        if not selection_active:
            # Legacy path: save last-step model_final.pt.
            final = _run_eval(model, eval_sets, cfg, device, cfg.train_steps, eval_fh)
            torch.save({"step": cfg.train_steps, "model_state_dict": model.state_dict()},
                       os.path.join(cfg.out_dir, "model_final.pt"))
        else:
            # Selection active: model_final.pt was saved on each best update.
            # Always also save model_last.pt for debugging.
            torch.save({"step": step + 1, "model_state_dict": model.state_dict()},
                       os.path.join(cfg.out_dir, "model_last.pt"))
            print(f"[selection] best step={best_step} score={best_score:.4f} "
                  f"halt_reason={halt_reason}")
            final = {"step": best_step, "best_score": best_score,
                     "halt_reason": halt_reason}
    finally:
        log_fh.close()
        eval_fh.close()
        if selection_log_fh is not None:
            selection_log_fh.close()

    return {"final_eval": final, "last_loss": last_loss, "num_params": n_params,
            "halt_reason": halt_reason if selection_active else "completed",
            "best_step": best_step if selection_active else None}
