# Adversarial Flip-Flop Language Modeling

## Project Goal

Find **structured, interesting failure modes** of Transformer FFLMs by training an adversary to discover input distributions where a trained FFLM breaks down, then retrain on the adversarial distribution. The end goal is a publishable result showing *when and why* Transformer FFLMs exhibit "attention glitches" on specific, learned-worst-case flip-flop distributions.

### Core Idea

We keep the same train → search → retrain loop as our prior adversarial-ICL work, but applied to the flip-flop toy from Liu et al. 2023:

1. **Train** an FFLM Transformer on a base distribution (canonical FFL(0.8)).
2. **Adversary** searches over a parametric family of flip-flop distributions — instruction probabilities `(p_w, p_r, p_i)`, possibly data-bit biases, non-stationary / per-position probabilities, sequence length — to maximize the trained model's read-error (glitch) rate.
3. **Retrain** on the adversarial distribution (or a mixture with FFL(0.8)) and measure how the glitch rate moves.

Liu et al. established the existence of breaking tails (FFL(0.98), FFL(0.1)) by hand and showed (R4) that training on rare sequences closes the gap. Our contribution is to automate that discovery — letting an optimizer find the worst distribution inside a structured search space, then close it.

## Reference Implementation

- **Paper**: "Exposing Attention Glitches with Flip-Flop Language Modeling" (Liu et al. 2023, arXiv:2306.00946).
- **Data release**: https://huggingface.co/datasets/synthseq/flipflop
- FFLM training in `flip_flop/` follows Appendix B.2 of the paper: 6-layer / 512-dim / 8-head GPT-2 baseline (~19M params), AdamW (β=0.9/0.999, lr=3e-4, wd=0.1), 50-step warmup + linear decay to 0 at step 10001, batch 16 × 10000 steps, "clean-mode" loss on read positions only.

## Repository Structure

```
adversary_ICL/
├── CLAUDE.md
├── flip_flop/              # PRIMARY: FFLM experiments
│   ├── data.py             # FFL(T, p) sampler
│   ├── model.py            # GPT-2 Transformer + 1-layer LSTM baseline
│   ├── train.py            # Training loop (paper Appendix B.2)
│   ├── eval.py             # Clean-mode loss + glitch-rate evaluation
│   ├── adversary/          # (planned) optimizer over FFL distribution params
│   ├── configs/            # baseline.yaml, lstm.yaml
│   └── scripts/            # run_baseline.py (entry point)
├── src/                    # Prior work: linear-regression ICL + CMA-ES adversary
│   ├── icl/  adversary/  tasks/  eval/
├── configs/  experiments/  # Legacy configs & launch scripts for src/
├── results/                # Checkpoints & logs (gitignored)
└── notebooks/              # Analysis only
```

## Key Concepts

- **FFLM**: autoregressive LM over the vocabulary `{w, r, i, 0, 1}`. `w`/`r`/`i` are write/read/ignore instructions; `0`/`1` are data bits. Valid strings alternate instruction/data, start with `w`, end with `r`, and every bit after `r` equals the most-recent-write bit.
- **FFL(T, p)**: distribution over length-T flip-flop strings parameterized by `p = (p_w, p_r, p_i)`. Canonical baseline: FFL(T=512, p=(0.1, 0.1, 0.8)).
- **Clean-mode loss**: cross-entropy on `x_{t+1}` only when `x_t = r`. Non-deterministic positions (after `w`/`i`) are not supervised.
- **Glitch rate**: fraction of read positions mispredicted under argmax decoding — the paper's primary metric. Reported on three held-out sets: FFL(0.8) in-dist, FFL(0.98) sparse tail, FFL(0.1) dense tail.
- **Adversary**: optimizer (CMA-ES or similar) that searches over FFL distribution parameters to maximize the trained FFLM's glitch rate.
- **Retrain loop**: after the adversary finds a breaking distribution, retrain on FFL(0.8) ∪ adversarial, then re-probe.

## Conventions

- Python 3.10+, PyTorch, HuggingFace `transformers`.
- All hyperparameters live in `flip_flop/configs/*.yaml` — no magic numbers in code.
- Every run logs config + seed and writes `train_log.jsonl` + `eval_log.jsonl` under `cfg.out_dir`.
- Large checkpoints are gitignored under `results/`.
- Keep `flip_flop/` close to the paper; any deviation from Appendix B.2 should be commented.
- Notebooks are for analysis only, not for running experiments.

## What Makes a Good Adversarial Distribution

Consistent with the paper's R4 finding ("training on rare sequences works by a wide margin"), a useful adversarial distribution:
1. Remains a **valid FFL distribution** (legal flip-flop strings).
2. Produces a **large glitch-rate gap** between the trained Transformer and the LSTM / Bayes-optimal automaton (which should be at 0% error).
3. Is **structured** — e.g. a specific `(p_w, p_r, p_i)` region or biased-bit distribution — not a pathological edge case (p_i → 0 or p_i → 1).
4. Ideally reveals something about the Transformer's inductive bias: long-range vs short-range dependency failure, attention-dilution regimes, etc.

## Development Notes

- `flip_flop/` is the primary working directory going forward. The `src/` tree (linear-regression ICL + CMA-ES adversary) is prior art, kept for reference but not the current focus.
- Always compare the Transformer's glitch rate against the **1-layer LSTM** baseline (`flip_flop/configs/lstm.yaml`) — paper R2 says LSTM reaches 0% error, so it is the "skyline" reference for whether a distribution is legitimately hard or just ill-defined.
- New adversary code goes in `flip_flop/adversary/`; new experiment configs in `flip_flop/configs/`.
- When adding to `flip_flop/`, keep it minimal — small, paper-faithful modules beat heavy abstraction.

## Sweep retrain — union of all training data

To train one model on the union of every family any prior retrain saw,
use `flip_flop/scripts/run_retrain_sweep.py` with `retrain_sweep.yaml`. It
reads each existing `sampler.json` (the ground-truth record of what each
prior retrain trained on), reconstructs the families verbatim via
`family_from_dict` in `flip_flop/adversary/family.py`, and trains on the
union with one shared `replay_frac`. No re-clustering or re-bisection — the
sweep dataset is exactly the families those retrains saw. The merged
`sampler.json` records each family's source path for provenance.

## Open TODOs

- **Family jitter** (`flip_flop/adversary/family.py::ClusterFamily`): the current v1 family is a single α-pulled-back distribution — every sample from the family has identical parameters. Introducing per-sequence jitter (draw params from `N(p_α*, σ²)` where σ is tied to the cluster's observed std) would expose the model to a neighborhood rather than a point. Decide between three options: (a) jitter proportional to `(1 - α)`, (b) jitter proportional to cluster std, (c) no jitter (current). See the family-design discussion thread for context.
