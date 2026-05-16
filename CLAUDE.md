# Adversarial Flip-Flop Language Modeling

## Project Goal

Find **structured, interesting failure modes** of Transformer FFLMs by training an adversary to discover input distributions where a trained FFLM breaks down, then retrain on the adversarial distribution. The end goal is a publishable result showing *when and why* Transformer FFLMs exhibit "attention glitches" on specific, learned-worst-case flip-flop distributions.

### Core Idea

Train → search → retrain loop, applied to the flip-flop toy from Liu et al. 2023:

1. **Train** an FFLM Transformer on a base distribution (canonical FFL(0.8)).
2. **Adversary** searches over a parametric family of flip-flop distributions — instruction probabilities `(p_w, p_r, p_i)`, data-bit biases, non-stationary / per-position probabilities — to maximize the trained model's read-error (glitch) rate.
3. **Retrain** on the adversarial distribution (or a mixture with FFL(0.8)) and measure how the glitch rate moves.

Liu et al. established the existence of breaking tails (FFL(0.98), FFL(0.1)) by hand and showed (R4) that training on rare sequences closes the gap. Our contribution is to automate that discovery — letting an optimizer find the worst distribution inside a structured search space, then close it.

**The current plan of record is [`docs/PLAN_beat_liu_r4.md`](docs/PLAN_beat_liu_r4.md).** It supersedes all earlier plans; read it before starting work. Results to date are logged in [`docs/RESULTS_adversary_retrain.md`](docs/RESULTS_adversary_retrain.md).

## Reference Implementation

- **Paper**: "Exposing Attention Glitches with Flip-Flop Language Modeling" (Liu et al. 2023, arXiv:2306.00946). Full text in `docs/ff_paper.txt`.
- **Data release**: https://huggingface.co/datasets/synthseq/flipflop
- FFLM training in `flip_flop/` follows Appendix B.2 of the paper: 6-layer / 512-dim / 8-head GPT-2 baseline (~19M params), AdamW (β=0.9/0.999, lr=3e-4, wd=0.1), 50-step warmup + linear decay to 0 at step 10001, batch 16 × 10000 steps, "clean-mode" loss on read positions only.

## Repository Structure

```
adversary_ICL/
├── CLAUDE.md                       # this file — project context & conventions
├── README.md                       # human-facing overview + quickstart
├── requirements.txt
├── docs/                           # all prose: plan, results, reference paper
│   ├── PLAN_beat_liu_r4.md         # CURRENT plan of record
│   ├── RESULTS_adversary_retrain.md# results log to date
│   └── ff_paper.txt                # Liu et al. 2023, reference text
└── flip_flop/                      # the entire codebase — pure Python
    ├── data.py                     # FFL(T, p) sampler, validity, write_read_gaps
    ├── model.py                    # GPT-2 Transformer + 1-layer LSTM
    ├── train.py                    # training loop (paper Appendix B.2)
    ├── eval.py                     # clean-mode loss + glitch-rate eval
    ├── adversary/                  # CMA-ES distribution search
    │   ├── distribution.py  family.py  heldout.py  io.py
    │   ├── mixture_sampler.py  objective.py  r4_sampler.py
    │   ├── run.py  search.py
    │   └── tests/                  # pytest suite
    ├── analysis/                   # LSTM/Transformer state-probing diagnostics
    │   └── ablation.py  lstm_state.py  probe.py  transformer_state.py
    ├── configs/                    # all hyperparameters (*.yaml)
    └── scripts/                    # entry points (run_baseline, run_adversary, …)

results/  results_full_loop/  results_tier1_v2/   # checkpoints & logs (gitignored)
```

## Key Concepts

- **FFLM**: autoregressive LM over the vocabulary `{w, r, i, 0, 1}`. `w`/`r`/`i` are write/read/ignore instructions; `0`/`1` are data bits. Valid strings alternate instruction/data, start with `w`, end with `r`, and every bit after `r` equals the most-recent-write bit.
- **FFL(T, p)**: distribution over length-T flip-flop strings parameterized by `p = (p_w, p_r, p_i)`. Canonical baseline: FFL(T=512, p=(0.1, 0.1, 0.8)).
- **Clean-mode loss**: cross-entropy on `x_{t+1}` only when `x_t = r`. Non-deterministic positions (after `w`/`i`) are not supervised.
- **Glitch rate**: fraction of read positions mispredicted under argmax decoding — the paper's primary metric. Reported on three held-out sets: FFL(0.8) in-dist, FFL(0.98) sparse tail, FFL(0.1) dense tail.
- **Adversary**: CMA-ES search over FFL distribution parameters that maximizes the trained FFLM's glitch rate while keeping a 1-layer LSTM "skyline" clean.
- **Retrain loop**: after the adversary finds a breaking distribution, retrain on FFL(0.8) ∪ adversarial, then re-probe.

## Conventions

- Python 3.10+, PyTorch, HuggingFace `transformers`.
- All hyperparameters live in `flip_flop/configs/*.yaml` — no magic numbers in code.
- Every run logs config + seed and writes `train_log.jsonl` + `eval_log.jsonl` under `cfg.out_dir`.
- Large checkpoints / run outputs are gitignored under `results*/`.
- Keep `flip_flop/` close to the paper; any deviation from Appendix B.2 should be commented.

## What Makes a Good Adversarial Distribution

Consistent with the paper's R4 finding ("training on rare sequences works by a wide margin"), a useful adversarial distribution:
1. Remains a **valid FFL distribution** (legal flip-flop strings).
2. Produces a **large glitch-rate gap** between the trained Transformer and the LSTM / Bayes-optimal automaton (which should be at 0% error).
3. Is **structured** — e.g. a specific `(p_w, p_r, p_i)` region or biased-bit distribution — not a pathological edge case (p_i → 0 or p_i → 1).
4. Ideally reveals something about the Transformer's inductive bias: long-range vs short-range dependency failure, attention-dilution regimes, etc.

## Development Notes

- `flip_flop/` is the only working tree. Keep additions minimal — small, paper-faithful modules beat heavy abstraction.
- Always compare the Transformer's glitch rate against the **1-layer LSTM** baseline (`flip_flop/configs/lstm.yaml`) — paper R2 says LSTM reaches 0% error, so it is the "skyline" reference for whether a distribution is legitimately hard or just ill-defined.
- New adversary code goes in `flip_flop/adversary/`; new experiment configs in `flip_flop/configs/`.
- Run the test suite before relying on a change: `python -m pytest flip_flop/adversary/tests -q`.

## Attention-position diagnostic

`flip_flop/scripts/diagnose_attention_position.py` measures, per (layer, head), how much attention each read places on the ideal target (`t*+1`), the early window `[0, 8)`, and the most-recent read — split CORRECT vs WRONG and by piecewise segment. Run on every newly trained checkpoint to track whether the paper's drift signature (Prop 4 / Fig 16d, concentrated in segment 3 of `piecewise_c00`) is shrinking:

```
python -m flip_flop.scripts.diagnose_attention_position \
  --ckpt results/flip_flop/<run>/model_final.pt --family piecewise_c00
```

Look at `summary.md` in `<ckpt_dir>/attention_position/piecewise_c00/`: lower `mass_early` and higher `mass_correct` on WRONG-segment-3 reads = improvement.
