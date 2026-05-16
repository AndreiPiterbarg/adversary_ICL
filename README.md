# Adversarial Flip-Flop Language Modeling

Automated discovery of structured failure modes ("attention glitches") in
Transformer flip-flop language models (FFLMs), following Liu et al. 2023
(["Exposing Attention Glitches with Flip-Flop Language Modeling"](https://arxiv.org/abs/2306.00946)).

A baseline FFLM is trained on the canonical FFL(0.8) distribution; a CMA-ES
adversary then searches a parametric family of flip-flop distributions for the
input regime that maximizes the model's read-error rate while a 1-layer LSTM
("Bayes skyline") stays at 0% error. The model is then retrained on the
discovered distribution and re-probed.

## Layout

| Path | What |
|------|------|
| `flip_flop/data.py model.py train.py eval.py` | core: sampler, models, training loop, glitch-rate eval |
| `flip_flop/adversary/` | CMA-ES distribution search + objective + held-out eval sets |
| `flip_flop/analysis/` | LSTM/Transformer state-probing diagnostics |
| `flip_flop/configs/` | all hyperparameters (`*.yaml`) — no magic numbers in code |
| `flip_flop/scripts/` | entry points (`python -m flip_flop.scripts.<name>`) |
| `docs/PLAN_beat_liu_r4.md` | **current plan of record — start here** |
| `docs/RESULTS_adversary_retrain.md` | results log to date |
| `docs/ff_paper.txt` | Liu et al. 2023, full reference text |
| `results*/` | checkpoints & logs (gitignored, kept locally) |

Project conventions and key concepts are documented in
[`CLAUDE.md`](CLAUDE.md).

## Setup

```bash
pip install -r requirements.txt   # Python 3.10+, PyTorch, HF transformers
```

## Quickstart

All entry points are config-driven modules under `flip_flop/scripts/`:

```bash
# Train the paper-faithful baseline FFLM (6L/512d/8H GPT-2, 10k steps)
python -m flip_flop.scripts.run_baseline --config flip_flop/configs/baseline.yaml

# Train the 1-layer LSTM skyline (should reach 0% glitch)
python -m flip_flop.scripts.run_baseline --config flip_flop/configs/lstm.yaml

# Run the adversary search against a trained checkpoint
python -m flip_flop.scripts.run_adversary --config flip_flop/configs/adversary_piecewise.yaml

# Retrain from a materialized adversarial dataset
python -m flip_flop.scripts.run_retrain_from_dataset \
  --config flip_flop/configs/full_from_scratch_adv.yaml \
  --dataset results/flip_flop/breaking_points/quick/dataset.pt

# Attention-position diagnostic on a checkpoint
python -m flip_flop.scripts.diagnose_attention_position \
  --ckpt results/flip_flop/<run>/model_final.pt --family piecewise_c00
```

Every run writes `config`, `train_log.jsonl`, and `eval_log.jsonl` under its
configured `out_dir` in `results/`.

## Tests

```bash
python -m pytest flip_flop/adversary/tests -q
```
