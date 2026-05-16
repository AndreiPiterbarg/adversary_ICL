# Adversarial Flip-Flop Retraining — Results

**Date:** 2026-05-14
**Run dir:** `results/flip_flop/full_from_scratch_adv/`
**Wall-clock (full retrain):** ~115 min on local GPU (10000 steps, batch 16)

## Headline

Training the paper's exact GPT-2 architecture from scratch on a pool of
adversary-discovered FFL distributions (plus the two Liu R4 tails) closes the
attention-glitch gap that Liu et al. 2023 reported on the sparse / dense
distribution tails — without sacrificing in-distribution accuracy.

| metric                  |    baseline | full_adv |  improvement |
|-------------------------|------------:|---------:|-------------:|
| in_distribution (FFL 0.8) |    0.000000 | 0.000000 | tied (perfect) |
| sparse_tail (FFL 0.98)    |    0.059891 | 0.000195 |    **307×**   |
| dense_tail (FFL 0.1)      |    0.015693 | 0.000009 |   **1810×**   |

`baseline` = `results/flip_flop/baseline/model_final.pt`, the paper-faithful
6L/512d/8H GPT-2 trained on canonical FFL(0.8) for 10000 steps.
`full_adv` = same architecture, same step count, same eval harness — only
the training distribution differs.

Raw error counts (identical eval N for both):
- in_distribution:   baseline 0/26523    vs full_adv 0/26523
- sparse_tail:       baseline 21198/353945 vs full_adv 69/353945
- dense_tail:        baseline 5429/345951  vs full_adv 3/345951

## Method

Three phases, end-to-end on a single local GPU:

### 1. Adversary discovery

Five adversary axes search FFL distribution parameters to maximize the
trained baseline's glitch rate while keeping a 1-layer LSTM "skyline"
clean (`lstm_glitch < 0.01`). All adversary outputs live under
`results/flip_flop/adversary/<axis>/adversary_log.jsonl`.

| axis        | strategy   | evals  | best T_glitch | LSTM glitch |
|-------------|-----------|-------:|--------------:|-----------:|
| bit_markov  | CMA-ES    |  3024  |        0.9615 |     0.0000 |
| write_flip  | CMA-ES    |  3024  |        0.5230 |     0.0000 |
| piecewise   | CMA-ES (quick, K=2, budget=200) | 200 | 0.4553 | 0.0004 |
| planted     | grid (decoy, quick 18-cell) |    18  |        0.0000 |     0.0000 |

`bitmarkov` + `writeflip` were already on disk from earlier full runs and
were reused. `piecewise` and `planted_decoy` were re-run with the
"quick" configs (`flip_flop/configs/quick_adversary_*.yaml`) — small CMA
budgets and trimmed grids to fit local wall-clock. The planted-decoy
template never escaped 0% glitch at this budget, so it contributed no
families to the pool.

### 2. Pool → MixedSampler → 32K-sequence dataset

`flip_flop/scripts/pool_breaking_points.py` reads each axis log,
filters to `T_glitch >= 0.1` and `lstm_glitch < 0.01`, dedups by
rounding-then-canonicalizing the distribution config, keeps the top-5
per axis by `T_glitch`, and appends the two Liu R4 tail distributions
(FFL(0.98) sparse, FFL(0.1) dense) as additional families.

Final pool — `results/flip_flop/breaking_points/quick/sampler.json`:

| axis        | n |
|-------------|---|
| bit_markov  | 5 |
| write_flip  | 5 |
| piecewise   | 5 |
| liu_tail    | 2 |
| **total**   | **17** |

`MixedSampler` draws each training sequence i.i.d. either from canonical
FFL(0.8) with probability `replay_frac = 0.5` or uniformly from the 17
families with probability 0.5. `flip_flop/scripts/dump_dataset.py`
materializes 32000 length-512 sequences → `dataset.pt` (131 MB).

### 3. From-scratch retrain on the materialized dataset

`flip_flop/configs/full_from_scratch_adv.yaml` mirrors `baseline.yaml`
verbatim — 6L/512d/8H GPT-2 (~19M params), AdamW (β=0.9/0.999, lr=3e-4,
wd=0.1), 50-step warmup + linear decay to 0 at step 10001, batch 16 ×
10000 steps, clean-mode loss (CE on `x_{t+1}` only when `x_t = r`),
`init_from_ckpt: ""` (from scratch). The only `data:` change is that
training batches come from `dataset.pt` instead of an online FFL(0.8)
sampler; the three held-out eval distributions and their N's match
`baseline.yaml` exactly.

Eval at step 10000 reports `error_rate` over the same 26523 / 353945 /
345951 read-position predictions on each subset as the baseline run.

## Sanity check: quick 2K-step retrain (same dataset)

To prove the dataflow before committing 2h of GPU, we first ran a 2K-step
version (`quick_from_scratch_adv.yaml`).

| metric            | quick_adv (2K) |
|-------------------|---------------:|
| in_distribution   |       0.250241 |
| sparse_tail       |       0.322952 |
| dense_tail        |       0.034932 |

Severely under-trained (this is 1/5 the paper's optimizer budget), so
much worse than baseline on absolute terms — but the pipeline produced
valid outputs at every phase. Full 10K-step run launched after this
passed.

## Wall-clock breakdown (local GPU)

| phase                              | time |
|------------------------------------|-----:|
| piecewise CMA (quick, 200 evals)   | ~10 min |
| planted grid (quick, 18 cells)     | ~2 min |
| pool + dump_dataset (32K × 512)    | ~1 min |
| full retrain (10000 steps)         | ~115 min |
| baseline-vs-adv comparison eval    | (rolled into retrain) |
| **end-to-end from scratch**        | **~130 min** |

## Reproducibility

```bash
# Adversary (axes already on disk for bitmarkov/writeflip; only piecewise + planted needed)
python -m flip_flop.scripts.run_adversary --config flip_flop/configs/quick_adversary_piecewise.yaml
python -m flip_flop.scripts.run_adversary --config flip_flop/configs/quick_adversary_planted.yaml

# Pool + dataset
python -m flip_flop.scripts.pool_breaking_points \
  --logs results/flip_flop/adversary/bitmarkov/adversary_log.jsonl \
         results/flip_flop/adversary/writeflip/adversary_log.jsonl \
         results/flip_flop/adversary/quick_piecewise/adversary_log.jsonl \
         results/flip_flop/adversary/quick_planted_decoy/adversary_log.jsonl \
  --out_dir results/flip_flop/breaking_points/quick \
  --top_k_per_axis 5 --min_t_glitch 0.1
python -m flip_flop.scripts.dump_dataset \
  --run_dir results/flip_flop/breaking_points/quick --n 32000 --seed 12345

# Full retrain
python -m flip_flop.scripts.run_retrain_from_dataset \
  --config flip_flop/configs/full_from_scratch_adv.yaml \
  --dataset results/flip_flop/breaking_points/quick/dataset.pt
```

The pipeline orchestrator at `flip_flop/scripts/run_quick_pipeline.py`
chains phases 2–5 of the quick variant automatically; for the full
retrain, swap step 4's config for `full_from_scratch_adv.yaml`.

## Caveats and honest scope

1. **Single seed.** Both `baseline` and `full_adv` are 1-seed runs.
   The paper trains 3 seeds per condition; rerunning `full_adv` at seeds
   1 and 2 would tighten the claim. The size of the gap (300×–1800×)
   makes seed noise unlikely to flip the sign, but the magnitudes will
   move.
2. **Planted-decoy contributed nothing** at the quick search budget.
   Either widen the grid (use the full `adversary_planted.yaml`) or
   drop the axis from the pool list.
3. **The pool includes the two Liu R4 tails by construction.** Some of
   the win is the Liu-R4 effect that the paper already reported.
   Disentangling "adversary-found vs Liu-supplied" requires an ablation
   that re-runs the retrain with `--no_liu_tails` — recommended next
   step.
4. **`replay_frac = 0.5`** was a guess; it could be lowered (more
   adversarial weight) or raised (more in-dist weight). The fact that
   in_distribution stayed at 0 errors suggests there's slack to push
   `replay_frac` down further and possibly squeeze the tails more.
5. **CMA budget was small for piecewise** (200 evals vs the
   full-config 1500). The full piecewise run typically finds
   `T_glitch ~ 0.6+`; we got `0.4553`. A full-budget piecewise rerun is
   the cheapest improvement.

## Files produced

- Pool: `results/flip_flop/breaking_points/quick/{sampler.json,breaking_points.jsonl,dataset.pt}`
- Model: `results/flip_flop/full_from_scratch_adv/model_final.pt`
- Logs: `results/flip_flop/full_from_scratch_adv/{train_log.jsonl,eval_log.jsonl}`
- Stdout: `results/flip_flop/_pipeline_logs/{piecewise.log,chunk1.log,train_full.log}`

## What's next (if pursuing publication)

- Multi-seed: rerun `full_adv` at seeds 1, 2 to bracket variance.
- Ablation: same retrain with `--no_liu_tails` to isolate the adversary contribution.
- Full piecewise CMA (1500-eval budget, K=4 segments) to push the
  piecewise peak above the bit_markov peak.
- Re-run the attention-position diagnostic
  (`flip_flop/scripts/diagnose_attention_position.py`) on
  `full_adv/model_final.pt` and contrast against `baseline/model_final.pt`
  to show the early-position attention drift (Liu Fig 16d) has shrunk.
