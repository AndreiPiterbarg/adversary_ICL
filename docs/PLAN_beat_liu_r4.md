# Plan: Beat Liu R4 on the Non-Stationary Axis It Cannot Cover

Final plan of record. Supersedes all earlier versions. Synthesis of a
12-agent + 9-agent investigation. The pivot: we do **not** chase a new way to
beat Liu — we **rigorously prove and frame the beat we already produced**, on
the non-stationary / segment-drift axis, with the one control that makes it
airtight.

---

## 0. The claim (exact wording, this is what every step serves)

> An automated regret adversary produces a training dataset that **ties Liu R4
> everywhere Liu R4 is strong** (Liu's own stationary tails and an unseen
> Periodic axis — both at the LSTM Bayes skyline) **and strictly dominates Liu
> R4 on the non-stationary segment-drift axis Liu's stationary recipe
> structurally cannot emit** (Liu ≈ 4.6 % glitch, ours ≈ skyline, LSTM = 0 %).
> The axis was discovered automatically (no human prior), and the
> non-stationary structure — not merely broader stationary coverage — is
> necessary to close it.

This is **dominance, not a trade-off**: ≥ Liu everywhere, > Liu on the axis it
misses. If we are ever *worse* than Liu on its home turf, the result dies — so
the "tie everywhere else" half is tested with equal rigor to the "win" half.

Why the axis exists: Liu R4 retrains only on **stationary** distributions
(instruction probs constant across the sequence). A non-stationary input — long
sparse-write opening that plants a bit, then a distant read burst — is
**unrepresentable by Liu's sampler**, so a Liu-trained model never trains on it
and its attention drifts (the paper's own Fig-16d segment-3 glitch). The LSTM
(the exact register automaton) stays at 0 % there, proving the distribution is
valid and the failure is a genuine Transformer attention glitch.

---

## 1. Assets that already exist (no work needed)

Trained, on disk, verified:

| name | path | role |
|---|---|---|
| `B_liu` | `results/flip_flop/full_from_scratch_liu_only/model_final.pt` | **the competitor** (Liu R4: FFL0.98+FFL0.1+FFL0.8, 10k steps) |
| `full_adv` | `results/flip_flop/full_from_scratch_adv/model_final.pt` | adversary-pool model (closes piecewise in prior data) |
| `C_regret` | `results/flip_flop/full_from_scratch_regret/model_final.pt` | regret-objective model |
| `beat_R1` | `results/flip_flop/beat_liu_loop/round1/model/model_final.pt` | regret+MixtureTwoStationary pool model |
| `baseline` | `results/flip_flop/baseline/model_final.pt` | FFL(0.8)-only reference |
| `LSTM` | `results/flip_flop/lstm/model_final.pt` | Bayes skyline |

Frozen pre-registered held-out configs already in
`flip_flop/adversary/heldout.py`: `HELDOUT_PIECEWISE` (the beat axis),
`HELDOUT_ANTICORR` (proven dead, kept for completeness), `LIU_TAILS`.

Existing infra reused: `eval_fair.py`, `evaluate_dataset` (T-agnostic),
`write_read_gaps`, `load_frozen_model` (rebuilds each model from its own
config), `diagnose_attention_position.py`, `pool_breaking_points.py`,
`run_retrain_from_dataset.py`.

So this is **mostly measurement plus exactly one new training run.**

---

## 2. Code changes (small, additive — do not modify existing callers)

### Change A — per-sequence eval (≈15 lines, `flip_flop/eval.py`)
Add `evaluate_dataset_per_seq(model, tokens, batch_size, device)` returning
two `(N,)` arrays: `errors_per_seq`, `reads_per_seq`. Pure refactor of the
existing loop (same masking/logits); the existing `evaluate_dataset` stays
untouched and can call it. Needed because a defensible claim at ~1e-5 error
requires a **paired bootstrap over sequences**, not aggregate means.

### Change B — head-to-head eval script (≈120 lines, new `flip_flop/scripts/eval_headtohead.py`)
- Loads `{baseline, B_liu, full_adv, C_regret, beat_R1, stationary_control,
  LSTM}` (skips any not present).
- For each frozen eval distribution (see §3) samples **N=16384** with a fixed
  seed; **same sequence tensor for every model** (paired).
- Uses Change A to get per-seq counts.
- Drops a distribution if `LSTM_err > 5e-4` (skyline-invalid → report, exclude
  from headline).
- Paired bootstrap: 10 000 resamples of sequence indices (BCa 95 % CI) on the
  **pooled error-rate difference** `B_liu − model` per distribution; also a
  Wilcoxon signed-rank cross-check on per-distribution rates across the band.
- Emits `summary.md` (table: model × distribution → err, 95 % CI, vs-LSTM
  ratio, vs-B_liu CI) + `per_distribution.jsonl`.

### Change C — pre-registered piecewise robustness band (≈15 lines, `flip_flop/adversary/heldout.py`)
Add `HELDOUT_PIECEWISE_BAND`: 16 configs = `HELDOUT_PIECEWISE` with the two
segment-boundary fractions and `p_r` jittered on a **fixed seeded grid**.
Makes the claim "region-level," not single-point (the diagnostic already
showed 59/60 region robustness — this freezes it for the headline eval).
Frozen, never fed to any adversary.

### Change D — stationary-only control dataset (≈30 lines, new `flip_flop/scripts/build_stationary_control.py`)
Writes a `sampler.json` whose families are **only `Stationary`**: the 2 Liu
tails + a fixed 8-point grid of `Stationary(p_w=p_r=(1−p_i)/2)` for
`p_i ∈ {0.02,0.05,0.15,0.30,0.50,0.70,0.90,0.97}`, `replay_frac=0.5`.
Explicitly **zero** piecewise/periodic/mixture/anticorr families. This is the
scientific control: "does more *stationary* variety alone close the
non-stationary break, or is the adversary's non-stationary discovery
necessary?"

### Change E — control retrain config (1 file, copy)
`flip_flop/configs/full_from_scratch_stationary_control.yaml` = exact copy of
`full_from_scratch_regret.yaml` with `out_dir:
results/flip_flop/full_from_scratch_stationary_control`. Architecture/optimizer
identical to B_liu so the comparison is fair.

No architecture changes anywhere. No new search machinery.

---

## 3. The frozen evaluation set (pre-register, commit to git BEFORE Rung 1)

Every model scored on the **same** distributions, T=512, N=16384, eval_seed=42:

1. `HELDOUT_PIECEWISE` + `HELDOUT_PIECEWISE_BAND` (17 configs) — **the beat
   axis** (non-stationary; Liu's sampler cannot emit it).
2. `liu_ffl_098`, `liu_ffl_010` — Liu's own tails (the "tie where Liu is
   strong" check).
3. A 20-config Periodic set from `eval_fair.py`'s full-simplex prior
   (seed 42) — unseen-axis "tie" check.
4. `HELDOUT_ANTICORR` — completeness (expected ≈0 for all; proven dead).

LSTM run on all; any distribution with `LSTM_err > 5e-4` is dropped and
reported (none expected — these are all valid FFL).

---

## 4. Experiments (gated rungs — cheapest decisive test first)

### Rung 0 — Does the beat hold at scale? (NO training, ~45 min)
Run Change B over the §3 set on the **already-trained** models
{baseline, B_liu, full_adv, C_regret, beat_R1, LSTM}.

- **PASS (proceed):** `B_liu_err(piecewise band) ≥ 100 × LSTM_err`, bootstrap
  95 % CI on `B_liu − full_adv` excludes 0 and favors full_adv; **and**
  full_adv vs B_liu CI within δ=2e-4 on Liu tails *and* Periodic (the
  dominance/tie condition).
- **FALSIFY (stop, publish negative):** B_liu ≈ LSTM on the piecewise band at
  N=16384 (the earlier 4.6 % was a small-N fluke) → no beat. This is itself a
  clean, reportable result and costs only 45 min.

### Rung 1 — Is the adversary necessary? (ONE new ~2 h training + ~30 min eval)
Only if Rung 0 passes. Run Change D → `dump_dataset` →
`run_retrain_from_dataset --config full_from_scratch_stationary_control.yaml`
(10k steps, from scratch, identical arch to B_liu). Then re-run Change B
including `stationary_control`.

- **AIRTIGHT:** `stationary_control` still fails the piecewise band
  (≥ 10 × LSTM) → broader *stationary* coverage does **not** close it; the
  adversary's **non-stationary** discovery is necessary. Strongest claim.
- **HONEST DOWNGRADE:** `stationary_control` closes it → claim shrinks to
  "Liu's 2 hand-picked points are insufficient; the adversary automates
  finding a sufficient set." Still publishable, weaker.

### Rung 2 — Why does Liu fail? (NO training, ~20 min)
`python -m flip_flop.scripts.diagnose_attention_position --ckpt <B_liu> --family piecewise_c00`
and same for `full_adv` (and `stationary_control`). Expect: high segment-3
attention-drift mass (Fig-16d signature) for B_liu, low for full_adv. The
mechanistic "why," corroborating Rungs 0–1.

### Rung 3 — Final headline numbers (OPTIONAL, ~6 h, only if publishing)
Purpose-built clean pairing for a 3-seed CI: pool = {Liu 2 tails + the
adversary's non-stationary discoveries}, retrain seeds 0/1/2; same for a
3-seed B_liu. Only needed if reviewers demand multi-seed; `full_adv`
already substantiates the effect for the internal result. Defer by default.

---

## 5. Pre-registration checklist (freeze to git before Rung 1)

Commit, with a timestamped note, BEFORE training `stationary_control`:

- [ ] `heldout.py` incl. `HELDOUT_PIECEWISE_BAND` (the 16 jitter configs +
      seed).
- [ ] Eval: N=16384, eval_seed=42, LSTM-drop threshold 5e-4.
- [ ] Bootstrap: 10 000 BCa paired resamples; Wilcoxon concordance required.
- [ ] Thresholds verbatim: beat = `B_liu ≥ 100×LSTM` on band **and** CI(B_liu−full_adv) excludes 0; tie = CI within δ=2e-4 on Liu tails+Periodic;
      control "necessary" = stationary_control ≥ 10×LSTM on band.
- [ ] The `stationary_control` family list (the 8 p_i grid points) — fixed so
      it cannot be tuned to fail.
- [ ] Falsification sentence verbatim: *"If B_liu ≤ 5×LSTM on the piecewise
      band at N=16384, there is no non-stationary beat; report negative."*

---

## 6. Explicitly NOT doing (every agent flagged these)

- **No** T>512 / length extrapolation — trivial learned-PE artifact (4 agents
  independent).
- **No** index-hints / scratchpad / since-last-write tokens — changes the
  task; not a fair comparison to Liu.
- **No** further model-in-the-loop rounds — R1 already hit the noise floor.
- **No** new objective/diversity machinery — the regret + read-floor +
  descriptor-grid we have is sufficient; this plan is measurement + 1 control.
- **No** architecture changes — fairness vs Liu requires identical 6L/512d/8H.

---

## 7. Operational rules (from the GPU-contention incident)

- **Exactly one GPU job at a time.** Verify a process is dead via
  `tasklist`/`nvidia-smi`, never a buffered log, before any (re)launch.
- Every long run: unbuffered (`python -u`), nohup to a logfile, a watcher that
  fires on done **or** error **or** train-log stall > 8 min.
- `set -e` on any chained script so a failed rung never feeds the next.

---

## 8. Wall-clock & order

| step | training? | wall-clock | decision |
|---|---|---|---|
| Changes A–E (code) | no | ~30 min | — |
| Rung 0 (scale eval) | no | ~45 min | PASS → Rung 1; FALSIFY → stop+write |
| pre-registration commit | no | ~5 min | gate before Rung 1 |
| Rung 1 (control train+eval) | **1 run** | ~2.5 h | AIRTIGHT / DOWNGRADE |
| Rung 2 (attention diagnostic) | no | ~20 min | mechanism |
| Rung 3 (3-seed headline) | optional | ~6 h | only if publishing |

**Core result: ~1 GPU-day, one training run.** Rung 0 is the cheap decisive
gate — if the beat isn't real at scale we know in 45 minutes.

---

## 9. Deliverables

1. `flip_flop/scripts/eval_headtohead.py` + `flip_flop/eval.py` per-seq fn.
2. `HELDOUT_PIECEWISE_BAND` in `heldout.py` (pre-registered, committed).
3. `build_stationary_control.py` + the control config + trained
   `stationary_control` model.
4. `results/flip_flop/eval_headtohead/summary.md` — the headline table
   (model × distribution, err, 95 % CI, vs-LSTM, vs-B_liu CI).
5. Attention-drift diagnostic outputs for B_liu vs full_adv.
6. `RESULTS_beat_liu.md` — the three-part story: **automation found the axis →
   Liu R4 fails on it (table+CI) → our dataset closes it & ties elsewhere
   (dominance) → why (attention drift)**, with the honest control outcome and
   the falsification result if it went that way.
