# expressive_adversary

Self-contained implementation of `docs/PLAN_expressive_adversary.md` — one
**feature-conditioned log-linear controller** that replaces the zoo of
hand-coded FFL families. The human supplies an *algebra* (a causal feature set
+ a linear-softmax form); the optimizer composes it. Expressiveness is no longer
bounded by how many parametric families are enumerated.

This folder **does not import from `flip_flop/`**. The data / reference-family /
CMA code is copied verbatim so (a) it cannot perturb anything running against
that tree (e.g. Rung 1), and (b) the subsumption tests compare the controller
against the *exact* project code, not a re-implementation.

## Layout

| file | contents |
|---|---|
| `data.py` | FFL token primitives, **verbatim** copy. `enforce_read_determinism` is the centralized validity gate. |
| `distribution.py` | `FFLDistribution` ABC + reference families (`Stationary`, `BitMarkov`, `WriteFlipRate`, `Piecewise`, verbatim) + **`FeatureControlledFFL`** (the contribution). |
| `encoder.py` | `Encoder` protocol + `DiagonalCMAES` (verbatim) + **`FeatureControlledEncoder`** (24-dim codec). |
| `tests/` | 34 CPU-only tests (no model, no GPU). |

## Run the tests (CPU, tiny)

```bash
cd <repo root>
CUDA_VISIBLE_DEVICES="" python -m pytest expressive_adversary/tests -q
```

Pure numpy + one `torch.from_numpy`; ~8 s; deliberately small (T ≤ 256,
B ≤ 2000). No heavy/GPU work — that is reserved for the running Rung 1.

## The controller

At every instruction slot `k`, a bounded **causal** feature vector `phi_k` is
formed, then:

```
instruction:  z_W = 0 ;  z_R = wR·phi + cR ;  z_I = wI·phi + cI
              q   = softmax([z_W, z_R, z_I])               # (W, R, I)
              p   = floor + (1 - sum(floor)) * q           # affine floor map
free bit:     p(bit=1) = sigmoid(wb·phi + cb)
```

Read-position bits are overwritten by `enforce_read_determinism` in `_finalize`,
so **every sample is valid FFL and the exact 1-layer-LSTM register automaton
stays at 0 % for any weights** — validity and skyline legitimacy are free.

### Features (v1) — 7 of them

`p`, `p²`, `gap_norm`, `nwrites_norm`, `runlen_norm`, `last_write_bit`,
`prev_data_bit`. Caps are stored as fractions of `n_inst`, so the controller is
T-agnostic and round-trips exactly. Each feature is exactly the latent variable
some hand-coded family conditioned on (see the subsumption matrix).

### Structural non-degeneracy (principled anti-edge-case)

The affine floor map guarantees `p_w ≥ p_w_min` and `p_r ≥ p_r_min` at **every**
position **by construction**. The `p_r → 0` read-starvation that turned
`bit_markov` into a degenerate edge case is **unreachable by the
parameterization**, not merely discouraged by the objective. This is the plan's
`π = (1−ε)·softmax + ε·π_floor` in its exact-bound affine form (equivalent with
`ε = sum(floor)`, `π_floor = floor/sum(floor)`). Verified directly in
`test_instruction_probs_respect_floor_and_simplex` (holds even at ±25 logits).

### Parameter count

`3·(d+1) = 3·8 = 24` (`wR,cR | wI,cI | wb,cb`). Comparable to the existing
16-dim `PiecewiseEncoder` regime that already converges under the project's
diagonal CMA-ES; `test_diagonal_cmaes_drives_encoder_unchanged` shows the exact
optimizer drives it with no pipeline change.

## Subsumption matrix (executable proof of the superset claim)

Each witness is a `FeatureControlledFFL.emulate_*` constructor; each row is a
passing test.

| Hand-coded family | Witness | Exactness | What it proves |
|---|---|---|---|
| `Stationary(p_w,p_r,bit_p1)` | `emulate_stationary` | **EXACT** (distributional) | constant-logit special case |
| `WriteFlipRate(flip_rate)` | `emulate_write_flip` | **EXACT** (write-bit law) | bit head on `last_write_bit` |
| `BitMarkov(bit_stay)` | `emulate_bit_markov` | **EXACT** (transition law) | bit head on `prev_data_bit` |
| `Piecewise` position schedule | `emulate_piecewise_ramp` | **APPROX** (exact in basis limit) | position-only conditioning is in-class |

Combinations the human never enumerated (e.g. gap × churn × late-position
simultaneously) require several weight groups non-zero at once and are reachable
**only** by this controller — that is the strict-superset gain.

## Deliberate deviation from the plan (an improvement, flagged)

`docs/PLAN_expressive_adversary.md` §1.1 lists a **6-feature / 21-param** v1.
This implementation uses **7 features / 24 params** — it adds `prev_data_bit`.

Reason: with only `last_write_bit`, the controller exactly subsumes
`Stationary` and `WriteFlipRate` but only *approximates* `BitMarkov` (whose
chain is over consecutive *data* bits across all slots, not write bits).
Adding `prev_data_bit` makes **BitMarkov exactly subsumed** while staying inside
the plan's stated dimensionality envelope (the plan itself anchors "≈24 dims ≈
PiecewiseEncoder's 16" and stages richer features in v2). Net: a strictly
stronger, fully-honest subsumption proof at no feasibility cost. `gap_cap_frac`
etc. are fixed structural config (not searched), so degeneracy stays unreachable
throughout any future search.

## Wiring status (wired into the real pipeline)

The controller is a first-class citizen of the `flip_flop` adversary pipeline.
All edits are **purely additive and import-guarded** — a path/env failure cannot
break the existing pipeline (e.g. a running Rung 1); worst case the family is
just unavailable. Verified: **38 package tests + 63 existing flip_flop adversary
tests all green on CPU**, zero regression.

| wiring point | change |
|---|---|
| `flip_flop/adversary/distribution.py` | registers `feature_controlled` in `REGISTRY` (imports the single source of truth here, under try/except) → the whole `pool_breaking_points → dump_dataset → run_retrain_from_dataset → eval_headtohead` chain reconstructs it via `FFLDistribution.from_dict` unchanged |
| `flip_flop/adversary/search.py` | exposes `FeatureControlledEncoder` (guarded import) |
| `flip_flop/adversary/run.py` | `_make_encoder` gains the `feature_controlled` branch |
| `flip_flop/configs/` | `quick_adversary_feature_controlled.yaml` (Phase 1) + `adversary_feature_controlled.yaml` (Phase 2), mirroring the regret-CMA convention |

`tests/test_pipeline_wiring.py` proves the whole path on CPU with **no model /
no GPU**: config parses → `run._make_encoder` returns the 24-dim encoder →
flip_flop's own `FFLDistribution.from_dict` round-trips the controller through
flip_flop's `REGISTRY` (the exact pool/dump/retrain reconstruction) → the
rebuilt sampler emits valid FFL.

## Run Phase 1 (only when the GPU is free)

This is the single remaining model-in-the-loop step; it needs the frozen
transformer/LSTM + GPU, so it must wait until the running Rung 1 job is verified
done (one-GPU-job rule, `PLAN_beat_liu_r4.md` §7):

```bash
python -m flip_flop.scripts.run_adversary \
  --config flip_flop/configs/quick_adversary_feature_controlled.yaml
```

Then inspect `descriptor()` in the adversary log: if mass concentrates on
position/gap features, the controller **rediscovered a breaking regime with no
hand-specified axis** — the headline automation result. Phase 2 swaps in
`adversary_feature_controlled.yaml` and runs the unchanged
pool→dump→retrain→`eval_headtohead` chain.

## What is intentionally NOT run

No model forward pass and no CMA *search* execution — that is Phase 1 above and
is GPU-gated. Everything up to that point (controller, encoder, pipeline wiring,
configs, correctness + subsumption + wiring proofs) is complete and tested.
