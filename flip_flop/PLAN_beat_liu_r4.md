# Plan: An Adversary Whose Dataset Beats — or Provably Matches — Liu R4

Synthesis of a 12-agent investigation (theory, UED/DRO/coreset literature,
adversarial-training pitfalls, search-space audit, evaluation design, devil's
advocate) plus a second critical pass that recovered two mechanisms wrongly
binned as overengineering. This is the canonical, self-contained plan.

---

## 0. Framing: what "win" means here (read first)

Three agents independently proved a hard constraint:

- Liu's two tails — **FFL(0.98)** sparse (`p_w=p_r=0.01`) and **FFL(0.1)**
  dense (`p_w=p_r=0.45`), bits i.i.d. uniform — are the **endpoints of the
  write→read-gap difficulty axis**. Training on both forces the exact
  flip-flop automaton; B_liu generalizes near-perfectly (≈6 errors / 2.7M
  predictions, even on the unseen Periodic axis).
- An error/regret-maximizer over stationary `(p_w, p_r)` **provably crawls to
  those same endpoints**. You cannot beat an endpoint-spanning cover from
  inside its convex hull. On Liu's own axis, error is already ~1e-6 — "beating"
  it is statistically meaningless.

So there are two distinct, legitimate targets. We pursue **both**, with
different mechanisms:

1. **PRIMARY — Match Liu automatically (high confidence; a methods result).**
   Show a regret-driven adversary with *no human prior* rediscovers the
   endpoint structure and equals Liu's generalization. Contribution =
   automation, not a lower number. Mechanism: §2.A + §2.C.
2. **STRETCH — Beat Liu off-axis (higher risk; the real headline if it lands).**
   Liu's tails are bit-symmetric and i.i.d., so a model trained on them can
   still pass them with a "predict the majority / most-recent recent bit"
   heuristic — it just happens to coincide with the automaton on i.i.d. data.
   A distribution whose target bit is **adversarially anti-correlated with
   that heuristic**, evaluated across *both* gap extremes, has provable
   residual error for Liu's model but not for a model trained on it.
   Mechanism: §2.B (this is the load-bearing piece; without it the plan has
   no path to its own headline).

A plan delivering only (1) is still publishable. (2) may return null — that is
an acceptable, reportable scientific outcome, not a failure.

---

## 1. The objective (implemented — keep, with one addition)

The regret + read-density-floor objective already shipped is exactly what the
UED (PAIRED regret), DRO (regret-DRO; Duchi–Namkoong), and coreset (RHO-LOSS
reducible loss) literatures prescribe. Final form:

```
gap        = mean write→read gap of the sampled batch        # free, from tokens
read_dens  = mean reads per sequence                          # free, from tokens
reject (is_valid=False) if read_dens < 3.0  OR  gap < 4        # hard gates
fitness    = (T_glitch − LSTM_glitch)                          # regret base
             − 5.0 · max(0, LSTM_glitch − 1e-3)                # soft validity hinge
```

- Regret zeroes out ill-posed regions automatically (LSTM also fails ⇒
  regret ≈ 0).
- `read_dens ≥ 3.0` matches FFL(0.98)'s ~3.5 reads/seq — blocks the `p_r→0`
  exploit while still admitting genuinely sparse-but-valid tails.
- `gap ≥ 4` rejects trivially short-range distributions (not glitches).
- Soft hinge (not the old hard reject) preserves near-boundary signal.

**ADD:** a `gap` field to `FitnessResult` and a `write_read_gaps(tokens)`
helper in `flip_flop/data.py` (~12 lines: for each `R` position, distance back
to the most recent `W`; return mean and p90). Zero extra forward passes.

---

## 2. The four first-class mechanisms (all small, no heavy infra)

### A. `MixtureTwoStationary` — search Liu's own family (PRIMARY win)

**Why:** the search-space audit found CMA-ES *only ever searched the Piecewise
axis*; it never searched Stationary, so it could not have rediscovered Liu's
endpoints even though the family contains them. The failure was search
coverage, not representational capacity.

**What:** a distribution = two free stationary endpoints `(p_w^A,p_r^A)`,
`(p_w^B,p_r^B)` mixed per-sequence by Bernoulli(`λ`). A ~7-dim CMA encoder:
two softmax-3 simplex triplets + one sigmoid for `λ`. `MixtureFamily` already
exists in `flip_flop/adversary/family.py`; `cma_search` already takes a
generic encoder (only the type hint widens).

**Size:** ~25–30 lines across `distribution.py` (new `MixtureTwoStationary` +
registry entry) and `search.py` (new `StationaryMixtureEncoder`).

**Expected outcome:** CMA slides the two endpoints to the model's true worst
gap regime. If it converges near `(0.01,0.01)`/`(0.45,0.45)`, that *is* the
automation result — Liu's hand-picked points recovered with no prior.

### B. `AntiCorrelatedBits` — the axis that can actually beat Liu (STRETCH)

**Why this is load-bearing:** §0 proved you cannot beat Liu inside Liu's
family. The only residual slack is the bit distribution: Liu's tails use i.i.d.
uniform bits, so "copy the majority of recent write-bits" or "copy the most
recent data token" are near-optimal there *by coincidence*. Break that
coincidence and Liu's model has provable residual error; a model trained on it
does not. `Piecewise`/`Periodic`/`MixtureTwoStationary` **structurally cannot
express this** (all are position-varying `(p_w,p_r)` with i.i.d. bits).

**What:** a valid FFL distribution identical to `Stationary` in instruction
sampling (free `p_w, p_r` — keeps the full gap axis), but the **written bit
stream is adversarial against recency/majority**: with strength `ρ`, the bit
at each write is set to the *minority* of the previous `k` written bits
(k fixed = 3) with probability `ρ`, else i.i.d. uniform. Read determinism is
still enforced (`enforce_read_determinism`), so the **automaton is still
exactly correct and the LSTM stays ≈0** — only heuristic shortcuts fail. This
is a *structured, valid* hard distribution, not a pathology.

**Encoder (CMA):** 4 dims — softmax-3 over `(p_w,p_r,p_i)` + sigmoid(`ρ`).
Searched jointly with the gap axis so it can place the anti-correlated bits at
*both* gap extremes simultaneously (the theory agent's exact prescription).

**Size:** ~25 lines in `distribution.py` + registry + a 4-dim encoder.

**Falsifiable prediction:** B_liu has non-trivial error on a held-out
`AntiCorrelatedBits` distribution while the LSTM stays ≈0; a model trained
including it closes that gap without losing Tier-C. If B_liu is *also* ≈0
here, the stretch goal is genuinely impossible — report that and stop (it is a
clean negative result: Liu's i.i.d. tails already induce the full automaton).

### C. Descriptor-grid pooling (coverage; replaces top-K-by-fitness)

**Why:** top-K-by-fitness collapsed 15 "families" into ~3 near-duplicate
corners. Liu wins with 2 *spread* points. The CVaR/DRO result formalizes this:
optimizing the worst α-fraction (not the single max) provably yields a diverse
*covering* set, not one point — and the Sener–Savarese core-set bound says
generalization scales with covering radius, not sample count.

**What:** replace `_dedup_top_k_per_axis` in `pool_breaking_points.py` with a
Quality-Diversity grid: bin every discovered distribution by **two free
descriptors** —

- write→read gap p90: 4 bins `[4,16) [16,64) [64,256) [256,T]`
- read density: 2 bins `[3,8) [8,∞)`

— keep the single **highest-regret** distribution per non-empty cell, cap at
4–8 families. This *structurally guarantees* the gap-axis extremes (Liu's
endpoint regimes) survive even when their individual regret is lower than
interior degenerate modes. Keep the Liu-tail append so the mixture is always a
strict superset of R4 (safety floor: cannot do worse than Liu in expectation).

**Size:** ~20–30 lines, no extra model evals.

### D. Two-checkpoint robustness filter (transferability; cheap)

**Why:** the deepest diagnosed failure is the adversary overfitting *one
frozen checkpoint's* idiosyncrasies — non-transferable. The full fix is the
iterative loop (§3), but a cheap partial fix attacks it directly and is worth
having independently.

**What:** train **one** second-seed baseline once (`seed=1`, ~2h, one-time,
reused for all rounds and both eval passes). Then a pooled family is admitted
only if its glitch gap **replicates on both baseline checkpoints** (regret
computed against each; keep the *min* regret as the score, or require both
≥ threshold). At search time this is +1 forward pass per candidate (we already
run a second model — the LSTM — every candidate, so the harness exists);
the only real cost is the one-time 2h baseline-seed-1 retrain.

**Size:** ~15 lines in `objective.py` (accept an optional second transformer,
take `min` regret) + reuse the existing retrain config with `seed: 1`.

---

## 3. The cheap iterative loop (model-in-the-loop, 2 rounds)

One-shot adversary-vs-frozen-baseline finds *that checkpoint's* failures, not
transferable hard distributions. §2.D mitigates this cheaply; the loop closes
it. Per round *k* against the current model `M_{k−1}`:

1. Quick adversary (regret obj; search space = `MixtureTwoStationary` ∪
   `AntiCorrelatedBits`) vs `M_{k−1}` — ~10 min.
2. Pool = descriptor-grid over **all** logs r1..rk + Liu tails (append-only;
   the covering set only grows) — ~3 min.
3. **From-scratch** full retrain (10k steps, paper-faithful) — ~2 h.
   From-scratch (not fine-tune) keeps the final number comparable to Liu's
   from-scratch model and removes forgetting confounds.
4. Eval on the pre-registered held-out axes + LSTM skyline — ~10 min.

Budget: ≈2.5 h/round. **2 rounds + 1 final clean retrain ≈ 7 h + the one-time
seed-1 baseline (2 h) ≈ 9 h** — fits one day. Stop when round-over-round
held-out improvement < 0.5% or the held-out number reaches the Liu-only
model's number (§4). This is fictitious play on the minimax game; its average
provably dominates any *fixed* mixture, including Liu's pair.

---

## 4. Pre-registered evaluation (commit to git BEFORE Rung 2)

- **Eval set:** widened full-simplex Tier-C Periodic prior (seed 42) — dump
  the ≥30 configs to JSON and freeze. **Two** held-out adversarial axes,
  neither fed to the adversary objective: (i) `piecewise` Tier-H (the paper's
  segment-3 drift signature), (ii) a held-out `AntiCorrelatedBits` config
  (the axis the stretch goal targets — pre-commit its params). Liu's three
  distributions frozen.
- **Success — PRIMARY (match):** on Tier C, adversary-model mean error ≤
  B_liu mean error + δ, δ = 2e-4 (non-inferiority), AND CMA's discovered
  endpoints lie within 0.05 (in `p_w,p_r`) of Liu's — i.e. it *rediscovered*
  them.
- **Success — STRETCH (beat):** B_liu has error > 10·LSTM on the held-out
  `AntiCorrelatedBits` axis while the adversary-trained model is ≤ 2·LSTM
  there, AND non-inferiority on Tier C still holds. Paired bootstrap (10k
  resamples) 95% BCa CI on summed error counts excludes 0.
- **Statistics:** at ~1e-5, per-distribution counts are too sparse — pool
  error counts across all valid Tier-C distributions; paired bootstrap +
  Wilcoxon signed-rank must agree. N ≥ 16384 seq/dist, ≥ 30 distributions, 3
  training seeds for the final claim.
- **#1 false-positive risk + guard:** the LSTM-validity filter (drop if LSTM
  > 0.5%) silently removes the *hardest* Periodic distributions — exactly
  where B_liu would fail — inflating non-inferiority. **Guard:** report
  dropped-distribution count and require the claim to survive a stricter LSTM
  threshold of 0.02 (sensitivity panel).

---

## 5. Experiment ladder (kill bad ideas cheaply)

| Rung | Runs | Wall-clock | Proceed iff |
|---|---|---|---|
| **1** | Existing checkpoints incl. the regret retrain now finishing; `eval_fair --tiers C`, 1 seed | ~15 min | regret-model mean-C ≤ A_adv's (objective fix moved the needle); else go straight to §2.A/B |
| **2** | Implement §1 add + §2.A + §2.C; one adversary→pool→full retrain→eval | ~2.5 h | non-inferiority (PRIMARY) holds AND held-out trends right |
| **2.5** | Add §2.B `AntiCorrelatedBits` + §2.D seed-1 baseline (one-time 2h); one adversary→pool→retrain→eval | ~4.5 h | B_liu shows residual error on held-out `AntiCorrelatedBits` (stretch is alive) |
| **3** | 2-round iterative loop (§3) + 3-seed final claim under §4 | ~9 h | PRIMARY across 3 seeds; STRETCH if §4 stretch criterion holds |

Stop at any rung that fails its gate. Rung 1 reuses the retrain already in
flight (PID 10417) — zero extra cost.

---

## 6. Explicitly DO NOT do (overengineering — confirmed after second pass)

- Learned soft-λ novelty/diversity term *in the fitness* — the hard
  descriptor grid (§2.C) dominates a brittle tunable λ.
- Full QD/MAP-Elites archive with archive-guided sampling, DPP selection —
  we already keep the useful 90% as grid pooling; the rest is new infra with
  no payoff at a ≤8-element pool.
- Online group-DRO *training-time* reweighting, multi-checkpoint *ensemble of
  many* checkpoints (the 2-checkpoint filter §2.D is the cheap useful kernel),
  3+ descriptor grids, per-position descriptor zoo.
- Fine `replay_frac` sweep — keep 0.5; probe 0.3 exactly once, only if
  in-dist error rises above baseline.
- Open-endedness / POET co-evolution — categorically out of scope on 1 GPU.

CVaR is **not** built — it is cited in the writeup as the theoretical
justification for choosing the descriptor grid over top-K. No code.

---

## 7. One-paragraph summary

Keep the regret + read-density objective (done); add a free write→read-gap
descriptor. Add two distribution axes: `MixtureTwoStationary` so CMA can
optimize Liu's *own* two-tail family (path to *matching* Liu automatically),
and `AntiCorrelatedBits` so the adversary can exploit the bit-distribution
slack Liu's i.i.d. tails leave (the only path to actually *beating* Liu).
Replace top-K pooling with a gap×density descriptor grid (CVaR-justified
coverage). Add a cheap two-checkpoint admission filter (one one-time seed-1
baseline) to kill non-transferable distributions, and wrap it in a 2-round
from-scratch model-in-the-loop retrain. Pre-register a Tier-C + two-held-out
benchmark with bootstrap CIs and an LSTM-filter sensitivity panel. Realistic
outcome: **automatically matching Liu's hand-picked R4 with no human prior**
(a methods win); the strict off-axis beat via `AntiCorrelatedBits` is the
headline if it lands and a clean negative result if it does not.
