# Plan: A Feature-Conditioned Controller — Resolving the Expressiveness Bottleneck

**Status:** plan of record for the adversary-expressiveness redesign. Complementary
to `PLAN_beat_liu_r4.md` (the Liu-beat track, Rung 1 in progress) — this does not
modify that track.

---

## 0. The problem, stated exactly

The adversary's reach is the **union of hand-coded parametric families**
(`Stationary`, `Piecewise`, `Periodic`, `BitMarkov`, `AntiCorrelatedBits`,
`WriteFlipRate`, …). Each new break direction = a human authoring a 13th family.
Expressiveness is bounded by human enumeration, and — per the feasibility audit —
the families are *also* inconsistently guarded against degeneracy
(`objective.py:78-79`: gates default off). We want the optimizer to **discover
the breaking structure itself**, over a space that contains every existing
family *and all of their combinations*, while keeping validity, the LSTM-0
skyline, interpretability, and the existing pipeline.

---

## 1. The core idea

Every valid FFL distribution is a **per-position emission process**: at each
instruction slot emit `a_t ∈ {W,R,I}`; at data slots emit a free bit `b_t`;
read-slot bits are forced by `enforce_read_determinism` (`data.py:22-41`, called
unconditionally by `FFLDistribution._finalize`, `distribution.py:58-62`). So the
entire controllable object is two conditionals:

> π_inst(a_t | φ_t)  and  π_bit(b_t | φ_t)

where **φ_t is a fixed, bounded feature summary of the causal prefix**. Every
hand-coded family is just this with φ restricted to one feature:

| family | what φ it conditions on | as a controller weight pattern |
|---|---|---|
| `Stationary` | nothing (constant) | only biases nonzero |
| `Piecewise`/`Periodic` | position | only position-basis weights nonzero |
| `BitMarkov`/`AntiCorrelatedBits` | recent written bits | only bit-history weight nonzero (bit head) |
| churn / gap / run-length axes (audit A1/A4/A8) | one history counter | only that counter's weight nonzero |
| **any combination** (never hand-coded) | several at once | **several weights nonzero — reachable only here** |

We replace the family zoo with **one log-linear controller** and search its
weights. The human supplies an *algebra* (a feature set + a linear-softmax form);
the optimizer composes it.

### 1.1 Exact parameterization

- Causal feature vector `φ_t` (all O(1) running counters in the sampling loop;
  all normalized to [0,1] or {0,1} so the space is bounded):
  - **v1 features (6):** `p = t/T`, `p²`, `gap_norm = min(since_last_W, G)/G`,
    `nwrites_norm = min(#W_so_far, Wm)/Wm`,
    `runlen_norm = min(cur_read_run, Rm)/Rm`, `last_write_bit ∈ {0,1}`.
  - **v2 features (+4, only if v1 CMA converges cleanly):** `sin(2πp)`,
    `cos(2πp)` (lets it *learn* periodicity), `gap_norm·nwrites_norm` (a joint /
    interaction term — captures A9-style frontier breaks), `distractor_norm`
    (#writes inside the current gap).
- Instruction head: `π_inst = softmax(W_a φ_t + c_a)`, W as reference class
  (zero row) → `2·(d_φ+1)` params.
- Bit head: `π_bit(1) = σ(w_b·φ_t + c_b)` → `d_φ+1` params.
- **Total params = 3·d_φ + 3.** v1: d_φ=6 → **21 dims** (≈ the 16-dim
  `PiecewiseEncoder` regime that already works under diagonal CMA). v2: d_φ=10 →
  33 dims, only adopted if v1's `cma_trajectory.jsonl` shows healthy descent.

### 1.2 Structural non-degeneracy (the principled anti-edge-case fix)

Degeneracy is removed **by construction**, not left to the objective: the
instruction head is mixed with a floor prior,
`π = (1−ε)·softmax(·) + ε·π_floor`, where `π_floor` guarantees
`p_r ≥ p_r_min` and `p_w ≥ p_w_min` at *every* position. The `p_r→0`
read-starvation that turned `bit_markov` into a degenerate edge case becomes
**unreachable by the parameterization**. The existing objective gates
(`--min_reads_per_seq 3 --min_gap 4`, `objective.py:105-108`) and the LSTM
hinge (`objective.py:115-121`) stay on as defense-in-depth. One central guard
replaces the per-family guards that were silently inconsistent.

---

## 2. Why this is the best option (vs every alternative)

1. **It dissolves the problem instead of adding to it.** The 12 audit families
   are "more enumeration." This changes the *type* of the search space from a
   union of human-picked points/curves to a parametric function class. Unseen
   combinations are in-class by construction.
2. **Code-verified strict superset → loses nothing.** The feasibility pass
   verified from `distribution.py` that ≥6 families are special cases; §3 adds
   *subsumption unit tests* making this a guarantee, not a hope. Adopting it
   cannot regress reach, only extend it (to all interactions).
3. **Minimal blast radius.** The pool→dump→retrain→eval pipeline is
   config-driven and reconstructs any logged distribution via
   `FFLDistribution.from_dict` (`dump_dataset.py:38-62`,
   `pool_breaking_points.py:171-178`). One new family + one encoder + 2 wiring
   lines; **zero** changes to `objective.py`, pooling, dump, retrain,
   `eval_headtohead`, `data.py`, `model.py`.
4. **More robust than the zoo, not just more expressive.** Validity
   (`enforce_read_determinism`) *and* non-degeneracy (§1.2) are enforced in
   exactly one place — fixing the audit's finding that family guards are
   inconsistent and default-off.
5. **Keeps interpretability — the reason families were attractive.** Log-linear
   weights are a readable scientific statement ("read-prob rises with gap,
   modulated by late-sequence churn"). A neural controller (meta-option A)
   forfeits this; the DSL (option C) needs a whole new search engine (violates
   the project's "no new search machinery" rule); option A's differentiable
   route is incompatible with the frozen-argmax eval (`eval.py:60-80`). B is the
   only point on the frontier that is both maximally subsuming and minimally
   invasive.
6. **Upgrades the scientific claim.** From "an adversary searching *our*
   families found the axis" to "given only generic features and a linear model —
   no human distribution families — the optimizer *autonomously discovered* the
   breaking structure, and we can read out what it is." This removes the "you
   hand-designed the families" critique against the Liu result entirely.
7. **It is the substrate the high-payoff future work needs anyway.** A9
   (MAP-Elites) requires a generator that moves all descriptors independently —
   this *is* that generator. D (open-ended loop) = this + the existing loop.
   Building any single family is a dead-end; building this unlocks the rest.

---

## 3. Code changes (small, additive — no existing caller modified)

- **C1 — `FeatureControlledFFL(FFLDistribution)`** in
  `flip_flop/adversary/distribution.py` (~90 lines). Per-position sampling loop
  modeled on `AntiCorrelatedBits.sample` (`distribution.py:286-312`): maintain
  running counters → build φ_t → softmax/σ → sample `a_t`,`b_t`. Implement
  `to_dict`/`_from_dict` (weights as nested lists) and `descriptor`. One line in
  `REGISTRY` (`distribution.py:571-580`).
- **C2 — `FeatureControlledEncoder`** in `search.py` (~30 lines): `n_dims =
  3·d_φ+3`; `decode(z)` → reshape to (W_a, c_a, w_b, c_b), pass through; mild
  scale so CMA's unit-variance init maps to sane logits; `random_init` = small
  Gaussian. Two lines in `_make_encoder` (`run.py:113-126`).
- **C3 — config** `flip_flop/configs/adversary_feature_controlled.yaml`: copy an
  existing adversary config, set `family: feature_controlled`,
  `min_reads_per_seq: 3`, `min_gap: 4`, `p_r_min`/`p_w_min` floors, `d_phi: 6`.
- **C4 — tests** in `flip_flop/adversary/tests/`: see §4 Phase 0.

No architecture, objective, pipeline, or data changes.

---

## 4. Validation (gated phases — cheapest decisive first; pre-register before Phase 2)

### Phase 0 — Correctness (no GPU, ~minutes)
- **Validity:** every sampled sequence passes the FFL validity check (mirror
  `test_distribution.py:39-49`).
- **Round-trip:** `to_dict`→`from_dict` identity (pattern
  `test_distribution.py:153-168`).
- **Subsumption tests (the key guarantee):** instantiate the controller with the
  weight pattern that should equal `Stationary(0.8)`, a known `Piecewise`, and
  `BitMarkov`; assert sample statistics match the hand-coded family within
  tolerance. This *proves* the superset claim and is a permanent regression
  guard.

### Phase 1 — Rediscovery sanity (cheap GPU, ~1–2 h)
Small-budget CMA vs the existing `B_liu`. **PASS:** without being told about any
axis, the controller's weights concentrate on position+gap features and it
recovers a non-stationary breaking regime comparable to the hand-found piecewise
axis. This *is* the headline automation demonstration. **FALSIFY-EARLY:** CMA
fails to descend at 21 dims → reduce d_φ / raise budget / sep-CMA before Phase 2.

### Phase 2 — The real result (one full search + one retrain, ~½ GPU-day)
Full-budget CMA, regret objective, gates on → existing `pool_breaking_points`
descriptor-grid pool → `dump_dataset` → `run_retrain_from_dataset` (arch
identical to B_liu) → `eval_headtohead` on the **frozen pre-registered held-out
set** (reuse `PRE_REGISTRATION.md` design: N, seed, paired bootstrap, LSTM-drop
5e-4, δ=2e-4).

- **WIN:** controller-pool retrain ties Liu where Liu is strong and matches/beats
  the best hand-found single-axis pool on the frozen band (CI excludes 0).
  Claim upgrades to "autonomous discovery, no human families."
- **HONEST NEGATIVE (pre-registered):** *"If the fully-expressive controller,
  under the regret objective + gates, cannot match the best hand-found
  single-axis pool on the frozen held-out set at the pre-registered N, then
  expressiveness was not the bottleneck — report the negative."* This is itself
  a clean methodological result.

### Phase 3 (optional, only if Phase 2 wins) — QD + loop
Online MAP-Elites over the existing descriptors using this controller as the
generator (audit A9), then close the existing `run_beat_liu_loop` on the archive
(audit D). Deferred until Phase 2 justifies it.

---

## 5. Risks & mitigations (honest)

- **CMA dimensionality (diagonal, ≤16 dims demonstrated).** → Staged d_φ
  (v1=21≈Piecewise's 16; v2 only if healthy); monitor `cma_trajectory.jsonl`;
  fallbacks: more restarts, larger budget (cost linear: 1 fwd over `search_n`),
  sep-CMA.
- **Log-linear too weak for multiplicative structure** (e.g. "long gap AND high
  churn"). → softmax already gives logit-space conjunctions; v2 adds explicit
  product features — still linear in params, still interpretable.
- **"It's still a model, not interpretable."** → log-linear weights are directly
  readable *and* the existing descriptor-grid pool names what was found; both are
  deliverables.
- **Subsumption regressing existing results.** → it is purely additive (old
  families untouched); Phase-0 subsumption tests guard equivalence.
- **GPU contention with the in-progress Rung 1.** → Phase 0 is CPU-only; Phase 1
  starts only after Rung 1's single-GPU job is verified done (`tasklist`/
  `nvidia-smi`, per `PLAN_beat_liu_r4.md` §7).

---

## 6. Effort & deliverables

~150–220 LOC total (C1 ≈ 90, C2 ≈ 30, C3 config, C4 tests), no pipeline edits.

1. `FeatureControlledFFL` + registry line; `FeatureControlledEncoder` + 2 wiring
   lines; config.
2. Phase-0 test suite incl. the subsumption proofs.
3. Phase-1 rediscovery result (weights readout showing it found the axis
   unaided).
4. Phase-2 head-to-head on the frozen held-out set + the pre-registered
   win/negative call.
5. `RESULTS_expressive_adversary.md`: the story — *one log-linear controller,
   no hand-coded families, autonomously rediscovered (and extended) the breaking
   structure*, with the learned-weight interpretation and the honest Phase-2
   outcome.
