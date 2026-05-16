# Audit: New Break Directions to Strengthen the Position vs Liu R4

**Date:** 2026-05-16
**Method:** 12 parallel research agents, each scouting one *unexplored* axis under
four hard constraints applied uniformly: (1) **general mechanism, not an edge
case** (no `p_r→0` / `p_i→1` / near-zero-read flukes); (2) **stays a valid FFL
distribution with the 1-layer-LSTM skyline at ≈0%** (else the "break" is
ill-defined, not a break); (3) **plausibly strengthens the Transformer's true
state-tracking when retrained on**; (4) **new** vs the already-explored axes
(stationary tails, bit_markov, write_flip, piecewise/segment-drift, planted_decoy,
anti-correlated-bits, periodic, mixture-of-two-stationary, regret objective).
Out of scope by prior decision: T>512 length-extrapolation, scratchpad/index
tokens, architecture changes.

This is an **ideas audit** — no code was written and none is recommended here;
each direction is specified to the point where it can be slotted into the
existing `flip_flop/adversary/` + `eval_headtohead` infra.

---

## 0. The unifying finding (why almost every new axis beats Liu structurally)

Liu R4's stationary i.i.d. sampler has **one knob** — the constant
`(p_w, p_r)` rate — and that single knob *simultaneously* fixes **five distinct
latent variables**:

| latent variable | what it controls | exploited by |
|---|---|---|
| write→read **gap shape** | how far back recall reaches | A1 |
| **distractor count** within a gap | how many superseded writes interfere | A2 |
| total **write count** (state churn) | how many updates to integrate | A4 |
| **read-run length** | retrieve-vs-maintain duration | A8 |
| **position schedule** of all the above | non-stationarity | A12 (current win) |

Under stationarity these are **coupled** — you cannot move one without dragging
the others. Almost every promising direction below is the same move: **decouple
one latent variable and hold the rest fixed.** That coupling is the deep,
defensible reason Liu's recipe cannot reach these regions — a stronger and more
general framing than "Liu only used two hand-picked points." It also tells us
the endgame: a **joint search over exactly these decoupled descriptors**
(direction A9) converts "we beat Liu on one axis" into "we map and cover a whole
structured difficulty manifold Liu's family structurally cannot enter."

The 1-layer LSTM is the exact register automaton and is **invariant to gap,
distractor count, churn, and read-run length**, so it stays at 0% on all
single-register axes — the one place this could fail (extreme gap × ultra-low
write density approaching finite-cell-precision decay) is exactly the
**LSTM-validity boundary** that direction A11 is chartered to map. Mapping that
boundary is a prerequisite for legitimacy of every other axis.

---

## 1. Prioritized summary

Verdict column is the agent's call, cross-checked. "Understanding" column marks
*why it strengthens the model*: **C** = forces genuine causal/addressable recall
by construction; **I** = isolates a distinct capability (so we know what we
fixed); **F** = framing/rigor that hardens the claim.

| # | Direction | Type | New? | Edge-case risk | LSTM=0? | Understanding | Verdict |
|---|---|---|---|---|---|---|---|
| A12 | **DriftShapeZoo** generalization probe of the current win | eval-only | yes | none | yes | F | **Strong adopt — do first** |
| A11 | Info-theoretic hardness criterion + **LSTM-validity boundary map** | theory/rigor | yes | n/a | (maps it) | F | **Pursue — rigor prerequisite** |
| A3 | **LureBits** — unsupervised bits anti-correlated with the answer | new family | yes | very low | yes | C | **Pursue** |
| A8 | **Read-run-length** (retrieve-vs-maintain) at fixed read density | new family | yes | low | yes | I | **Strong pursue** |
| A2 | **InterferenceBurst** — clustered overwritten distractor writes | new family | yes | low (rate-boxed) | yes | I | **Pursue** (cheap pre-check) |
| A1 | **Gap-distribution shape** at fixed read density | new family | yes | medium (aliasing) | yes | I | **Pursue (with guard)** |
| A4 | **ChurnRate** — #writes/state-updates at fixed T=512 | new family | yes | low | yes | I | **Pursue** |
| A10 | **CounterfactualPairs** — balanced minimal-pair hard negatives | family + eval | yes | very low | yes | C | **Pursue** |
| A6 | **PredictableBits** — bit motif + pre-read violation, rates boxed | new family | yes | medium (guarded) | yes | C | **Pursue (with guard)** |
| A9 | **Joint descriptor-space QD / MAP-Elites** over the 5 latents | search framing | yes | low | by construction | F | **Pursue — biggest payoff** |
| A7 | **Mechanistic attention-targeted** objective (Fig-16d drift) | search framing | yes | low | gated | C/F | **Pursue** |
| A5 | **Multi-register** addressable memory | task extension | yes | low | **changes skyline** | C | **Pursue as companion claim only** |

No direction came back "likely-dead." That is expected — the constraints were
strict — but it also means the differentiator is **cost and decisiveness**, not
viability. The tiering below is ordered by exactly that.

---

## 2. Recommended sequencing (falsification-gated, cheapest-decisive-first)

Consistent with the plan's "cheap decisive gate first" discipline and the
pre-registration rule (freeze held-out configs *before* training).

### Tier 0 — Defend and harden what we already have (no training, ~1 GPU-day total)

- **A12 DriftShapeZoo (do this first).** Config-only, eval-only. Builds a
  *pre-registered, frozen* zoo of out-of-band non-stationary shapes —
  ramp / sinusoid / multi-plateau / reversed / relocated-plant — and re-runs the
  existing head-to-head harness. This **directly neutralizes the strongest
  reviewer attack on the current result** ("the piecewise win is band overfit").
  Either it confirms the win is a *capability* (our models ≪ B_liu across all
  shapes, LSTM=0) or it draws a precise map of where the capability ends. Highest
  leverage / lowest cost move available; it strengthens the *present* position
  before any new break is chased.
- **A11 LSTM-validity boundary.** Pure measurement (sampling only, no training).
  Sweep gap × write-density and flag any region where `LSTM_err > 5e-4`. Every
  other axis silently assumes the skyline holds; this is the rigor floor that
  makes "the LSTM proves the distribution is legitimate" defensible. The
  information-theoretic `Hardness = R_T − I_min` criterion is the longer-horizon
  payoff but the boundary map alone justifies the rung.

### Tier 1 — New general break families, cheap, mechanistically clean (each ≈1 GPU-day to a Rung-0-style verdict)

Run a **Rung-0-style scale eval first** (B_liu vs full_adv vs LSTM on a frozen
held-out band, N≥4096, paired bootstrap) before any retrain — the same gate
that already de-risked the piecewise win in <1 GPU-hour.

- **A3 LureBits** — cleanest mechanistic story: make the *unsupervised*
  (post-`w` / post-`i`) data bits — which Liu leaves uniform random by
  construction at *every* p — systematically point at the wrong answer.
  Provably LSTM-0 (the automaton ignores free bits). A model that spikes here
  was provably using a copy-nearest/positional shortcut; retraining removes the
  only shortcut and forces the recall circuit. **C-type** understanding gain.
- **A8 Read-run-length** — isolates a *distinct capability the project has been
  conflating*: **maintenance** (re-emit a held value across a long no-input read
  burst) vs **retrieval** (recall a recent write). Single interpretable axis,
  Liu's i.i.d. sampler cannot decouple read-burst length from mean read density.
- **A2 InterferenceBurst** — directly targets the paper's *own stated glitch
  cause* (proactive interference / attention dilution): a clustered run of
  overwritten decoy writes ending just before a read. Cheap pre-check matters
  here: B_liu may *partially* cover it (would downgrade "beat" to "weaker beat"),
  so the Rung-0 eval is the decisive 1-hour test before committing a retrain.
- **A1 Gap-distribution shape** — make the per-read gap distribution
  (heavy-tailed / bimodal) the directly-controlled variable, **at fixed read
  density**. Guard required: if read density is not pinned, a heavy tail aliases
  onto Liu's known sparse stationary tail and the axis is no longer clean — the
  pre-registration must fix read density across the family.

### Tier 2 — Capacity and shortcut axes (each ≈1 GPU-day)

- **A4 ChurnRate** — vary number of state updates at fixed T=512 (no
  length-extrapolation artifact). Isolates state-*integration* capacity from
  gap/position. Clean, cheap, fully LSTM-invariant.
- **A10 CounterfactualPairs** — balanced minimal pairs differing only in one
  far-back governing write. Doubles as a **shortcut-proof eval metric** (pair
  accuracy: a constant/positional predictor scores zero by construction) and a
  **retrain signal whose gradient rewards only genuine causal recall**.
  Strongest **C-type** understanding argument; near-zero edge-case risk.
- **A6 PredictableBits** — the *non-degenerate* recovery of bit_markov's real
  insight: a learnable bit motif with targeted pre-read pattern violations,
  **with `p_r`/`p_i` hard-boxed so it cannot collapse to read-starvation** (the
  exact failure that made bit_markov a degenerate edge case). Guard is the whole
  point; the agent specified the box constraint precisely.

### Tier 3 — Search-framing upgrades (bigger engineering, largest scientific payoff)

- **A9 Joint descriptor-space QD / MAP-Elites.** The natural unification of
  everything above: search the 5-D interpretable descriptor manifold (gap mean,
  gap variance, distractor density, churn, bit predictability) with MAP-Elites,
  retrain on the *whole illuminated frontier*, and **plot B_liu's 4.06% failure
  band against descriptor coordinates** to show it lives in a region Liu's
  sampler provably cannot reach. This upgrades the headline from "beat Liu on one
  axis" to "cover a structured manifold Liu structurally misses," and exposes
  interaction-effect breaks single-axis CMA cannot see. High reuse of existing
  CMA/descriptor/pool infra; the only real cost is a small MAP-Elites archive.
- **A7 Mechanistic attention-targeted objective.** Add a `circuit` objective
  mode that maximizes the Prop-4 / Fig-16d drift signature (early-position
  attention mass minus mass on the ideal target) instead of black-box glitch
  rate. Yields *general-by-construction* breaks (it stresses the structural
  failure circuit, not a checkpoint-specific argmax fluke) and a stronger
  mechanistic claim: "we closed the drift channel," measured by the existing
  `diagnose_attention_position` / `probe.py` / `ablation.py` diagnostics.

### Tier 4 — Scoped task extension (companion claim, explicitly not apples-to-apples)

- **A5 Multi-register addressable memory.** The strongest *genuine
  understanding* candidate — the paper itself names parallel-register / Dyck as
  the open harder capability (Krohn–Rhodes cascade). **Honest cost, stated
  plainly:** the in-bounds restricted form (concatenated independent FFL blocks)
  is worthless — it collapses to "k easy problems" with no addressing. The
  valuable form genuinely *leaves strict single-register FFL*: vocabulary gains
  slot tags and the fair skyline is no longer the 1-layer LSTM but the **exact
  k-register automaton plus a capacity-matched recurrent baseline**. Run it as a
  **companion claim alongside** the clean single-register head-to-head — not as a
  replacement, and reported as "beyond Liu's family / strictly harder," never as
  an R4 win.

---

## 3. Cross-cutting risks and honest caveats

- **"No dead directions" is a selection effect, not a guarantee.** The
  constraints filtered hard; the real risk per family is that **B_liu already
  partially covers it** (turning "beat" into "marginal beat"). Mitigation is
  structural: every Tier-1/2 family must clear a Rung-0-style scale eval
  (N≥4096, paired bootstrap, LSTM-drop > 5e-4) *before* any retrain — the
  cheapest decisive gate, already proven on the piecewise win.
- **Three families have a named collapse mode** that must be frozen in
  pre-registration *before* searching, or they degrade into the edge cases the
  user explicitly rejected: A1 (pin read density, else aliases onto Liu's sparse
  tail), A2 and A6 (hard-box `p_r`/`p_i` so the search cannot flee to
  read-starvation — the exact pathology that killed bit_markov).
- **A5 changes the comparison.** It is the highest-ceiling understanding result
  but is *not* a fair-skyline beat of Liu R4; mis-framing it as one would be the
  single biggest credibility risk in this audit.
- **Orthogonality is real and is the asset.** A1 (gap distance), A2 (distractors
  within the gap), A4 (total write count), A8 (read-run length), A12 (position
  schedule) are genuinely independent latent variables. That independence is
  what makes A9's joint search the natural endgame rather than yet another axis.
- **All single-register families keep the LSTM at 0 by construction** (the
  automaton is invariant to count/clustering/gap/bits) — *contingent on A11's
  boundary map confirming no finite-precision LSTM decay at the extremes.** A11
  is therefore not optional flavor; it gates the legitimacy of the rest.

---

## 4. Concrete next three moves (no code, decision-ready)

1. **Freeze and run A12 (DriftShapeZoo).** Pre-register the shape zoo, run the
   existing head-to-head harness on already-trained models. Eval-only,
   ~hours-of-GPU, no training. Outcome either *hardens the current paper claim*
   (capability, not overfit) or *maps the band boundary* — both publishable.
2. **Run A11's LSTM-validity sweep** in parallel (sampling only, no GPU
   contention with A12). Produces the skyline-legitimacy boundary that every
   subsequent axis cites.
3. **Pick one Tier-1 family for the first new break — recommend A3 (LureBits)**:
   cleanest mechanistic narrative (shortcut → genuine recall), provably LSTM-0,
   lowest edge-case risk, ~1 GPU-day to a Rung-0 verdict. A8 (maintenance vs
   retrieval) is the strong second if a *new capability axis* is preferred over
   a *shortcut-defeating* one.

Full per-direction briefs (mechanism, why-Liu-can't-cover, falsification,
feasibility, novelty) are preserved verbatim in the agent transcripts; this
audit is the synthesis and prioritization layer over them.
