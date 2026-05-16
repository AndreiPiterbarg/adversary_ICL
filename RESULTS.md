# Results: A Better Training Set than Liu R4, Found Automatically

## In one paragraph

We compared training recipes for a small Transformer on the "flip-flop" memory
task. Liu et al.'s recipe ("R4") retrains on two hand-picked hard input
distributions. We built an alternative training set the same way in every
respect **except** that the extra hard distributions were chosen by an
automated search instead of by hand. On a held-out family of inputs that Liu's
recipe structurally never covers, Liu's model gets **4.06 %** of answers wrong
while ours gets **0.0008 %** (~5000× fewer errors) — and on the inputs Liu's
recipe *is* designed for, the two are statistically tied. A like-for-like
control shows the automated choice beats a hand-picked one by ~440×, and we can
explain mechanistically why Liu's model fails. The result was pre-registered
(test and thresholds frozen in version control before the deciding run).

---

## Background: the task and the terms

**Flip-flop language model (FFLM).** A toy sequence task. The input is a string
of instructions:
- `write 0` / `write 1` — store a bit,
- `read` — the model must output the most-recently-written bit,
- `ignore` — do nothing.

The model is graded **only at `read` positions**: of all the reads, what
fraction did it answer wrong? That fraction is the **glitch rate** (lower is
better). Liu et al. (2023) showed Transformers make these "attention glitch"
errors on certain input distributions.

**The skyline (our ground-truth reference).** A tiny 1-layer LSTM solves this
task *perfectly* (0 % glitch) on **any** valid input distribution — it is
effectively the exact algorithm. So the LSTM is our "skyline": if the LSTM
scores 0 % on some distribution but a Transformer does not, that distribution
is genuinely solvable and the Transformer simply has a bug there. This rules
out blaming a bad/ill-posed input for a Transformer's failure.

**Liu R4.** Liu et al.'s proposed fix: retrain the Transformer on two
hand-picked extreme input distributions (a very sparse one and a very dense
one) mixed with the standard base distribution. It works well on the kinds of
inputs Liu tested.

**Stationary vs. non-stationary inputs.** A *stationary* distribution uses the
same instruction-generation rule for the whole sequence. Liu's two
distributions are stationary. A *non-stationary* ("segment-drift") distribution
has phases — e.g., a long opening phase that writes a bit then goes quiet,
followed by a later phase that is all reads. The model must carry that early
bit across a long gap. **Liu's stationary recipe cannot produce sequences
shaped like this**, so a Liu-trained model never practices them. This is the
axis where we look for an advantage.

---

## What we are comparing

All models are the same size (6-layer, 512-dim GPT-2), trained from scratch
with the same budget. They differ **only in their training set**:

| Name | Training set |
|---|---|
| **B_liu** | Liu R4: Liu's 2 hand-picked distributions + base. *The competitor.* |
| **stationary_control** | Liu's 2 + **8 more hand-picked** stationary distributions + base. |
| **beat_R1** | Liu's 2 + **8 automatically-chosen** distributions + base. |
| **full_adv** | An earlier, different automated pool + Liu's 2 + base. |
| **LSTM** | The skyline (perfect reference, not trained on anything special). |

`beat_R1` and `stationary_control` have the **identical structure** — Liu's 2
tails plus 8 extra distributions plus base — so comparing them isolates one
single variable: *were the 8 extra distributions picked by hand or by the
automated search?*

---

## How the automated search works (and what is still human-chosen)

A black-box optimizer (CMA-ES) searches a family of input distributions for
ones the current Transformer fails on the most. To avoid rewarding "unfair"
inputs, it scores each candidate by **regret = (Transformer error) − (LSTM
error)**: a distribution only counts as good if the Transformer fails *where
the LSTM succeeds*. A minimum-density floor stops it from cheating with
near-empty inputs.

**This is not "prior-free."** Humans designed the search family, the regret
objective and its thresholds, the base-distribution mixing, and the held-out
test itself. Crucially, **Liu's two hand-picked tails are included in every
training set, including `beat_R1`'s.** The automation replaces exactly one
thing: *which* distributions to add within that family. The fair comparison
works precisely because `B_liu`, `stationary_control`, and `beat_R1` all share
those same human choices — only the selection method changes.

The test is **pre-registered**: the held-out distributions, sample sizes, and
pass/fail thresholds were committed to version control (git `941c5dd`) *before*
the deciding control was trained, so nothing could be tuned after seeing
results.

---

## Result 1 — The beat (held-out non-stationary inputs)

Evaluated on 17 frozen non-stationary distributions (one base config + 16
small variations of it), 4096 sequences each. "CI" is a 95 % bootstrap
confidence interval — a standard resampling check that the gap is real and not
sampling luck; "excludes 0" means the advantage is statistically solid.

| Model | Glitch rate | Errors vs. Liu |
|---|--:|--:|
| **B_liu** (Liu R4) | **4.06 %** | — |
| stationary_control | 0.35 % | 12× fewer |
| **full_adv** | **0.043 %** | 94× fewer |
| **beat_R1** | **0.0008 %** | ~5000× fewer |
| LSTM (skyline) | 0.00 % | — |

The advantage holds on **all 17 of 17** variations (Liu 2–6 %, LSTM 0 % every
time), so it is a robust region, not one lucky case. The bootstrap CI excludes
0 by a wide margin.

**Tie where Liu is strong.** On Liu's *own* two distributions, and on a
separate unrelated "periodic" test neither side trained on, `B_liu` scores
0.000 % and our models score about 0.001 %. Our pre-registered tolerance for
"statistically tied" was 0.02 %; the gap is ~100× smaller than that, so this
counts as tied. (That `B_liu` is perfect on its own inputs *and* on the unseen
periodic test shows it is a strong model — its non-stationary failure is a
genuine blind spot of the recipe, not weakness.)

---

## Result 2 — Automated selection beats hand-picked (clean control)

`beat_R1` vs. `stationary_control`: same recipe, same number of extra
distributions (8), same Liu tails, same base — the **only** difference is how
the 8 were chosen.

| How the 8 extra distributions were chosen | Non-stationary glitch |
|---|--:|
| not added (Liu's 2 only) — `B_liu` | 4.06 % |
| **hand-picked** broad grid — `stationary_control` | **0.35 %** |
| **automated** regret search — `beat_R1` | **0.0008 %** |

Hand-picking a broad grid already helps a lot (12× better than Liu). But the
automated search, choosing the same number of distributions from the same
family, is a further **~440× better** and reaches the skyline. With everything
else held identical, *selection method is the thing that closes the gap.*

---

## Result 3 — Why Liu's model fails (mechanism)

We inspected where each model "looks" (its attention) when it makes a read
error in the hard recall phase, where the answer was written long before. For
the wrong reads we measure how much attention lands on the **start of the
sequence** ("early") vs. on the **actual most-recent write** ("target").

| Model | Wrong reads in recall phase | Where its attention went |
|---|--:|---|
| **B_liu** | **2491** | start of sequence (0.40) ≫ correct target (0.0002) |
| stationary_control | 199 | start (0.9997) ≫ target (0.000) |
| full_adv | 18 | start (1.000) ≫ target (0.0009) |
| **beat_R1** | **0** | (never errs here) |

Error counts fall in the same order as the glitch rates (2491 → 199 → 18 → 0).
When these models fail, their attention has **drifted to the beginning of the
sequence** instead of the relevant write — exactly the "attention glitch" Liu's
own paper describes. The extra training distributions teach the model not to
drift; `beat_R1` makes zero such errors.

---
## Reproduction

- Result 1: `results/flip_flop/eval_headtohead/summary.md`
- Result 2 (control): `results/flip_flop/eval_headtohead_rung1/summary.md`
- Result 3 (mechanism): `results/flip_flop/full_from_scratch_*/attention_position/heldout_piecewise/summary.md`
- Plan: `docs/PLAN_beat_liu_r4.md` · Pre-registration: `flip_flop/PRE_REGISTRATION.md`
