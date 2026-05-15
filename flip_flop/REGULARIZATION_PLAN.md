# Regularization plan for flip_flop FFLM

Scope: decide which regularization knobs to actually try in this repo, write
configs, and estimate cost. **No training runs in this task.** The project goal
is *adversarial discovery* (changing the failure landscape so the CMA-ES loop
finds different/more-interesting glitch families), not zero-glitch retraining.

## (a) Knob inventory — current vs Liu et al. swept

Code touchpoints established by file:line scan; "swept" column from
ff_paper.txt §5.2 + Appendix B.4 (Figs 10/11/16).

| Knob                 | Current value (file:line)                                    | Liu swept over                                                            | Liu best                                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| `weight_decay`       | 0.1 ([baseline.yaml:28](configs/baseline.yaml#L28); flat AdamW group, no LN/bias exclusion at [train.py:167-172](train.py#L167-L172)) | {0, 0.01, 0.03, 0.1, 0.3}                                                 | No clear winner; baseline 0.1 already chosen.           |
| `attn_pdrop`         | 0.0 ([baseline.yaml:13](configs/baseline.yaml#L13))          | {0.05, 0.1, 0.15, 0.2, 0.3, 0.4}                                          | None highlighted; ≥0.3 hurts.                           |
| `resid_pdrop` (MLP)  | 0.0 ([baseline.yaml:11](configs/baseline.yaml#L11))          | {0.05 … 0.4}                                                              | None highlighted.                                       |
| `embd_pdrop`         | 0.0 ([baseline.yaml:12](configs/baseline.yaml#L12))          | {0.05 … 0.75}                                                             | **0.5** — ~2 orders of magnitude reduction on FFL(0.98) sparse tail (R6). |
| Position embedding   | learned absolute `wpe`, hard-coded by GPT2Config in [model.py:26-36](model.py#L26-L36) | learned / sinusoidal / RoPE / T5-rel / ALiBi / zero                       | Non-trainable PEs (sinusoidal etc.) win on FFL(0.1) dense tail (R6). |
| Attention sharpening | none — loss is plain `clean_loss` at [train.py:272-275](train.py#L272-L275) | entropy / −ℓ₂ / −ℓ∞ on attention; coef ∈ {0.01,0.03,…,30}; linear/oscillating anneal starting step ∈ {0, 2000, 5000} | Annealed sharpening + sinusoidal PE + emb-drop is the Pareto frontier (Fig 13). |
| `grad_clip`          | 1.0 ([baseline.yaml:35](configs/baseline.yaml#L35))          | not swept                                                                 | —                                                       |
| Optimizer (β, lr)    | (0.9, 0.999), 3e-4 ([baseline.yaml:27-29](configs/baseline.yaml#L27-L29)) | β₁∈{0.85,0.9,0.95}, β₂∈{0.95,0.99,0.999}, lr∈{1e-4 … 3e-3}                | None — "no significant improvements".                   |
| Softmax τ            | 1.0 (default)                                                | {0.1, 0.3, 1, 3, 10}                                                      | None — "glitches mostly worsen".                        |

Liu's R7 (paper-wide caveat): "nothing eliminates the glitches entirely."

## (b) Per-knob payoff in *our* (adversarial-discovery) setting

- `weight_decay` ↑: paper's evidence is null; intuition that "stronger wd → simpler solution → cleaner attention" is plausible but unsupported. Unlikely to change which families CMA-ES finds.
- `attn_pdrop`, `resid_pdrop`: paper's effect is null/small. No story for why this would shift the adversary's discovery.
- `embd_pdrop = 0.5`: the one standard regularizer Liu found large (R6). Closes the *known symmetric* sparse tail. **Adversary-relevant question**: does it also close `piecewise_c00` (dense→sparse transition) and `bit_markov_c00` (write-heavy boundary)? If it closes the symmetric tail but not those, that strengthens the round-1 finding ("regularization fixes hand-designed tails but not adversarially discovered ones"). If it closes both, our findings reduce to a regularization gap. Either outcome is informative.
- Non-trainable position embeddings: blocked — selecting `position_embedding_type ≠ absolute` requires editing [model.py:26-36](model.py#L26-L36); task forbids modifying `model.py`. Skip.
- **Attention sharpening (annealed entropy / −ℓ₂)**: the most adversary-interesting knob. Paper R8 + Prop. 4 + Fig 16d explicitly states sharpening induces a *new* failure mode where the model confidently locks onto an *early* (wrong) write rather than the most-recent one. An adversary searching FFL distributions that place the most-recent correct write far from the read position should preferentially exploit this. **However**, requires a new loss term in `train.py`; task forbids that change. Stub only.
- `grad_clip`, optimizer hparams, softmax τ: Liu found them null; no project-specific story to override that.

## (c) Ranked recommendations (default: NO)

| # | Experiment                          | Run? | Justification                                                                                                                                                                                                              |
| - | ----------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `baseline_embd_drop05`              | **YES** | Only Liu-confirmed large-effect knob (R6, ~2-orders-of-magnitude on sparse tail). Lets us re-run the adversary loop on a model that has *already* closed the known symmetric tail and ask: do `piecewise_c00` / `bit_markov_c00` survive? Direct test of whether our findings are "another standard regularizer's blind spot" or genuinely new. |
| 2 | `baseline_attn_sharp_anneal` (stub) | **YES (stub)** — requires `train.py` change | Sharpening is the only intervention the paper itself identifies as creating a *new, structured* failure mode (Fig 16d + Prop. 4 — confidently attending to wrong early writes). That is precisely what an adversary should find. Worth queueing as a future experiment once `train.py` is editable; ship the config now so the stack is ready. |
| 3 | `weight_decay = 1.0` (10× default)  | NO   | Paper evidence is null across {0…0.3}; one extra step (1.0) doesn't have a strong prior. Cost not justified vs the two above. User's IDE hypothesis ("simpler solution") is reasonable but Liu already disproved it.       |
| — | `attn_pdrop = 0.1`, `resid_pdrop = 0.1`, combined-dropout stack | NO | Paper-null individually; combining without sharpening + non-trainable PE only gets a fragment of Liu's Pareto-best stack and doesn't have its own story. |
| — | softmax τ, optimizer β/lr, grad_clip | NO  | All explicitly null per Liu §5.2.                                                                                                                                                                                          |

## (d) Configs delivered

- [configs/baseline_embd_drop05.yaml](configs/baseline_embd_drop05.yaml) — runnable. Only diff from `baseline.yaml`: `embd_pdrop: 0.5`, new `out_dir`.
- [configs/baseline_attn_sharp_anneal.yaml](configs/baseline_attn_sharp_anneal.yaml) — **STUB. Requires `train.py` change.** Header comment lists the missing pieces (auxiliary loss term, anneal schedule, coefficient scalar in optimizer step). Do not pass to `run_baseline.py` until those are added.

## (e) Cost estimate

From [results/flip_flop/baseline/train_log.jsonl](../results/flip_flop/baseline/train_log.jsonl): step 9950 at `elapsed=3705.86s`, so a full 10000-step run ≈ **3725 s ≈ 62 min ≈ 1.04 GPU-hours** at the project standard (batch=16, GPT-2 6L/512d/8H, ~19M params, single-GPU). Eval phase adds ~30 s, negligible.

| Experiment                          | Train steps | GPU-hours (train) | Notes                                                                 |
| ----------------------------------- | ----------- | ----------------- | --------------------------------------------------------------------- |
| `baseline_embd_drop05`              | 10000       | ~1.04             | Then re-run adversary loop on the resulting checkpoint (separate cost). |
| `baseline_attn_sharp_anneal` (stub) | 10000       | ~1.04 *projected* | Will not run until `train.py` supports an auxiliary loss; cost is the projected baseline since sharpening is a per-step add (\< 5% overhead). |

Total committed-to-run cost this round: **~1 GPU-hour** (just experiment 1). Downstream adversary search + retrain + decisive eval on the new checkpoint reuses the existing pipeline and is not part of this estimate.

## Notes / caveats

- The "best stack" in Liu Fig 13 (sinusoidal PE + emb-drop 0.5 + annealed sharpening) cannot be reproduced under the current task constraints because two of its three components require code edits (`model.py` for non-trainable PE, `train.py` for sharpening loss). Experiment 1 is the largest-effect *config-only* component of that stack.
- If the round-1 piecewise_c00 finding survives experiment 1 (i.e. CMA-ES still finds it on the embd_drop=0.5 model), that is a direct strengthening of the round-1 paper claim. If it doesn't, the round-1 finding gets reframed as "a known regularizer's blind spot."
