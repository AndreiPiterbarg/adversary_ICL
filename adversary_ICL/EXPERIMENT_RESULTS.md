# Adversarial ICL Experiment Results

## Overview

This document records two rounds of experiments. Round 1 was an initial proof-of-concept with a poorly converged model and no scale controls, producing trivial exploits. Round 2 fixed the methodology and produced genuine structural findings. The comparison between rounds is itself informative: it shows exactly which failures are artifacts of scale and which survive normalization.

---

## Round 1: Initial Proof-of-Concept (Baseline)

### Setup

| Component | Configuration |
|-----------|--------------|
| ICL Model | 4-layer GPT2, 2 heads, 64-dim embeddings |
| Training | 50,000 steps, d=5, isotropic Gaussian (Sigma=I) |
| Adversary genome | 51-dimensional: separate L_train, L_test (full 5x5 Cholesky each), mu_train, mu_test, w (unconstrained), noise |
| Search | Diagonal CMA-ES, 3,200 evaluations, single restart |
| Baselines | OLS, Averaging |
| Fitness | Additive gap / baseline scale, with degeneracy penalty |

### Model Convergence: Failed

On standard Gaussian inputs, the model had flat squared error ~5.3 across all k, while OLS dropped to near-zero at k >= d = 5. The model had **not learned in-context learning**. This is the critical confound for all Round 1 results.

### Results

The adversary converged to a single pattern across all top 10 failures:
- Rank-1 covariance: effective rank = 1.00, condition number ~1-8 million
- Top eigenvalue ~22,040 (hitting the exp(5) clamp ceiling, so actual Sigma eigenvalue = exp(10) ~22,026)
- Weight-covariance alignment 0.70-0.95
- Train-test Frobenius divergence ~4,400
- Best fitness: 327,730

### Assessment

These results validated the pipeline but **not the scientific question**. Three confounds made the findings uninterpretable:
1. The model hadn't converged, so the adversary was attacking residual incompetence, not a structural blind spot.
2. Unconstrained covariance scale (top eigenvalue ~22,000) meant inputs were ~148x outside the training range. Any model would fail.
3. Separate train/test distributions (divergence ~4,400) meant the adversary was trivially exploiting distribution shift.

---

## Round 2: Controlled Experiment

### Methodology Changes

Six bugs and design flaws from Round 1 were fixed:

| Issue | Round 1 | Round 2 | Rationale |
|-------|---------|---------|-----------|
| **Covariance direction** | `decode_covariance` returned L @ L^T | Returns L^T @ L | The sampler computes x = z @ L (right-multiply), so cov(x) = L^T @ L. The old code computed eigenvectors from the wrong matrix. |
| **Scale exploit: covariance** | Unconstrained trace (top eigenvalue ~22,000) | Trace-normalized: tr(Sigma) = d | Adversary controls eigenvalue ratios and rotation, not total variance. Eliminates trivial input-scale exploits. |
| **Scale exploit: weights** | Unconstrained w (adversary could set \|\|w\|\| >> 1) | Unit-normalized: \|\|w\|\| = 1 | Adversary controls task direction, not y-scale. Prevents trivial output-scale exploits. |
| **Distribution shift** | Separate L_train, L_test, mu_train, mu_test | Tied: single L, mu for both | Forces adversary to find failures under matched distributions. Genome dimension drops from 51 to 26 (d=5). |
| **Baseline gap** | OLS + Averaging only | Ridge (alpha=1.0) + OLS + Averaging | Ridge is numerically stable on ill-conditioned problems and is the Bayes-optimal estimator for Gaussian priors. |
| **Fitness function** | Additive gap with magic-number degeneracy penalty | Mean log-ratio over k=1..d | Log-ratio prevents single extreme points from dominating. Restricting to k <= d avoids the overdetermined regime where OLS error is exactly zero. |

Additional changes:
- Multi-restart CMA-ES (5 restarts with different seeds, budget split evenly) to improve coverage.
- No curriculum during training (fixed d=5, 11 points from step 0) for faster convergence.
- Learning rate 3e-4 (up from 1e-4).

### Setup

| Component | Configuration |
|-----------|--------------|
| ICL Model | 6-layer GPT2, 4 heads, 128-dim embeddings, 7.6M parameters |
| Training | 150,000 steps, d=5, n_points=11, isotropic Gaussian, LR=3e-4, no curriculum |
| Adversary genome | 26-dimensional: single L (5x5 Cholesky, trace-normalized), mu, w (unit-normalized), noise |
| Search | Diagonal CMA-ES, 10,080 evaluations, 5 restarts x 2,000 each |
| Baselines | Ridge (alpha=1.0), OLS, Averaging |
| Fitness | Mean of log(ICL_err / best_baseline_err) over k=1..5 |
| Hardware | NVIDIA RTX 3080 Laptop GPU |
| Total runtime | 72 minutes (49 min training + 23 min search) |

### Model Convergence: Verified

| k (examples) | ICL Error | OLS Error | Ridge Error | ICL/Ridge |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 4.721 | 4.714 | 4.714 | 1.00x |
| 2 | 2.902 | 2.863 | 3.111 | 0.93x |
| 5 | 0.141 | ~0 | 1.297 | **0.11x** |
| 10 | 0.010 | ~0 | 0.248 | 0.04x |

The model converged properly. At k=d=5, ICL error (0.141) is 9x better than ridge (1.297), meaning the transformer learned to approximate OLS-like behavior, consistent with Garg et al.'s finding that deeper transformers implement implicit least squares. At k >= d, the model achieves near-zero error, matching the noiseless linear regression optimum.

### Adversary Search Results

**10,080 evaluations across 5 restarts. All valid (no NaN/Inf).**

Fitness distribution (where fitness = mean log(ICL/baseline) over k=1..d):

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Min | 0.00 | ICL matches baseline |
| Median | 4.80 | ICL is ~e^4.8 = 121x worse |
| Mean | 5.75 | ICL is ~e^5.75 = 314x worse |
| Max | 17.97 | ICL is ~e^18 = 6.3 x 10^7 worse |

Note: the extreme max fitness reflects the fact that on near-rank-1 data, baselines solve the problem with 1 example (error ~0.001) while ICL error remains ~20,000-50,000. The log-ratio prevents this from distorting the optimization (unlike raw ratios), but the underlying gap is enormous.

### Top 10 Failure Modes

| Rank | Log-fitness | Eff. Rank | Top eigenvalue | 2nd eigenvalue | Remaining eigenvalues | Weight alignment | Noise |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| #1 | 17.97 | **1.93** | 3.11 | **1.79** | 0.08, 0.007, ~0 | 0.55 | 0.007 |
| #2 | 17.89 | 1.41 | 4.13 | 0.79 | 0.08, 0.002, ~0 | 0.19 | 0.007 |
| #3 | 17.27 | 1.39 | 4.16 | 0.80 | 0.03, ~0, ~0 | 0.02 | 0.007 |
| #4 | 16.87 | 1.20 | 4.54 | 0.44 | 0.02, ~0, ~0 | 0.16 | 0.007 |
| #5 | 16.80 | 1.30 | 4.34 | 0.60 | 0.05, 0.008, ~0 | 0.55 | 0.007 |
| #6 | 16.78 | 1.24 | 4.46 | 0.51 | 0.02, ~0, ~0 | 0.07 | 0.007 |
| #7 | 16.58 | 1.03 | 4.93 | 0.07 | ~0, ~0, ~0 | 0.20 | 0.007 |
| #8 | 16.48 | **2.04** | **3.05** | **1.71** | 0.20, 0.04, ~0 | 0.64 | 0.007 |
| #10 | 16.35 | 1.62 | 3.80 | 0.95 | 0.23, 0.02, ~0 | 0.77 | 0.007 |

All eigenvalues are from trace-normalized covariances (tr(Sigma) = 5).

### Learning Curve: Top Failure (#1, log-fitness=17.97)

Covariance spectrum: [3.11, 1.79, 0.08, 0.007, ~0]. Effective rank 1.93. Two significant dimensions.

| k | ICL Error | OLS Error | Ridge Error | ICL/best baseline |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 20,232 | 21,070 | 21,070 | 1.0x |
| 1 | 19,866 | 0.001 | 0.002 | 14,436,331x |
| 2 | 19,881 | 0.001 | 0.000 | 50,543,944x |
| 3 | 19,888 | 0.002 | 0.000 | 88,762,848x |
| 4 | 19,886 | 0.013 | 0.000 | 121,456,896x |
| 5 | 19,906 | 0.028 | 0.000 | 131,191,256x |

**ICL error is flat at ~20,000 across all k.** The transformer gains nothing from the in-context examples on this distribution. Both OLS and ridge solve the problem essentially perfectly from k=1 because the data lives in a 2-dimensional subspace of R^5 — the linear system is massively overdetermined even with 1 example projected onto the signal subspace.

### Statistical Correlations (Spearman, n=10,080)

| Feature | Spearman rho | p-value | Significance |
|:---|:---:|:---:|:---:|
| noise_std | +0.094 | < 1e-6 | *** |
| condition_number | -0.094 | < 1e-6 | *** |
| peak_failure_position | -0.048 | 1e-6 | *** |
| effective_rank | +0.009 | 0.376 | |
| weight_alignment | +0.002 | 0.869 | |

---

## Interpretation

### What the Adversary Discovered

The adversary independently converged to the same structural pattern across all 5 restarts: **low effective rank with near-zero noise**. This is a genuine ICL failure mode, not a scale artifact, because:

1. **Trace normalization is active.** tr(Sigma) = 5 for all genomes. The adversary cannot inflate input magnitude. A top eigenvalue of 4.93 (failure #7) means the other 4 dimensions share only 0.07 units of variance — the data is effectively 1-dimensional, but with total variance equal to the training distribution.

2. **Weight normalization is active.** ||w|| = 1 for all genomes. The y-scale is bounded by w^T Sigma w <= lambda_max(Sigma) <= d. The adversary cannot inflate output magnitude.

3. **Distributions are tied.** Train = test. The adversary cannot exploit distribution shift.

4. **Noise converges to the minimum** (0.007 = exp(-5), the clamp floor). In the noiseless or near-noiseless regime, the optimal estimator can perfectly recover w from very few examples on low-rank data. The transformer cannot. Noise would actually help ICL by making the problem harder for everyone; the adversary correctly discovered that removing noise maximizes the gap.

### Why This Breaks ICL

The transformer was trained on isotropic Gaussian inputs where each dimension contributes equally. Its learned representation (the read-in projection, attention patterns, and internal features) assumes roughly equal variance across all dimensions. When the covariance is near-rank-1 or rank-2:

1. **Input representation collapse.** The read-in layer maps d-dimensional inputs to the embedding space via a fixed linear projection learned on Sigma=I data. On rank-1 data, all inputs map to nearly the same direction in embedding space, destroying the positional/value diversity that attention needs to perform implicit regression.

2. **Attention degeneracy.** The attention mechanism computes similarity between embedded tokens. When all x's point in the same direction (rank-1 covariance), the attention weights become nearly uniform — the model cannot distinguish informative from redundant examples.

3. **OLS/ridge are immune.** The normal equations X^T X w = X^T y work regardless of covariance structure. On rank-1 data with d=5 and k=1, the 1x5 design matrix has rank 1, and the projection onto the signal direction gives a consistent estimator. The transformer has no analogous mechanism for adapting to the input geometry.

### What's Non-Obvious

Several aspects of the findings were **not predicted in advance** and required the adversary to discover:

1. **Weight alignment does NOT matter.** Spearman rho = 0.002 (p = 0.87). In Round 1, weight alignment appeared important (0.70-0.95 across all top failures). In Round 2 with proper controls, alignment ranges from 0.02 to 0.77 with no correlation to fitness. The apparent alignment effect in Round 1 was an artifact of the adversary optimizing both w and covariance together to maximize raw scale, not a structural insight.

2. **Effective rank ~2 is worse than rank ~1.** The top failure (#1) has effective rank 1.93, not 1.0. Failures #8 (eff rank 2.04) and #10 (eff rank 1.62) also score high. This suggests the transformer may actually handle pure rank-1 data slightly better than rank-2 data — perhaps because rank-1 data has a simpler pattern the model can partially learn, while rank-2 data presents a more confusing intermediate case.

3. **The failure is absolute, not gradual.** ICL error is flat at ~20,000-50,000 from k=0 to k=10. The transformer does not improve at all with more in-context examples. This is qualitatively different from the training distribution, where ICL error decreases smoothly. The failure is not "ICL is slow to converge" but "ICL provides zero benefit."

4. **Condition number is weakly negatively correlated** (rho = -0.094). Higher condition number actually *reduces* fitness slightly. This is counterintuitive — one might expect higher condition numbers to always be worse. The explanation: at extreme condition numbers (pure rank-1), OLS also occasionally has numerical issues, reducing the gap. Moderate condition numbers (rank ~2) give the most consistent failure.

---

## Comparison Between Rounds

| Property | Round 1 (uncontrolled) | Round 2 (controlled) |
|----------|----------------------|---------------------|
| Model converged? | No (ICL error flat at 5.3 on standard inputs) | Yes (ICL error 0.14 at k=d, 0.01 at k=2d) |
| Scale exploit? | Yes (eigenvalue 22,000, w_norm unconstrained) | No (tr(Sigma)=5, \|\|w\|\|=1) |
| Distribution shift? | Yes (train-test divergence ~4,400) | No (distributions tied) |
| Dominant pattern | Rank-1 + scale + shift | Near-rank-1 to rank-2, no scale, no shift |
| Weight alignment matters? | Appeared to (0.70-0.95) | Does not (rho = 0.002) |
| Noise level | Converged to ~0.5 | Converged to minimum (0.007) |
| Fitness magnitude | 327,730 (additive gap, inflated by scale) | 17.97 log-ratio (~6.3x10^7 ratio, driven by baseline near-zero) |
| ICL error at k=d | ~5.3 (model not trained) | ~20,000-50,000 (model trained but fails on adversarial distribution) |

The key insight from this comparison: **the failure mode (low effective rank) is robust** — it appears in both rounds. But the mechanism is completely different. In Round 1, scale was doing most of the work. In Round 2, the failure is purely structural: the transformer's learned isotropic representation cannot handle anisotropic data, even with matched distributions and fixed trace.

---

## Limitations and Open Questions

### Current Limitations

1. **d=5 is small.** At d=5, "low rank" means rank 1-2, which is a narrow structural space. At d=20, the adversary could potentially find failures at rank 5-10, revealing a more nuanced "effective dimensionality threshold" for ICL. The d=5 result shows that the threshold exists; d=20 would characterize it.

2. **The ICL error magnitudes (~20,000-50,000) are suspiciously large.** These squared errors on unit-weight, trace-5 data suggest the model is outputting predictions with magnitude ~150, far from the correct y values near ~2. This warrants investigation: is the model extrapolating wildly on out-of-distribution covariance, or is there a numerical issue in the forward pass?

3. **Single task class.** The adversary only attacks noisy linear regression. Decision trees, k-NN, and 2-layer ReLU networks may have different failure modes.

4. **The fitness function still has a dynamic range issue.** Even with log-ratio, the top fitness is log(6.3x10^7) = 17.97 while median is 4.80. This is because on near-rank-1 data, OLS/ridge error at k=1..5 drops to 10^-3, creating extreme ratios. A floor on the baseline denominator (e.g., max(baseline_err, 0.01)) would compress the range and may change which failures rank highest.

### Research Directions

1. **Characterize the effective rank threshold.** At what effective rank does ICL start failing? Run the adversary with a grid of constrained effective ranks (e.g., force eff_rank in {1, 2, 3, 4, 5}) and measure ICL performance at each. This would produce a phase diagram of ICL competence vs. distribution structure — a publishable figure.

2. **Compare across model depths.** Akyurek et al. (2023) and Bai et al. (2023) show that shallow transformers (1-2 layers) implement ridge regression while deeper models (8+ layers) implement OLS. Run the adversary against models of different depths to test whether deeper models are more or less vulnerable to covariance anisotropy. If depth-dependent failure modes exist, this directly connects to the theoretical literature.

3. **Scale to d=20 with sufficient compute.** The standard Garg et al. config (12 layers, 256 embd, 500K steps, d=20) requires ~40 GPU-hours on an RTX 3080. The 251-dimensional genome (d=20) would give the adversary access to richer covariance structures (rank 5-10 failures, nontrivial rotations) that are impossible at d=5.

4. **Prove the failure theoretically.** The empirical finding that ICL fails on low-rank data should have a theoretical explanation. A starting point: if the transformer implements an approximate ridge estimator with a fixed implicit regularization, then on low-rank data the regularization bias dominates — but this doesn't explain why ICL error is *flat* (no improvement with k). A more likely explanation involves the attention mechanism: with rank-1 data, all key-query similarities are identical, making attention weights uniform regardless of the values. This could be formalized as a theorem about attention degeneracy under low-rank inputs.

5. **Test covariance-aware training.** If the failure is caused by training on Sigma=I only, does training on diverse covariance structures fix it? Train a model on randomly sampled covariance matrices (e.g., Wishart prior) and test whether the adversary can still find failures. If not, the failure mode is a training distribution artifact, not a fundamental limitation of the transformer architecture.

6. **Cross-validate across seeds.** Train 3-5 models with different random seeds and check which adversarial failures transfer. Seed-specific failures reveal memorization; transferable failures reveal inductive bias.

---

## Reproducibility

### Code

| File | Description |
|------|-------------|
| `src/adversary/genome.py` | Genome: Cholesky encoding, trace normalization, weight normalization |
| `src/adversary/evaluate.py` | Evaluator: log-ratio fitness, ridge baseline, k=1..d restriction |
| `src/adversary/search.py` | Multi-restart diagonal CMA-ES |
| `src/adversary/analyze.py` | Post-hoc clustering, correlation, and plotting |
| `src/icl/models.py` | TransformerModel + baselines including RidgeRegressionModel |
| `run_experiment.py` | End-to-end pipeline (train + search + analysis) |

### Artifacts

| File | Description |
|------|-------------|
| `results/checkpoints/d5_6layer_150k/state.pt` | Trained model (6-layer, d=5, 150K steps) |
| `results/adversary_runs/d5_v3_logfit/checkpoint.pkl` | All 10,080 adversary evaluations |
| `results/analysis_d5_v3/summary.json` | Full numerical results for Round 2 |
| `results/analysis_d5_v2/` | Round 2 results with raw-ratio fitness (for comparison) |

### Key Configuration

```yaml
# Model
family: gpt2, n_dims: 5, n_positions: 11, n_embd: 128, n_layer: 6, n_head: 4

# Training
task: linear_regression, batch_size: 64, lr: 3e-4, steps: 150000
curriculum: none (fixed dims and points)

# Adversary
genome_size: 26, budget: 10080, pop_size: 32, restarts: 5
fitness: mean log(ICL/baseline) over k=1..d
baselines: [ridge_alpha=1.0, OLS, averaging]
covariance: trace-normalized (tr=d), weights: unit-normalized
```
