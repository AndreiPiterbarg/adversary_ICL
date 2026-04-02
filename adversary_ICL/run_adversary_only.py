"""Run adversary search against already-trained model.

Legacy script from the initial d=5 experiment. Updated to use the new
genome API (tied distributions, trace normalization, unit weights).
Use run_experiment.py for the full pipeline instead.
"""

import os, sys, time, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.icl.eval import get_model_from_run
from src.icl.models import LeastSquaresModel, RidgeRegressionModel
from src.icl.samplers import GaussianSampler
from src.icl.tasks import get_task_sampler
from src.adversary.genome import Genome
from src.adversary.evaluate import GenomeEvaluator, EvalResult
from src.adversary.search import cma_search

N_DIMS = 5
N_POINTS = 11
BUDGET = 3200
POP_SIZE = 16
NUM_RESTARTS = 4  # 4 restarts x 800 budget each

checkpoint_dir = os.path.join(ROOT, "results", "checkpoints", "experiment_run")
adversary_dir = os.path.join(ROOT, "results", "adversary_runs", "run2")
analysis_dir = os.path.join(ROOT, "results", "analysis2")

# Load model
print("Loading model...")
model, conf = get_model_from_run(checkpoint_dir, step=-1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# Quick baseline check
print("\n--- Baseline check on standard Gaussian ---")
sampler = GaussianSampler(N_DIMS)
ols = LeastSquaresModel()
ridge = RidgeRegressionModel(alpha=1.0)

all_icl, all_ols, all_ridge = [], [], []
for _ in range(10):
    xs = sampler.sample_xs(N_POINTS, 64)
    task = get_task_sampler("linear_regression", N_DIMS, 64)()
    ys = task.evaluate(xs)
    with torch.no_grad():
        pred_icl = model(xs.to(device), ys.to(device)).cpu()
    pred_ols = ols(xs, ys)
    pred_ridge = ridge(xs, ys)
    all_icl.append(((pred_icl - ys)**2).mean(dim=0))
    all_ols.append(((pred_ols - ys)**2).mean(dim=0))
    all_ridge.append(((pred_ridge - ys)**2).mean(dim=0))

icl_err = torch.stack(all_icl).mean(dim=0).numpy()
ols_err = torch.stack(all_ols).mean(dim=0).numpy()
ridge_err = torch.stack(all_ridge).mean(dim=0).numpy()

print(f"  {'k':>3s}  {'ICL':>10s}  {'OLS':>10s}  {'Ridge':>10s}  {'Gap':>10s}")
for k in range(N_POINTS):
    gap = icl_err[k] - min(ols_err[k], ridge_err[k])
    print(f"  {k:3d}  {icl_err[k]:10.4f}  {ols_err[k]:10.4f}  {ridge_err[k]:10.4f}  {gap:+10.4f}")

# Test evaluator on identity genome
print("\n--- Testing evaluator with identity genome ---")
evaluator = GenomeEvaluator(
    icl_model=model, task_name="noisy_linear_regression",
    n_dims=N_DIMS, n_points=N_POINTS,
    batch_size=32, num_batches=5,
    baseline_names=["ridge", "least_squares", "averaging"],
)

g_id = Genome.identity(N_DIMS)
r_id = evaluator.evaluate(g_id)
print(f"  Identity genome fitness: {r_id.fitness:.4f} (should be small, near 0)")
print(f"  ICL curve: {r_id.icl_curve.round(4)}")
for name, curve in r_id.baseline_curves.items():
    print(f"  {name} curve: {curve.round(4)}")

# Run adversary
print(f"\n{'='*60}")
print(f"ADVERSARIAL SEARCH: budget={BUDGET}, pop_size={POP_SIZE}, restarts={NUM_RESTARTS}")
print(f"{'='*60}")
t0 = time.time()
results = cma_search(
    evaluator=evaluator, n_dims=N_DIMS,
    budget=BUDGET, pop_size=POP_SIZE,
    sigma_init=0.3, num_restarts=NUM_RESTARTS,
    save_dir=adversary_dir, save_interval=20, seed=42,
)
t_search = time.time() - t0
print(f"\nSearch time: {t_search:.1f}s")

# Analysis
print(f"\n{'='*60}")
print(f"ANALYSIS")
print(f"{'='*60}")

os.makedirs(analysis_dir, exist_ok=True)
valid = [r for r in results if r.is_valid]
fitnesses = np.array([r.fitness for r in valid])

print(f"Total: {len(results)}, Valid: {len(valid)}")
print(f"Fitness: min={fitnesses.min():.4f}, median={np.median(fitnesses):.4f}, "
      f"mean={fitnesses.mean():.4f}, max={fitnesses.max():.4f}")

valid.sort(key=lambda r: r.fitness, reverse=True)

print(f"\n--- Top 10 Failures ---")
for i, r in enumerate(valid[:10]):
    g = r.genome
    print(f"\n  #{i+1}: fitness={r.fitness:.4f}")
    print(f"    cond={g.condition_number():.1f}, eff_rank={g.effective_rank():.2f}")
    print(f"    noise={g.decode_noise_std():.4f}")
    print(f"    spectrum: {r.covariance_spectrum[:5].round(3)}")
    d = r.descriptors
    print(f"    align={d.get('weight_alignment',0):.3f}, "
          f"peak_pos={d.get('peak_failure_position',0):.2f}")

# Detailed learning curves for top 3
print(f"\n--- Detailed Learning Curves (Top 3) ---")
for i, r in enumerate(valid[:3]):
    bl_names = list(r.baseline_curves.keys())
    print(f"\n  Failure #{i+1} (fitness={r.fitness:.4f}):")
    header = f"    {'k':>3s}  {'ICL':>10s}"
    for name in bl_names:
        header += f"  {name:>12s}"
    header += f"  {'ICL/best':>10s}"
    print(header)
    for k in range(len(r.icl_curve)):
        best_bl = min(r.baseline_curves[name][k] for name in bl_names)
        ratio = r.icl_curve[k] / (best_bl + 1e-8)
        row = f"    {k:3d}  {r.icl_curve[k]:10.4f}"
        for name in bl_names:
            row += f"  {r.baseline_curves[name][k]:12.4f}"
        row += f"  {ratio:10.2f}x"
        print(row)

# Correlations
from scipy.stats import spearmanr
desc_keys = list(valid[0].descriptors.keys()) if valid[0].descriptors else []
features = {k: [r.descriptors.get(k, 0) for r in valid] for k in desc_keys}
features["condition_number"] = [r.genome.condition_number() for r in valid]

print(f"\n--- Correlations with Fitness ---")
print(f"  {'Feature':>30s}  {'rho':>8s}  {'p':>10s}")
corrs = []
for name, vals in features.items():
    try:
        rho, p = spearmanr(vals, fitnesses)
        if not np.isnan(rho):
            corrs.append((name, rho, p))
    except Exception:
        pass
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, rho, p in corrs:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:>30s}  {rho:8.4f}  {p:10.6f} {sig}")

# Plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fitness over time
    fig, ax = plt.subplots(figsize=(10, 4))
    all_fit = [r.fitness for r in results if r.is_valid]
    running_best = np.maximum.accumulate(all_fit)
    ax.scatter(range(len(all_fit)), all_fit, s=1, alpha=0.3, label="Individual")
    ax.plot(running_best, color="red", linewidth=2, label="Running best")
    ax.set_xlabel("Evaluation #"); ax.set_ylabel("Fitness (ratio - 1)"); ax.set_title("Search Progress")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "fitness_over_time.png"), dpi=150); plt.close()

    # Top 5 learning curves
    n_plot = min(5, len(valid))
    fig, axes = plt.subplots(1, n_plot, figsize=(4*n_plot, 4))
    if n_plot == 1: axes = [axes]
    for i, (ax, r) in enumerate(zip(axes, valid[:n_plot])):
        x = np.arange(len(r.icl_curve))
        ax.plot(x, r.icl_curve, label="ICL", linewidth=2)
        for nm, c in r.baseline_curves.items():
            ax.plot(x, c, label=nm, linestyle="--")
        ax.set_xlabel("k"); ax.set_ylabel("Squared error")
        ax.set_title(f"#{i+1} fit={r.fitness:.4f}\ncond={r.genome.condition_number():.0f}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "top_failures.png"), dpi=150); plt.close()

    # Spectra
    fig, axes = plt.subplots(1, n_plot, figsize=(4*n_plot, 3))
    if n_plot == 1: axes = [axes]
    for i, (ax, r) in enumerate(zip(axes, valid[:n_plot])):
        ax.bar(range(len(r.covariance_spectrum)), r.covariance_spectrum)
        ax.set_xlabel("Index"); ax.set_ylabel("Eigenvalue")
        ax.set_title(f"#{i+1} cond={r.genome.condition_number():.0f}")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "top_spectra.png"), dpi=150); plt.close()

    # Cond vs fitness
    fig, ax = plt.subplots(figsize=(8, 5))
    conds = [np.log10(r.genome.condition_number() + 1) for r in valid]
    ax.scatter(conds, fitnesses, s=5, alpha=0.5)
    ax.set_xlabel("Log10 condition number"); ax.set_ylabel("Fitness")
    ax.set_title("Condition Number vs ICL Failure"); plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "cond_vs_fitness.png"), dpi=150); plt.close()

    print(f"\nPlots saved to {analysis_dir}/")
except Exception as e:
    print(f"Plotting error: {e}")

print(f"\nDone. Search took {t_search:.1f}s")
