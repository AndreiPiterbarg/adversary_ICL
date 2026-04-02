"""
End-to-end experiment: train ICL model, run adversarial search, analyze results.

Usage: python run_experiment.py
"""

import os
import sys
import time
import pickle
import json

import numpy as np
import torch
import yaml

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.icl.models import build_model, TransformerModel, LeastSquaresModel, RidgeRegressionModel, AveragingModel
from src.icl.schema import AttrDict, load_config
from src.icl.train import train
from src.adversary.genome import Genome
from src.adversary.evaluate import GenomeEvaluator, EvalResult
from src.adversary.search import cma_search


# ===========================================================================
# STEP 1: Train an ICL model
# ===========================================================================

def train_icl_model(out_dir, train_steps=500000, n_dims=20, n_layer=12, n_head=8, n_embd=256):
    """Train an ICL model for linear regression matching Garg et al. config."""
    os.makedirs(out_dir, exist_ok=True)

    # n_positions = 2*n_dims + 1 gives enough overdetermination
    n_points_end = 2 * n_dims + 1

    config = {
        "out_dir": out_dir,
        "test_run": True,  # disables wandb
        "model": {
            "family": "gpt2",
            "n_dims": n_dims,
            "n_positions": n_points_end,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
        },
        "training": {
            "task": "linear_regression",
            "task_kwargs": {},
            "data": "gaussian",
            "batch_size": 64,
            "learning_rate": 3e-4,
            "train_steps": train_steps,
            "save_every_steps": train_steps // 5,  # save 5 checkpoints
            "keep_every_steps": -1,
            "resume_id": None,
            "num_tasks": None,
            "num_training_examples": None,
            "curriculum": {
                "dims": {"start": n_dims, "end": n_dims, "inc": 1, "interval": 2000},
                "points": {"start": n_points_end, "end": n_points_end, "inc": 2, "interval": 2000},
            },
        },
        "wandb": {
            "project": "test", "entity": "test", "notes": "", "name": "test",
            "log_every_steps": 500,
        },
    }

    args = load_config(config)
    args.test_run = False
    args.out_dir = out_dir

    # Save config for later loading
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(args.toDict(), f, default_flow_style=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    model.to(device)
    model.train()

    print(f"Training ICL model: {n_layer} layers, {n_head} heads, {n_embd} embd, {n_dims} dims")
    print(f"  Device: {device}")
    print(f"  Steps: {train_steps}")

    train(model, args)

    # Verify checkpoint saved
    state_path = os.path.join(out_dir, "state.pt")
    assert os.path.exists(state_path), f"Checkpoint not saved at {state_path}"
    print(f"  Model saved to {out_dir}")

    return model, args


# ===========================================================================
# STEP 2: Quick baseline evaluation (verify model learned something)
# ===========================================================================

def quick_eval(model, n_dims, n_points=41, batch_size=64, num_batches=10):
    """Quick check that the trained model does ICL on standard Gaussian inputs."""
    from src.icl.samplers import GaussianSampler
    from src.icl.tasks import get_task_sampler

    device = next(model.parameters()).device
    model.eval()

    sampler = GaussianSampler(n_dims)
    task_sampler = get_task_sampler("linear_regression", n_dims, batch_size)

    all_icl_err = []
    all_ols_err = []
    all_ridge_err = []

    ols = LeastSquaresModel()
    ridge = RidgeRegressionModel(alpha=1.0)

    for _ in range(num_batches):
        xs = sampler.sample_xs(n_points, batch_size)
        task = task_sampler()
        ys = task.evaluate(xs)

        with torch.no_grad():
            pred_icl = model(xs.to(device), ys.to(device)).cpu()
        pred_ols = ols(xs, ys)
        pred_ridge = ridge(xs, ys)

        icl_err = ((pred_icl - ys) ** 2).mean(dim=0)
        ols_err = ((pred_ols - ys) ** 2).mean(dim=0)
        ridge_err = ((pred_ridge - ys) ** 2).mean(dim=0)

        all_icl_err.append(icl_err)
        all_ols_err.append(ols_err)
        all_ridge_err.append(ridge_err)

    icl_err = torch.stack(all_icl_err).mean(dim=0).numpy()
    ols_err = torch.stack(all_ols_err).mean(dim=0).numpy()
    ridge_err = torch.stack(all_ridge_err).mean(dim=0).numpy()

    print(f"\n--- Baseline Evaluation (standard Gaussian inputs) ---")
    print(f"  {'k':>3s}  {'ICL_err':>10s}  {'OLS_err':>10s}  {'Ridge_err':>10s}  {'ICL/Ridge':>10s}")
    for k in [0, 2, 5, 10, 15, 20, 30, 40]:
        if k < len(icl_err):
            ratio = icl_err[k] / (ridge_err[k] + 1e-8)
            print(f"  {k:3d}  {icl_err[k]:10.4f}  {ols_err[k]:10.4f}  {ridge_err[k]:10.4f}  {ratio:10.2f}x")

    ratio_at_d = icl_err[min(n_dims, len(icl_err)-1)] / (ridge_err[min(n_dims, len(icl_err)-1)] + 1e-8)
    print(f"\n  ICL/Ridge ratio at k={n_dims}: {ratio_at_d:.2f}x")

    if ratio_at_d > 5.0:
        print(f"  WARNING: Model may not be fully converged (ICL/Ridge ratio > 5x at k=d)")

    return icl_err, ols_err, ridge_err


# ===========================================================================
# STEP 3: Run adversarial search
# ===========================================================================

def run_adversary(model, n_dims, n_points=41, budget=50000, pop_size=32,
                  num_restarts=5, save_dir=None):
    """Run the adversarial multi-restart CMA-ES search."""
    model.eval()

    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name="noisy_linear_regression",
        n_dims=n_dims,
        n_points=n_points,
        batch_size=64,
        num_batches=10,
        baseline_names=["ridge", "least_squares", "averaging"],
    )

    results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=budget,
        pop_size=pop_size,
        sigma_init=0.5,
        num_restarts=num_restarts,
        save_dir=save_dir,
        save_interval=50,
        seed=42,
    )

    return results


# ===========================================================================
# STEP 4: Analyze results
# ===========================================================================

def analyze_results(results, n_dims, output_dir):
    """Analyze and print detailed results."""
    os.makedirs(output_dir, exist_ok=True)

    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]

    print(f"\n{'='*70}")
    print(f"ADVERSARIAL SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(results)}")
    print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")

    if not valid:
        print("No valid results found!")
        return

    fitnesses = np.array([r.fitness for r in valid])
    print(f"\nFitness distribution (ratio - 1, so 0 = matches baseline):")
    print(f"  Min:    {fitnesses.min():.4f}")
    print(f"  Median: {np.median(fitnesses):.4f}")
    print(f"  Mean:   {fitnesses.mean():.4f}")
    print(f"  Max:    {fitnesses.max():.4f}")
    print(f"  Std:    {fitnesses.std():.4f}")

    # Top failures
    valid.sort(key=lambda r: r.fitness, reverse=True)
    top_k = min(10, len(valid))

    print(f"\n--- Top {top_k} Failure Modes ---")
    for i, r in enumerate(valid[:top_k]):
        g = r.genome
        spectrum = r.covariance_spectrum
        print(f"\n  #{i+1}: fitness={r.fitness:.4f} (ICL is {r.fitness+1:.2f}x worse than baseline)")
        print(f"    Condition number: {g.condition_number():.1f}")
        print(f"    Effective rank:   {g.effective_rank():.2f}")
        print(f"    Noise std:        {g.decode_noise_std():.4f}")
        print(f"    Top 5 eigenvalues: {spectrum[:5].round(4)}")
        if r.descriptors:
            print(f"    Weight-cov alignment: {r.descriptors.get('weight_alignment', 0):.3f}")
            print(f"    Peak failure position: {r.descriptors.get('peak_failure_position', 0):.2f}")
            print(f"    Spectral entropy:     {r.descriptors.get('spectral_entropy', 0):.3f}")

    # Detailed learning curves for top 3
    print(f"\n--- Detailed Learning Curves (Top 3) ---")
    for i, r in enumerate(valid[:3]):
        print(f"\n  Failure #{i+1} (fitness={r.fitness:.4f}):")
        bl_names = list(r.baseline_curves.keys())
        header = f"    {'k':>4s}  {'ICL_err':>10s}"
        for name in bl_names:
            header += f"  {name:>12s}"
        header += f"  {'ICL/best':>10s}"
        print(header)

        for k in range(0, len(r.icl_curve), max(1, len(r.icl_curve) // 10)):
            icl_e = r.icl_curve[k]
            best_bl = min(r.baseline_curves[name][k] for name in bl_names)
            ratio = icl_e / (best_bl + 1e-8)
            row = f"    {k:4d}  {icl_e:10.4f}"
            for name in bl_names:
                row += f"  {r.baseline_curves[name][k]:12.4f}"
            row += f"  {ratio:10.2f}x"
            print(row)

    # Descriptor correlation analysis
    print(f"\n--- What Predicts Failure? (Spearman Correlations) ---")
    from scipy.stats import spearmanr

    desc_keys = list(valid[0].descriptors.keys()) if valid[0].descriptors else []
    extra_features = {
        "condition_number": [r.genome.condition_number() for r in valid],
    }

    all_features = {}
    for key in desc_keys:
        all_features[key] = [r.descriptors.get(key, 0) for r in valid]
    all_features.update(extra_features)

    correlations = []
    for name, values in all_features.items():
        rho, pval = spearmanr(values, fitnesses)
        correlations.append((name, rho, pval))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':>30s}  {'Spearman rho':>12s}  {'p-value':>10s}")
    for name, rho, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:>30s}  {rho:12.4f}  {pval:10.6f} {sig}")

    # Plot top failures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fitness over time
        fig, ax = plt.subplots(figsize=(10, 4))
        all_fit = [r.fitness for r in results if r.is_valid]
        running_best = np.maximum.accumulate(all_fit) if all_fit else []
        ax.scatter(range(len(all_fit)), all_fit, s=1, alpha=0.3, label="Individual")
        ax.plot(running_best, color="red", linewidth=2, label="Running best")
        ax.set_xlabel("Evaluation #")
        ax.set_ylabel("Fitness (ICL/baseline ratio - 1)")
        ax.set_title("Adversarial Search Progress (Multi-Restart)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fitness_over_time.png"), dpi=150)
        plt.close()
        print(f"\n  Saved: {output_dir}/fitness_over_time.png")

        # Top 5 learning curves
        n_plots = min(5, len(valid))
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        for i, (ax, r) in enumerate(zip(axes, valid[:n_plots])):
            x = np.arange(1, len(r.icl_curve) + 1)
            ax.plot(x, r.icl_curve, label="ICL", linewidth=2)
            for name, curve in r.baseline_curves.items():
                ax.plot(x, curve, label=name, linestyle="--")
            ax.set_xlabel("k")
            ax.set_ylabel("Squared error")
            ax.set_title(f"#{i+1} fit={r.fitness:.3f}\ncond={r.genome.condition_number():.0f}")
            ax.legend(fontsize=6)
            ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_failures.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/top_failures.png")

        # Eigenvalue spectra of top 5
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
        if n_plots == 1:
            axes = [axes]
        for i, (ax, r) in enumerate(zip(axes, valid[:n_plots])):
            ax.bar(range(len(r.covariance_spectrum)), r.covariance_spectrum)
            ax.set_xlabel("Index")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"#{i+1} cond={r.genome.condition_number():.0f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_spectra.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/top_spectra.png")

        # Scatter: condition number vs fitness
        fig, ax = plt.subplots(figsize=(8, 5))
        conds = [r.genome.condition_number() for r in valid]
        ax.scatter(conds, fitnesses, s=5, alpha=0.5)
        ax.set_xlabel("Condition number (trace-normalized covariance)")
        ax.set_ylabel("Fitness (ratio - 1)")
        ax.set_title("Condition Number vs ICL Failure")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cond_vs_fitness.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/cond_vs_fitness.png")

    except Exception as e:
        print(f"  Plotting failed: {e}")

    # Save summary
    summary = {
        "total_evals": len(results),
        "valid_evals": len(valid),
        "best_fitness": float(fitnesses.max()),
        "mean_fitness": float(fitnesses.mean()),
        "top_5": [
            {
                "fitness": float(r.fitness),
                "condition_number": float(r.genome.condition_number()),
                "effective_rank": float(r.genome.effective_rank()),
                "noise_std": float(r.genome.decode_noise_std()),
                "descriptors": r.descriptors,
            }
            for r in valid[:5]
        ],
        "correlations": {name: float(rho) for name, rho, _ in correlations},
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir}/summary.json")

    return valid


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    N_DIMS = 5
    TRAIN_STEPS = 150000
    ADVERSARY_BUDGET = 10000  # 5 restarts x 2000 each
    POP_SIZE = 32
    NUM_RESTARTS = 5

    checkpoint_dir = os.path.join(ROOT, "results", "checkpoints", "d5_6layer_150k")
    adversary_dir = os.path.join(ROOT, "results", "adversary_runs", "d5_6layer_150k")
    analysis_dir = os.path.join(ROOT, "results", "analysis_d5_v2")

    print("=" * 70)
    print("STEP 1: Training ICL Transformer")
    print("=" * 70)
    t0 = time.time()
    model, args = train_icl_model(
        out_dir=checkpoint_dir,
        train_steps=TRAIN_STEPS,
        n_dims=N_DIMS,
        n_layer=6,
        n_head=4,
        n_embd=128,
    )
    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 2: Baseline Evaluation")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    n_points = 2 * N_DIMS + 1
    icl_err, ols_err, ridge_err = quick_eval(model, N_DIMS, n_points=n_points)

    print("\n" + "=" * 70)
    print("STEP 3: Adversarial Search")
    print("=" * 70)
    t0 = time.time()
    results = run_adversary(
        model, N_DIMS, n_points=n_points,
        budget=ADVERSARY_BUDGET, pop_size=POP_SIZE,
        num_restarts=NUM_RESTARTS,
        save_dir=adversary_dir,
    )
    t_search = time.time() - t0
    print(f"\nSearch time: {t_search:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 4: Analysis")
    print("=" * 70)
    analyze_results(results, N_DIMS, analysis_dir)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"  Training: {t_train:.1f}s")
    print(f"  Search:   {t_search:.1f}s")
    print(f"  Results:  {analysis_dir}")
    print(f"{'='*70}")
