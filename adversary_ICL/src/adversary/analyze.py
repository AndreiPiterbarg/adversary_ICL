import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from .evaluate import EvalResult


def load_results(save_dir: str) -> list[EvalResult]:
    """Load results from a checkpoint file."""
    path = os.path.join(save_dir, "checkpoint.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["results"]


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    """Convert results into a DataFrame for analysis."""
    rows = []
    for r in results:
        if not r.is_valid:
            continue
        row = {
            "fitness": r.fitness,
            "condition_number": r.genome.condition_number(),
            "effective_rank": r.genome.effective_rank(),
            "noise_std": r.genome.decode_noise_std(),
        }
        row.update(r.descriptors)
        rows.append(row)
    return pd.DataFrame(rows)


def top_failures(results: list[EvalResult], k: int = 10) -> list[EvalResult]:
    """Return the top-k highest-fitness valid results."""
    valid = [r for r in results if r.is_valid]
    valid.sort(key=lambda r: r.fitness, reverse=True)
    return valid[:k]


def correlate_with_fitness(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each feature with fitness."""
    cols = [c for c in df.columns if c != "fitness"]
    rows = []
    for col in cols:
        mask = df[col].notna()
        if mask.sum() < 10:
            continue
        rho, pval = spearmanr(df.loc[mask, col], df.loc[mask, "fitness"])
        rows.append({"feature": col, "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False, key=abs)


def cluster_failures(results: list[EvalResult], min_fitness: float = 1.5):
    """Cluster high-fitness results by covariance spectrum using HDBSCAN."""
    try:
        import hdbscan
    except ImportError:
        print("Install hdbscan for clustering: pip install hdbscan")
        return None, None

    valid = [r for r in results if r.is_valid and r.fitness >= min_fitness]
    if len(valid) < 10:
        print(f"Only {len(valid)} results above fitness {min_fitness}, skipping clustering.")
        return None, None

    # Feature matrix: covariance eigenvalue spectra
    spectra = np.array([r.covariance_spectrum for r in valid])
    # Log-transform for better clustering
    spectra_log = np.log(spectra + 1e-8)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(spectra_log)

    return valid, labels


def describe_cluster(results: list[EvalResult], label: int, labels: np.ndarray, all_results: list) -> str:
    """Auto-generate a description for a cluster of failures."""
    cluster_results = [r for r, l in zip(all_results, labels) if l == label]
    if not cluster_results:
        return "Empty cluster"

    fitnesses = [r.fitness for r in cluster_results]
    conds = [r.genome.condition_number() for r in cluster_results]
    ranks = [r.genome.effective_rank() for r in cluster_results]
    noises = [r.genome.decode_noise_std() for r in cluster_results]

    descs = [r.descriptors for r in cluster_results if r.descriptors]
    if descs:
        alignments = [d.get("weight_alignment", 0) for d in descs]
    else:
        alignments = [0]

    return (
        f"Cluster {label} ({len(cluster_results)} elites, "
        f"mean fitness={np.mean(fitnesses):.4f}): "
        f"cond_number={np.mean(conds):.1f} +/- {np.std(conds):.1f}, "
        f"eff_rank={np.mean(ranks):.1f}, "
        f"noise={np.mean(noises):.3f}, "
        f"weight_align={np.mean(alignments):.2f}"
    )


# --- Plotting ---


def plot_top_failures(results: list[EvalResult], output_dir: str, k: int = 5):
    """Plot learning curves and spectra for top-k failures."""
    os.makedirs(output_dir, exist_ok=True)
    top = top_failures(results, k)

    for i, r in enumerate(top):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Learning curves
        ax = axes[0]
        n_pts = len(r.icl_curve)
        x = np.arange(1, n_pts + 1)
        ax.plot(x, r.icl_curve, label="ICL Transformer", linewidth=2)
        for name, curve in r.baseline_curves.items():
            ax.plot(x, curve, label=name, linestyle="--")
        ax.set_xlabel("In-context examples")
        ax.set_ylabel("Squared error")
        ax.set_title(f"Failure #{i+1} (fitness={r.fitness:.4f})")
        ax.legend(fontsize=8)
        ax.set_yscale("log")

        # Eigenvalue spectrum
        ax = axes[1]
        spectrum = r.covariance_spectrum
        ax.bar(range(len(spectrum)), spectrum)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(
            f"Covariance spectrum (cond={r.genome.condition_number():.0f}, "
            f"eff_rank={r.genome.effective_rank():.1f})"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"failure_{i+1}.png"), dpi=150)
        plt.close()


def plot_fitness_over_time(results: list[EvalResult], output_dir: str):
    """Plot fitness over the course of the search."""
    os.makedirs(output_dir, exist_ok=True)
    fitnesses = [r.fitness for r in results if r.is_valid]
    if not fitnesses:
        return
    running_best = np.maximum.accumulate(fitnesses)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(fitnesses)), fitnesses, s=1, alpha=0.3, label="Individual")
    ax.plot(running_best, color="red", linewidth=2, label="Running best")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Fitness (ICL/baseline ratio - 1)")
    ax.set_title("Search Progress")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_over_time.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap of descriptors with fitness."""
    os.makedirs(output_dir, exist_ok=True)
    corr_df = correlate_with_fitness(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    features = corr_df["feature"].tolist()
    rhos = corr_df["spearman_rho"].tolist()
    colors = ["red" if r > 0 else "blue" for r in rhos]
    ax.barh(features, rhos, color=colors)
    ax.set_xlabel("Spearman correlation with fitness")
    ax.set_title("What predicts ICL failure?")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlations.png"), dpi=150)
    plt.close()


def plot_descriptor_scatter(df: pd.DataFrame, output_dir: str,
                            x_col: str = "condition_number_log",
                            y_col: str = "effective_rank"):
    """2D scatter of descriptors colored by fitness."""
    os.makedirs(output_dir, exist_ok=True)

    if x_col not in df.columns or y_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df[x_col], df[y_col],
        c=df["fitness"], cmap="hot", s=5, alpha=0.5,
    )
    plt.colorbar(sc, label="Fitness")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Failure landscape")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{x_col}_vs_{y_col}.png"), dpi=150)
    plt.close()


def run_analysis(save_dir: str, output_dir: str | None = None):
    """Full post-hoc analysis pipeline."""
    if output_dir is None:
        output_dir = os.path.join(save_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {save_dir}...")
    results = load_results(save_dir)
    valid = [r for r in results if r.is_valid]
    print(f"  {len(valid)} valid results out of {len(results)} total")

    # Top failures
    top = top_failures(results, 10)
    print(f"\nTop 10 failures:")
    for i, r in enumerate(top):
        print(f"  #{i+1}: fitness={r.fitness:.4f} | {r.genome}")

    # DataFrame
    df = results_to_dataframe(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    # Correlations
    print("\nCorrelations with fitness:")
    corr = correlate_with_fitness(df)
    print(corr.to_string(index=False))

    # Clustering
    clustered, labels = cluster_failures(results)
    if clustered is not None and labels is not None:
        unique_labels = set(labels) - {-1}
        print(f"\n{len(unique_labels)} failure clusters found:")
        for label in sorted(unique_labels):
            desc = describe_cluster(results, label, labels, clustered)
            print(f"  {desc}")

    # Plots
    print(f"\nGenerating plots in {output_dir}...")
    plot_top_failures(results, output_dir)
    plot_fitness_over_time(results, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_descriptor_scatter(df, output_dir, "condition_number_log", "effective_rank")
    plot_descriptor_scatter(df, output_dir, "noise_std", "weight_alignment")
    plot_descriptor_scatter(df, output_dir, "spectral_entropy", "peak_failure_position")

    print("Analysis complete.")
    return df, top
