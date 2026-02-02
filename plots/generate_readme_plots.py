"""Generate publication-quality figures for README / paper.

Requires a LaTeX installation (e.g. MiKTeX or TeX Live) for Computer Modern
font rendering via matplotlib's pgf backend.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("pgf")          # LaTeX-native rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Publication rcParams (LaTeX + Computer Modern) ────────────────────────────
plt.rcParams.update({
    # LaTeX rendering
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.preamble": "\n".join([
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
    ]),
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # Sizes — target single-column (3.25 in) or double-column (6.75 in)
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    # Axes
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.labelpad": 3,
    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.5,
    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # Grid
    "grid.linewidth": 0.3,
    "grid.alpha": 0.5,
    # Output
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Tol muted palette — colorblind-safe
COLORS = {
    "1-10":    "#332288",   # indigo
    "10-50":   "#44AA99",   # teal
    "50-100":  "#CC6677",   # rose
    "100-200": "#DDCC77",   # sand
}
KAPPA_LABELS = {
    "1-10":    r"$\kappa \in [1, 10]$",
    "10-50":   r"$\kappa \in [10, 50]$",
    "50-100":  r"$\kappa \in [50, 100]$",
    "100-200": r"$\kappa \in [100, 200]$",
}
KAPPA_ORDER = ["1-10", "10-50", "50-100", "100-200"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _plot_mse_lines(ax, mse_dict, add_labels=False):
    """Plot MSE curves for all kappa ranges on a given axes."""
    for kappa in KAPPA_ORDER:
        d = mse_dict[kappa]
        # Support both key styles from the two JSON files
        inner = d.get("mse_by_iteration") or d.get("mse_per_iteration")
        iters = sorted(inner.keys(), key=int)
        x = np.array([int(i) for i in iters])
        mu = np.array([inner[i]["mean"] for i in iters])
        sd = np.array([inner[i]["std"] for i in iters])
        label = KAPPA_LABELS[kappa] if add_labels else None
        ax.plot(x, mu, "-", color=COLORS[kappa], marker="o",
                markersize=2.5, markeredgewidth=0, label=label)
        ax.fill_between(x, np.clip(mu - sd, 1e-9, None), mu + sd,
                        color=COLORS[kappa], alpha=0.10, linewidth=0)


def fig1_comparison():
    """Side-by-side: naive failure vs. RDR success (independent y-axes).

    Panel (a) uses a log scale spanning the full blowup.
    Panel (b) normalises each kappa curve by its iteration-0 MSE and uses a
    linear scale so the monotonic decrease is clearly visible.
    """
    naive = load_json(os.path.join(PROJECT_ROOT, "results", "section1",
                                    "naive_refinement_results.json"))["testing"]
    rdr = load_json(os.path.join(PROJECT_ROOT, "results", "section2",
                                  "evaluation", "evaluation_results.json"))["mse_curves"]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(5.5, 2.4),
        gridspec_kw={"wspace": 0.30})

    # ── Left panel: Naive (log scale, absolute MSE) ──
    _plot_mse_lines(ax_l, naive, add_labels=False)
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Refinement iteration")
    ax_l.set_ylabel("Mean squared error")
    ax_l.set_title(r"\textbf{(a)} Naive self-refinement")
    ax_l.set_xticks(range(5))
    ax_l.set_ylim(2e-5, 2.0)
    ax_l.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax_l.grid(True, which="major", linewidth=0.3, color="0.80")
    ax_l.grid(True, which="minor", linewidth=0.15, color="0.90")

    # Degradation annotation
    ax_l.annotate(
        r"${\sim}8600\times$ degradation",
        xy=(1, 0.68), xytext=(2.5, 0.004),
        fontsize=7, ha="center", color="#aa0000",
        arrowprops=dict(arrowstyle="->, head_width=0.12, head_length=0.08",
                        color="#aa0000", lw=0.6, shrinkA=1, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#aa0000",
                  lw=0.4, alpha=0.95))

    # ── Right panel: RDR (linear scale, normalised to iter-0) ──
    all_ratios = []
    for kappa in KAPPA_ORDER:
        d = rdr[kappa]
        inner = d.get("mse_by_iteration") or d.get("mse_per_iteration")
        iters = sorted(inner.keys(), key=int)
        x = np.array([int(i) for i in iters])
        mu = np.array([inner[i]["mean"] for i in iters])
        sd = np.array([inner[i]["std"] for i in iters])
        mu0 = mu[0]
        ratios = mu / mu0
        all_ratios.extend(ratios)
        ax_r.plot(x, ratios, "-", color=COLORS[kappa], marker="o",
                  markersize=3.5, markeredgewidth=0.4, markeredgecolor="black",
                  label=KAPPA_LABELS[kappa])

    ax_r.set_xlabel("Refinement iteration")
    ax_r.set_ylabel(r"MSE / MSE$_0$")
    ax_r.set_title(r"\textbf{(b)} Role-Disambiguated Residual")
    ax_r.set_xticks(range(0, 15, 2))
    ymin = min(all_ratios) * 0.95
    ax_r.set_ylim(ymin, 1.05)
    ax_r.axhline(1.0, color="0.5", linewidth=0.4, linestyle="--", zorder=0)
    ax_r.grid(True, which="major", linewidth=0.3, color="0.80")
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=2, columnspacing=1.0, handletextpad=0.4)

    out = os.path.join(SCRIPT_DIR, "fig1_naive_vs_rdr.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig2_algorithm_heatmap():
    """Algorithm identification: cosine similarity heatmap."""
    data = load_json(os.path.join(PROJECT_ROOT, "results", "section3",
                                   "hypothesis_results.json"))

    algorithms = ["newton", "jacobi", "steepest_descent", "richardson",
                  "gradient_descent"]
    algo_labels = ["Newton", "Jacobi", "Steepest Desc.", "Richardson",
                   "Gradient Desc."]

    matrix = np.zeros((len(algorithms), len(KAPPA_ORDER)))
    for j, kappa in enumerate(KAPPA_ORDER):
        for i, algo in enumerate(algorithms):
            matrix[i, j] = data[kappa][algo]["cosine_similarity"]["mean"]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    cmap = plt.cm.YlGnBu
    norm = mcolors.Normalize(vmin=0.60, vmax=1.0)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(algorithms)):
        for j in range(len(KAPPA_ORDER)):
            val = matrix[i, j]
            rgba = cmap(norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.45 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=tc)

    ax.set_xticks(range(len(KAPPA_ORDER)))
    ax.set_xticklabels([KAPPA_LABELS[k] for k in KAPPA_ORDER], fontsize=7)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algo_labels, fontsize=7.5)
    ax.set_title(r"Cosine similarity to learned correction")
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, aspect=15)
    cbar.ax.tick_params(labelsize=6.5, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)

    out = os.path.join(SCRIPT_DIR, "fig2_algorithm_heatmap.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig1_comparison()
    fig2_algorithm_heatmap()
    print("All figures generated.")
