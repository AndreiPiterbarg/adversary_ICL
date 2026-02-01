"""
Section 3: Analysis - What Algorithm Does the Model Learn?

This is the key scientific contribution: testing whether the learned refinement
corresponds to known iterative algorithms.

Hypothesis Testing Framework:
-----------------------------
For each refinement step, observe:
- Current estimate: x_k
- Correction: delta_k = f(context, query, x_k)
- Next estimate: x_{k+1} = x_k + delta_k
- Residual: r_k = b - A @ x_k
- Gradient: grad_k = A.T @ (A @ x_k - b) = A.T @ (-r_k)

The question: What does delta_k approximate?

Candidate Algorithms to Test:
1. Richardson Iteration:     delta = alpha * r      (scaled residual)
2. Gradient Descent:         delta = -alpha * grad  (negative gradient)
3. Newton's Method:          delta = A^{-1} @ r     (exact correction)
4. Jacobi Iteration:         delta = D^{-1} @ r     (diagonal preconditioner)
5. Steepest Descent:         delta = (r^T r)/(r^T A r) * r (optimal step)

Statistical Tests:
- Compute correlation between delta_k and candidate corrections
- Test if relationship holds across different kappa ranges
- Analyze learned "step size" alpha and how it adapts

Key Research Questions:
- Does the learned algorithm change with condition number?
- Does it interpolate between simple (Richardson) and complex (Newton)?
- Can we extract the learned step size schedule?

Usage:
    python experiments/section3_analysis/hypothesis_tests.py --model_path path/to/model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

import sys
_src_dir = Path(__file__).parent.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import (
    ComponentTransformerModel,
    ComponentModelConfig,
)
from curriculum_model.roles import Role
from data.spd_sampler import sample_spd


@dataclass
class AnalysisConfig:
    """Configuration for hypothesis testing."""
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4

    num_context: int = 5
    num_samples: int = 1000  # Samples for statistical analysis
    batch_size: int = 64

    # Kappa ranges to analyze
    kappa_ranges: List[Tuple[float, float]] = None

    output_dir: str = "results/section3"
    device: str = "cuda"

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]


class AlgorithmHypothesisTester:
    """
    Tests whether the learned correction corresponds to known algorithms.

    For each sample, computes:
    1. Model's predicted correction: delta = f(context, query, x_k)
    2. Candidate corrections:
       - Richardson: alpha * r_k where r_k = b - A @ x_k
       - Gradient Descent: -alpha * A.T @ (A @ x_k - b)
       - Newton: A^{-1} @ r_k
       - Jacobi: diag(A)^{-1} @ r_k
       - Steepest Descent: (r^T r)/(r^T A r) * r

    Then computes:
    - Cosine similarity between delta and each candidate
    - Optimal alpha that minimizes ||delta - alpha * candidate||
    - R^2 of linear fit
    """

    def __init__(
        self,
        model: ComponentTransformerModel,
        config: AnalysisConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.d = config.d

        # Cache role indices for token building
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        """Get role embedding by name."""
        return self.model.role_embedding(self._role_indices[name])

    def compute_candidate_corrections(
        self,
        A: torch.Tensor,      # (B, d, d) SPD matrices
        b: torch.Tensor,      # (B, d) right-hand side
        x_k: torch.Tensor,    # (B, d) current estimate
    ) -> Dict[str, torch.Tensor]:
        """
        Compute candidate corrections for comparison.

        Returns:
            Dict mapping algorithm name to correction tensor (B, d)
        """
        B, d = x_k.shape

        # Residual: r = b - A @ x
        r = b - torch.bmm(A, x_k.unsqueeze(-1)).squeeze(-1)

        corrections = {}

        # Richardson iteration: delta = r (scaled by optimal alpha later)
        corrections["richardson"] = r

        # Gradient descent: delta = A.T @ r = A @ r (since A is SPD)
        # Note: For SPD matrices, A = A.T
        corrections["gradient_descent"] = torch.bmm(A, r.unsqueeze(-1)).squeeze(-1)

        # Newton's method: delta = A^{-1} @ r (exact correction)
        corrections["newton"] = torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1)

        # Jacobi iteration: delta = D^{-1} @ r where D = diag(A)
        D_inv = 1.0 / torch.diagonal(A, dim1=-2, dim2=-1)  # (B, d)
        corrections["jacobi"] = D_inv * r

        # Steepest descent with optimal step size: alpha = (r^T r) / (r^T A r)
        # delta = alpha * r
        Ar = torch.bmm(A, r.unsqueeze(-1)).squeeze(-1)  # (B, d)
        r_norm_sq = (r * r).sum(dim=-1, keepdim=True)  # (B, 1)
        rAr = (r * Ar).sum(dim=-1, keepdim=True)  # (B, 1)
        alpha_steepest = r_norm_sq / (rAr + 1e-10)  # (B, 1)
        corrections["steepest_descent"] = alpha_steepest * r

        return corrections

    def _build_tokens_with_estimate(
        self,
        A: torch.Tensor,       # (B, d, d)
        b_ctx: torch.Tensor,   # (B, K, d)
        x_ctx: torch.Tensor,   # (B, K, d)
        b_query: torch.Tensor, # (B, d)
        x_estimate: torch.Tensor,  # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build tokens with current estimate for residual prediction.

        Format: [SEP, A, SEP, b_0, x_0, ..., SEP, b_query, x_estimate, MASK]

        Returns:
            tokens: (B, seq_len, n_embd)
            ex_pos: (B, seq_len) example-level positions
            mask_pos: (B,) indices of MASK tokens
        """
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device

        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
        output_role = self._get_role('output')
        estimate_role = self._get_role('estimate')

        A_emb = embedders.matrix(A) + matrix_role
        b_flat = b_ctx.reshape(B * K, d)
        x_flat = x_ctx.reshape(B * K, d)
        n_embd = embedders.vector(b_flat).shape[-1]

        b_emb = embedders.vector(b_flat).reshape(B, K, n_embd) + bias_role
        x_emb = embedders.vector(x_flat).reshape(B, K, n_embd) + output_role
        b_q_emb = embedders.vector(b_query) + bias_role
        x_est_emb = embedders.vector(x_estimate) + estimate_role

        seq_len = 3 * K + 6
        tokens = torch.zeros(B, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        sep, mask = special.get_sep_batch(B), special.get_mask_batch(B)
        tokens[:, 0], tokens[:, 1] = sep, A_emb

        for i in range(K):
            idx = 2 + i * 3
            tokens[:, idx] = sep
            tokens[:, idx + 1] = b_emb[:, i]
            tokens[:, idx + 2] = x_emb[:, i]
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx] = sep
        tokens[:, q_idx + 1] = b_q_emb
        tokens[:, q_idx + 2] = x_est_emb
        tokens[:, q_idx + 3] = mask
        ex_pos[:, q_idx:q_idx + 4] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)

    def get_model_correction(
        self,
        A: torch.Tensor,
        b_ctx: torch.Tensor,
        x_ctx: torch.Tensor,
        b_query: torch.Tensor,
        x_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the model's predicted correction delta_k.

        Args:
            A: (B, d, d) SPD matrices
            b_ctx: (B, K, d) context b vectors
            x_ctx: (B, K, d) context solutions
            b_query: (B, d) query b vector
            x_k: (B, d) current estimate

        Returns:
            Tensor of shape (B, d) with predicted corrections
        """
        tokens, ex_pos, mask_pos = self._build_tokens_with_estimate(
            A, b_ctx, x_ctx, b_query, x_k
        )
        output = self.model(tokens, ex_pos, mask_pos)
        return output.vector_output

    def compute_algorithm_fit(
        self,
        model_delta: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute how well the model's correction fits a candidate algorithm.

        Args:
            model_delta: (B, d) model predictions
            candidate_delta: (B, d) candidate algorithm predictions

        Returns:
            Dict with:
            - cosine_similarity: avg cos sim between model and candidate
            - optimal_alpha: alpha that minimizes ||delta - alpha * candidate||
            - r_squared: coefficient of determination
            - mse: mean squared error with optimal alpha
        """
        # Flatten to (B*d,) for scalar statistics
        model_flat = model_delta.reshape(-1)
        candidate_flat = candidate_delta.reshape(-1)

        # Cosine similarity (per sample, then average)
        model_norm = model_delta.norm(dim=-1, keepdim=True) + 1e-10
        candidate_norm = candidate_delta.norm(dim=-1, keepdim=True) + 1e-10
        cos_sim_per_sample = (model_delta * candidate_delta).sum(dim=-1) / (
            model_norm.squeeze(-1) * candidate_norm.squeeze(-1)
        )
        cosine_similarity = cos_sim_per_sample.mean().item()

        # Optimal alpha: minimize ||model - alpha * candidate||^2
        # d/dalpha = -2 * candidate^T (model - alpha * candidate) = 0
        # alpha = (candidate^T model) / (candidate^T candidate)
        numerator = (candidate_flat * model_flat).sum()
        denominator = (candidate_flat * candidate_flat).sum() + 1e-10
        optimal_alpha = (numerator / denominator).item()

        # Compute MSE with optimal alpha
        scaled_candidate = optimal_alpha * candidate_flat
        residual = model_flat - scaled_candidate
        mse = (residual * residual).mean().item()

        # R² = 1 - SS_res / SS_tot
        # SS_tot = sum((model - mean(model))^2)
        model_mean = model_flat.mean()
        ss_tot = ((model_flat - model_mean) ** 2).sum().item() + 1e-10
        ss_res = (residual ** 2).sum().item()
        r_squared = 1.0 - ss_res / ss_tot

        return {
            "cosine_similarity": cosine_similarity,
            "optimal_alpha": optimal_alpha,
            "r_squared": r_squared,
            "mse": mse,
        }

    def run_hypothesis_tests(
        self,
        kappa_min: float,
        kappa_max: float,
    ) -> Dict[str, Dict]:
        """
        Run hypothesis tests for a specific kappa range.

        Args:
            kappa_min: Minimum condition number
            kappa_max: Maximum condition number

        Returns:
            Dict mapping algorithm names to fit statistics (aggregated)
        """
        B = self.config.batch_size
        K = self.config.num_context
        d = self.d
        num_batches = self.config.num_samples // B

        # Accumulate fit statistics for each algorithm
        algorithm_stats = {}

        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Sample data
                A = sample_spd(B, d, self.device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=self.device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx = b_all[:, :K]
                x_ctx = x_all[:, :K]
                b_query = b_all[:, K]
                x_target = x_all[:, K]

                # Create a noisy initial estimate (simulate refinement scenario)
                noise = torch.randn_like(x_target) * 0.5
                x_k = x_target + noise

                # Get model's correction
                model_delta = self.get_model_correction(A, b_ctx, x_ctx, b_query, x_k)

                # Compute candidate corrections
                candidates = self.compute_candidate_corrections(A, b_query, x_k)

                # Compute fit for each candidate
                for algo_name, candidate_delta in candidates.items():
                    fit_stats = self.compute_algorithm_fit(model_delta, candidate_delta)

                    if algo_name not in algorithm_stats:
                        algorithm_stats[algo_name] = {
                            key: [] for key in fit_stats.keys()
                        }

                    for key, value in fit_stats.items():
                        algorithm_stats[algo_name][key].append(value)

        # Aggregate statistics
        aggregated = {}
        for algo_name, stats in algorithm_stats.items():
            aggregated[algo_name] = {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                for key, values in stats.items()
            }

        return aggregated


def analyze_algorithm_by_kappa(
    model: ComponentTransformerModel,
    config: AnalysisConfig,
) -> Dict[str, Dict]:
    """
    Analyze what algorithm the model learns for each kappa range.

    Output format:
    {
        "1-10": {
            "richardson": {"cosine_similarity": {...}, "r_squared": {...}, ...},
            "gradient_descent": {...},
            "newton": {...},
            ...
        },
        "50-100": {...}
    }

    Key findings to look for:
    - Does the best-fit algorithm change with kappa?
    - Is there a transition from simple to complex algorithms?
    - Does the learned alpha adapt to the problem difficulty?
    """
    device = torch.device(config.device)
    tester = AlgorithmHypothesisTester(model, config, device)

    results = {}
    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        print(f"\nAnalyzing kappa range [{kappa_min}, {kappa_max}]...")
        results[kappa_key] = tester.run_hypothesis_tests(kappa_min, kappa_max)

    return results


def generate_analysis_report(results: Dict[str, Dict]) -> str:
    """
    Generate human-readable report of algorithm analysis.

    Includes:
    - Table of fit statistics per kappa range
    - Best-fit algorithm for each range
    - Discussion of transitions/patterns
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ALGORITHM HYPOTHESIS TESTING REPORT")
    lines.append("=" * 70)

    # Summary table
    lines.append("\nSUMMARY: Best-fit algorithm by kappa range")
    lines.append("-" * 70)
    lines.append(f"{'Kappa Range':<15} {'Best Algorithm':<20} {'Cosine Sim':<12} {'R²':<12}")
    lines.append("-" * 70)

    for kappa_key, algo_results in results.items():
        # Find best algorithm by cosine similarity
        best_algo = max(
            algo_results.keys(),
            key=lambda a: algo_results[a]["cosine_similarity"]["mean"]
        )
        cos_sim = algo_results[best_algo]["cosine_similarity"]["mean"]
        r_sq = algo_results[best_algo]["r_squared"]["mean"]
        lines.append(f"κ ∈ [{kappa_key}]".ljust(15) + f"{best_algo:<20} {cos_sim:.4f}      {r_sq:.4f}")

    # Detailed results per kappa range
    for kappa_key, algo_results in results.items():
        lines.append(f"\n{'='*70}")
        lines.append(f"KAPPA RANGE: [{kappa_key}]")
        lines.append("=" * 70)
        lines.append(f"{'Algorithm':<20} {'Cos Sim':<12} {'R²':<12} {'α (optimal)':<12} {'MSE':<12}")
        lines.append("-" * 70)

        # Sort by cosine similarity descending
        sorted_algos = sorted(
            algo_results.items(),
            key=lambda x: x[1]["cosine_similarity"]["mean"],
            reverse=True
        )

        for algo_name, stats in sorted_algos:
            cos_sim = stats["cosine_similarity"]["mean"]
            r_sq = stats["r_squared"]["mean"]
            alpha = stats["optimal_alpha"]["mean"]
            mse = stats["mse"]["mean"]
            lines.append(f"{algo_name:<20} {cos_sim:.4f}       {r_sq:.4f}       {alpha:.4f}       {mse:.6f}")

    # Analysis observations
    lines.append("\n" + "=" * 70)
    lines.append("KEY OBSERVATIONS")
    lines.append("=" * 70)

    # Check if Newton wins everywhere (would suggest exact correction)
    newton_best_count = 0
    for kappa_key, algo_results in results.items():
        best_algo = max(
            algo_results.keys(),
            key=lambda a: algo_results[a]["cosine_similarity"]["mean"]
        )
        if best_algo == "newton":
            newton_best_count += 1

    if newton_best_count == len(results):
        lines.append("• The model appears to learn Newton's method (exact correction)")
        lines.append("  across all condition number ranges. This is the optimal solution!")
    else:
        lines.append("• The best-fit algorithm varies with condition number range.")
        lines.append("  This suggests the model may interpolate between algorithms.")

    # Check if alpha varies significantly across kappa ranges
    alpha_by_kappa = {}
    for kappa_key, algo_results in results.items():
        if "richardson" in algo_results:
            alpha_by_kappa[kappa_key] = algo_results["richardson"]["optimal_alpha"]["mean"]

    if alpha_by_kappa:
        alphas = list(alpha_by_kappa.values())
        if max(alphas) > 2 * min(alphas):
            lines.append("• The optimal step size (α) varies significantly with kappa,")
            lines.append("  suggesting the model adapts its behavior to problem difficulty.")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test what algorithm the model learns")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results/section3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for analysis")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d", type=int, default=4, help="Vector dimension")
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--num_context", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("SECTION 3: ALGORITHM HYPOTHESIS TESTING")
    print("=" * 60)
    print("\nThis analysis tests whether the learned correction matches:")
    print("  1. Richardson Iteration: delta = alpha * r")
    print("  2. Gradient Descent:     delta = alpha * A @ r")
    print("  3. Newton's Method:      delta = A^{-1} @ r")
    print("  4. Jacobi Iteration:     delta = D^{-1} @ r")
    print("  5. Steepest Descent:     delta = (r'r)/(r'Ar) * r")

    # Load model
    device = torch.device(args.device)
    print(f"\nLoading model from {args.model_path}...")

    model_config = ComponentModelConfig(
        d=args.d,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=128,
        max_examples=64,
        dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # Create analysis config
    config = AnalysisConfig(
        d=args.d,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        num_context=args.num_context,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run analysis
    print(f"\nAnalyzing with {config.num_samples} samples per kappa range...")
    results = analyze_algorithm_by_kappa(model, config)

    # Generate report
    report = generate_analysis_report(results)
    print("\n" + report)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "hypothesis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir / 'hypothesis_results.json'}")
    print(f"Report saved to {output_dir / 'hypothesis_report.txt'}")


if __name__ == "__main__":
    main()
