"""
Section 3: Kappa Range Analysis - How Does Performance Vary with Condition Number?

This analysis examines how the model's refinement performance varies with
the condition number (kappa) of the input matrices.

Key Questions:
1. At what kappa does refinement start to fail?
2. Is there a transition point where the learned algorithm changes?
3. How does convergence rate depend on kappa?

Analysis Components:
1. MSE vs Kappa: Plot MSE reduction as function of condition number
2. Convergence Curves: MSE over iterations for different kappa ranges
3. Algorithm Transition: Does the learned algorithm shift at high kappa?
4. Failure Mode Analysis: What happens when refinement fails?

Expected Results Format:
```
kappa in [1, 10]:    Model learns X, converges in K iterations
kappa in [50, 100]:  Model learns Y, converges in K' iterations
kappa in [100, 200]: Model learns Z (or fails to learn anything useful)
```

Usage:
    python experiments/section3_analysis/kappa_range_analysis.py --model_path path/to/model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
from data.spd_sampler import sample_spd


@dataclass
class KappaAnalysisConfig:
    """Configuration for kappa range analysis."""
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4

    num_context: int = 5
    batch_size: int = 64
    test_batches: int = 50
    test_iterations: int = 10

    # Fine-grained kappa ranges for detailed analysis
    kappa_points: List[float] = None  # Specific kappa values to test

    output_dir: str = "results/section3"
    device: str = "cuda"

    def __post_init__(self):
        if self.kappa_points is None:
            # Log-spaced kappa values for detailed analysis
            self.kappa_points = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500]


class KappaRangeAnalyzer:
    """Analyzes model performance as a function of condition number."""

    def __init__(
        self,
        model: ComponentTransformerModel,
        config: KappaAnalysisConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.d = config.d

        # Import Role for token building
        from curriculum_model.roles import Role
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        return self.model.role_embedding(self._role_indices[name])

    def _build_tokens_standard(
        self, A: torch.Tensor, b_ctx: torch.Tensor, x_ctx: torch.Tensor, b_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tokens for direct prediction (iteration 0)."""
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device

        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role, bias_role, output_role = self._get_role('matrix'), self._get_role('bias'), self._get_role('output')
        A_emb = embedders.matrix(A) + matrix_role
        n_embd = embedders.vector(b_ctx[:, 0]).shape[-1]

        b_emb = embedders.vector(b_ctx.reshape(B * K, d)).reshape(B, K, n_embd) + bias_role
        x_emb = embedders.vector(x_ctx.reshape(B * K, d)).reshape(B, K, n_embd) + output_role
        b_q_emb = embedders.vector(b_query) + bias_role

        seq_len = 3 * K + 5
        tokens = torch.zeros(B, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        sep, mask = special.get_sep_batch(B), special.get_mask_batch(B)
        tokens[:, 0], tokens[:, 1] = sep, A_emb

        for i in range(K):
            idx = 2 + i * 3
            tokens[:, idx], tokens[:, idx + 1], tokens[:, idx + 2] = sep, b_emb[:, i], x_emb[:, i]
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx], tokens[:, q_idx + 1], tokens[:, q_idx + 2] = sep, b_q_emb, mask
        ex_pos[:, q_idx:q_idx + 3] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)

    def _build_tokens_with_estimate(
        self, A: torch.Tensor, b_ctx: torch.Tensor, x_ctx: torch.Tensor,
        b_query: torch.Tensor, x_estimate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build tokens with current estimate for residual prediction."""
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device

        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role, bias_role = self._get_role('matrix'), self._get_role('bias')
        output_role, estimate_role = self._get_role('output'), self._get_role('estimate')

        A_emb = embedders.matrix(A) + matrix_role
        n_embd = embedders.vector(b_ctx[:, 0]).shape[-1]

        b_emb = embedders.vector(b_ctx.reshape(B * K, d)).reshape(B, K, n_embd) + bias_role
        x_emb = embedders.vector(x_ctx.reshape(B * K, d)).reshape(B, K, n_embd) + output_role
        b_q_emb = embedders.vector(b_query) + bias_role
        x_est_emb = embedders.vector(x_estimate) + estimate_role

        seq_len = 3 * K + 6
        tokens = torch.zeros(B, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(B, seq_len, dtype=torch.long, device=device)

        sep, mask = special.get_sep_batch(B), special.get_mask_batch(B)
        tokens[:, 0], tokens[:, 1] = sep, A_emb

        for i in range(K):
            idx = 2 + i * 3
            tokens[:, idx], tokens[:, idx + 1], tokens[:, idx + 2] = sep, b_emb[:, i], x_emb[:, i]
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx], tokens[:, q_idx + 1] = sep, b_q_emb
        tokens[:, q_idx + 2], tokens[:, q_idx + 3] = x_est_emb, mask
        ex_pos[:, q_idx:q_idx + 4] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)

    def test_at_kappa(
        self,
        kappa: float,
        tolerance: float = 0.1,  # Allow kappa in [kappa*(1-tol), kappa*(1+tol)]
    ) -> Dict:
        """
        Test model performance at a specific condition number.

        Args:
            kappa: Target condition number
            tolerance: Relative tolerance for kappa sampling

        Returns:
            Dict with:
            - mse_by_iteration: {0: mse, 1: mse, ...}
            - improvement_factor: mse[0] / mse[final]
            - convergence_iteration: iteration where MSE stabilizes
            - actual_kappas: statistics of sampled condition numbers
        """
        self.model.eval()
        kappa_min = max(1.0, kappa * (1 - tolerance))
        kappa_max = kappa * (1 + tolerance)

        all_mse = {i: [] for i in range(self.config.test_iterations)}
        improvements = []
        sampled_kappas = []

        for _ in range(self.config.test_batches):
            B, K, d = self.config.batch_size, self.config.num_context, self.d

            with torch.no_grad():
                A = sample_spd(B, d, self.device, kappa_min, kappa_max)
                # Track actual kappas
                from data.spd_sampler import compute_condition_number
                sampled_kappas.extend(compute_condition_number(A).cpu().tolist())

                b_all = torch.randn(B, K + 1, d, device=self.device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                mse_history = []

                # Iteration 0: Direct prediction
                tokens, ex_pos, mask_pos = self._build_tokens_standard(A, b_ctx, x_ctx, b_query)
                x_current = self.model(tokens, ex_pos, mask_pos).vector_output
                mse = F.mse_loss(x_current, x_target).item()
                mse_history.append(mse)
                all_mse[0].append(mse)

                # Iterations 1+: Residual refinement
                for i in range(1, self.config.test_iterations):
                    tokens_r, ex_pos_r, mask_pos_r = self._build_tokens_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    residual = self.model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                    x_current = x_current + residual

                    mse = F.mse_loss(x_current, x_target).item()
                    mse_history.append(mse)
                    all_mse[i].append(mse)

                imp = mse_history[0] / mse_history[-1] if mse_history[-1] > 1e-12 else float('inf')
                improvements.append(imp)

        # Find convergence iteration (where MSE stops decreasing significantly)
        mse_means = [np.mean(all_mse[i]) for i in range(self.config.test_iterations)]
        conv_iter = self.config.test_iterations - 1
        for i in range(1, self.config.test_iterations):
            if i > 0 and mse_means[i] >= mse_means[i - 1] * 0.99:
                conv_iter = i
                break

        return {
            "mse_by_iteration": {i: {"mean": np.mean(all_mse[i]), "std": np.std(all_mse[i])}
                                 for i in range(self.config.test_iterations)},
            "improvement_factor": {"mean": np.mean(improvements), "std": np.std(improvements)},
            "convergence_iteration": conv_iter,
            "actual_kappas": {"mean": np.mean(sampled_kappas), "std": np.std(sampled_kappas),
                             "min": np.min(sampled_kappas), "max": np.max(sampled_kappas)},
        }

    def analyze_convergence_curve(
        self,
        kappa: float,
    ) -> Dict:
        """
        Analyze the convergence curve at a specific kappa.

        Returns:
            Dict with:
            - mse_curve: list of MSE values
            - convergence_rate: estimated rate of MSE decrease
            - iterations_to_threshold: iterations to reach MSE < threshold
            - is_converging: whether MSE is consistently decreasing
        """
        result = self.test_at_kappa(kappa)
        mse_curve = [result["mse_by_iteration"][i]["mean"] for i in range(self.config.test_iterations)]

        # Estimate convergence rate (ratio of consecutive MSEs)
        rates = [mse_curve[i] / mse_curve[i - 1] if mse_curve[i - 1] > 1e-12 else 1.0
                 for i in range(1, len(mse_curve))]
        convergence_rate = np.mean(rates) if rates else 1.0

        # Check if MSE is consistently decreasing
        is_converging = all(mse_curve[i] < mse_curve[i - 1] * 1.01 for i in range(1, len(mse_curve)))

        return {
            "mse_curve": mse_curve,
            "convergence_rate": convergence_rate,
            "is_converging": is_converging,
        }

    def find_failure_threshold(self) -> Optional[float]:
        """
        Find the kappa threshold where refinement starts to fail.

        Failure defined as: improvement_factor < 1.0 (refinement makes things worse)

        Returns:
            Kappa threshold, or None if no failure detected
        """
        # Sweep through kappa values to find failure point
        kappa_sweep = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
        last_working = None

        for kappa in kappa_sweep:
            result = self.test_at_kappa(kappa)
            imp = result["improvement_factor"]["mean"]
            print(f"  kappa={kappa}: improvement={imp:.3f}")

            if imp >= 1.0:
                last_working = kappa
            else:
                # Found failure, return midpoint between last working and current
                if last_working is not None:
                    return (last_working + kappa) / 2
                return kappa

        return None  # No failure detected in range


def run_full_kappa_analysis(
    model: ComponentTransformerModel,
    config: KappaAnalysisConfig,
) -> Dict:
    """
    Run complete kappa range analysis.

    Returns:
        Dict with results for each kappa point and failure threshold
    """
    device = torch.device(config.device)
    analyzer = KappaRangeAnalyzer(model, config, device)

    results = {"by_kappa": {}, "failure_threshold": None}

    print("\nTesting at each kappa point:")
    print("-" * 70)

    for kappa in config.kappa_points:
        print(f"\nkappa = {kappa}")
        result = analyzer.test_at_kappa(kappa)
        results["by_kappa"][kappa] = result

        mse_0 = result["mse_by_iteration"][0]["mean"]
        mse_final = result["mse_by_iteration"][config.test_iterations - 1]["mean"]
        imp = result["improvement_factor"]["mean"]
        converged = imp > 1.0

        print(f"  MSE: {mse_0:.6f} -> {mse_final:.6f}")
        print(f"  Improvement: {imp:.2f}x, Converged: {converged}")

    # Find failure threshold
    print("\n" + "-" * 70)
    print("Finding failure threshold...")
    results["failure_threshold"] = analyzer.find_failure_threshold()

    return results


def plot_mse_vs_kappa(
    results: Dict,
    output_path: Path,
) -> None:
    """
    ??? TODO: IMPLEMENT

    Plot MSE as a function of kappa for different iterations.

    Expected plot:
    - X-axis: log(kappa)
    - Y-axis: log(MSE)
    - Multiple lines: iteration 0, 1, 2, ..., final
    - Show where refinement helps vs hurts
    """
    raise NotImplementedError()


def plot_convergence_curves(
    results: Dict,
    output_path: Path,
) -> None:
    """
    ??? TODO: IMPLEMENT

    Plot convergence curves for different kappa ranges.

    Expected plot:
    - X-axis: iteration number
    - Y-axis: MSE (log scale)
    - Multiple lines: different kappa values
    - Annotate convergence rates
    """
    raise NotImplementedError()


def generate_kappa_report(results: Dict, test_iterations: int = 10) -> str:
    """
    Generate detailed report on kappa dependence.

    Returns:
        Formatted string with summary table and findings
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("KAPPA RANGE ANALYSIS REPORT")
    lines.append("=" * 70)

    # Summary table
    lines.append("\n| kappa | MSE_iter0 | MSE_final | improvement_factor | converged |")
    lines.append("|-------|-----------|-----------|-------------------|-----------|")

    for kappa, data in sorted(results["by_kappa"].items(), key=lambda x: float(x[0])):
        mse_0 = data["mse_by_iteration"][0]["mean"]
        mse_final = data["mse_by_iteration"][test_iterations - 1]["mean"]
        imp = data["improvement_factor"]["mean"]
        converged = "Yes" if imp > 1.0 else "No"

        lines.append(f"| {kappa:5} | {mse_0:9.6f} | {mse_final:9.6f} | {imp:17.2f} | {converged:9} |")

    # Failure threshold
    lines.append("\n" + "-" * 70)
    threshold = results.get("failure_threshold")
    if threshold is not None:
        lines.append(f"Failure threshold kappa = {threshold:.1f}")
        lines.append("(Refinement starts making things worse above this kappa)")
    else:
        lines.append("No failure threshold detected in tested range")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze performance vs condition number")
    parser.add_argument("--model_path", type=str, default="results/section2/model.pt", help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results/section3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batches", type=int, default=50)
    parser.add_argument("--test_iterations", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("SECTION 3: KAPPA RANGE ANALYSIS")
    print("=" * 60)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        d = checkpoint.get("d", 4)
        n_embd = checkpoint.get("n_embd", 128)
        n_layer = checkpoint.get("n_layer", 6)
        n_head = checkpoint.get("n_head", 4)
    else:
        state_dict = checkpoint
        # Infer config from state_dict shapes
        n_embd = state_dict["backbone.final_norm.weight"].shape[0]
        n_layer = sum(1 for k in state_dict if k.endswith(".layer_norm1.weight"))
        n_head = 4  # default; not recoverable from state_dict
        d = state_dict["output_head.vector_head.linear.weight"].shape[0]

    model_config = ComponentModelConfig(
        d=d, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run analysis
    config = KappaAnalysisConfig(
        d=model_config.d,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        batch_size=args.batch_size,
        test_batches=args.test_batches,
        test_iterations=args.test_iterations,
        device=args.device,
    )

    results = run_full_kappa_analysis(model, config)

    # Generate and print report
    report = generate_kappa_report(results, config.test_iterations)
    print(report)

    # Save results
    results_path = output_dir / "kappa_analysis_results.json"
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
