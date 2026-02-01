"""
Section 4: Bonus - Classical Iterative Solver Comparison

??? TODO: IMPLEMENT THIS

Compare the learned refinement against classical iterative solvers:
- Jacobi Iteration
- Gauss-Seidel
- Successive Over-Relaxation (SOR)
- Conjugate Gradient
- Gradient Descent with optimal step size

Key Comparison Metrics:
1. Iterations to reach MSE = epsilon
2. Convergence rate (MSE reduction per iteration)
3. Computational cost per iteration
4. Robustness across condition numbers

Expected Results:
- If learned method is competitive with CG: Strong result!
- If faster than Jacobi but slower than CG: Still interesting
- This tells us what level of algorithm the model figured out

Experiment Setup:
- Same test problems as model evaluation
- Same initial conditions (model's iteration 0 prediction)
- Compare convergence curves

Usage:
    python experiments/section4_bonus/classical_solvers.py --model_path path/to/model.pt
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import json

import sys
_src_dir = Path(__file__).parent.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data.spd_sampler import sample_spd


@dataclass
class SolverComparisonConfig:
    """Configuration for solver comparison."""
    d: int = 4
    batch_size: int = 64
    num_problems: int = 500
    max_iterations: int = 50
    target_mse: float = 1e-6  # Target MSE for convergence

    kappa_ranges: List[Tuple[float, float]] = None
    output_dir: str = "results/section4"
    device: str = "cuda"

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]


class ClassicalSolvers:
    """
    ??? TODO: IMPLEMENT

    Implementations of classical iterative solvers for comparison.
    All solvers should have the same interface: (A, b, x0, max_iter) -> x_history
    """

    @staticmethod
    def jacobi_iteration(
        A: torch.Tensor,      # (B, d, d)
        b: torch.Tensor,      # (B, d)
        x0: torch.Tensor,     # (B, d) initial guess
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Jacobi iteration: x_{k+1} = D^{-1} @ (b - (L + U) @ x_k)
        where A = D + L + U (diagonal, lower, upper)

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()

    @staticmethod
    def gauss_seidel(
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Gauss-Seidel iteration: x_{k+1} = (D + L)^{-1} @ (b - U @ x_k)

        Note: This requires sequential updates, harder to batch efficiently.

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()

    @staticmethod
    def sor_iteration(
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        omega: float = 1.5,   # Relaxation parameter
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Successive Over-Relaxation (SOR) iteration.
        omega = 1 gives Gauss-Seidel.

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()

    @staticmethod
    def conjugate_gradient(
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Conjugate Gradient method for SPD matrices.
        This is the gold standard for comparison.

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()

    @staticmethod
    def gradient_descent_optimal(
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Gradient descent with optimal step size (line search).
        alpha = r.T @ r / (r.T @ A @ r)

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()

    @staticmethod
    def richardson_iteration(
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor,
        alpha: float = 0.1,   # Step size (should be < 2/lambda_max for convergence)
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
        """
        ??? TODO: IMPLEMENT

        Richardson iteration: x_{k+1} = x_k + alpha * (b - A @ x_k)
        Simple but slow; good baseline.

        Returns:
            List of x tensors at each iteration
        """
        raise NotImplementedError()


class SolverComparer:
    """
    ??? TODO: IMPLEMENT

    Compares learned refinement against classical solvers.
    """

    def __init__(self, config: SolverComparisonConfig, device: torch.device):
        self.config = config
        self.device = device
        raise NotImplementedError("SolverComparer not yet implemented")

    def run_solver_comparison(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x_star: torch.Tensor,  # True solution
        x0: torch.Tensor,       # Initial guess (model's iteration 0)
        model_trajectory: List[torch.Tensor],  # Model's refinement history
    ) -> Dict[str, Dict]:
        """
        ??? TODO: IMPLEMENT

        Compare all solvers on the same problem.

        Returns:
            Dict mapping solver name to:
            - mse_history: MSE at each iteration
            - iterations_to_target: iterations to reach target_mse (or -1)
            - final_mse: MSE at max_iterations
            - converged: whether reached target
        """
        raise NotImplementedError()

    def aggregate_results(
        self,
        all_results: List[Dict[str, Dict]],
    ) -> Dict[str, Dict]:
        """
        ??? TODO: IMPLEMENT

        Aggregate results across multiple problems.

        Returns:
            Dict with statistics for each solver
        """
        raise NotImplementedError()


def run_comparison_experiment(
    model_path: str,
    config: SolverComparisonConfig,
) -> Dict:
    """
    ??? TODO: IMPLEMENT

    Full comparison experiment.

    1. Load trained model
    2. Generate test problems
    3. Get model's refinement trajectory
    4. Run all classical solvers
    5. Compare convergence

    Returns:
        Dict with comprehensive comparison results
    """
    raise NotImplementedError("Comparison experiment not yet implemented")


def plot_convergence_comparison(
    results: Dict,
    output_path: Path,
) -> None:
    """
    ??? TODO: IMPLEMENT

    Plot convergence curves comparing all solvers.

    Expected plot:
    - X-axis: iteration
    - Y-axis: MSE (log scale)
    - Lines: learned model, Jacobi, GS, CG, etc.
    - Separate plots for each kappa range
    """
    raise NotImplementedError()


def generate_comparison_report(results: Dict) -> str:
    """
    ??? TODO: IMPLEMENT

    Generate report comparing learned vs classical solvers.

    Should answer:
    - Is learned method competitive with classical solvers?
    - Is it faster than Jacobi? Slower than CG?
    - What does this tell us about the learned algorithm?
    """
    raise NotImplementedError()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare with classical solvers")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results/section4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("SECTION 4: CLASSICAL SOLVER COMPARISON")
    print("=" * 60)
    print("\n??? THIS COMPARISON IS NOT YET IMPLEMENTED\n")
    print("This experiment will compare the learned refinement against:")
    print("  1. Jacobi Iteration")
    print("  2. Gauss-Seidel")
    print("  3. SOR (Successive Over-Relaxation)")
    print("  4. Conjugate Gradient")
    print("  5. Gradient Descent (optimal step)")
    print("  6. Richardson Iteration")
    print("\nMetrics:")
    print("  - Iterations to reach MSE = epsilon")
    print("  - Convergence rate")
    print("  - Robustness across kappa ranges")
    print("\nPlease implement ClassicalSolvers and SolverComparer")


if __name__ == "__main__":
    main()
