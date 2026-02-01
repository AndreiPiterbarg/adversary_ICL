"""
Section 4: Bonus - Extrapolation and Generalization Tests

Test the model's ability to extrapolate beyond training distribution:
1. Higher condition numbers than seen during training (kappa extrapolation)
2. More refinement iterations than used during training (iteration extrapolation)
3. Failure mode analysis at high kappa

Key Questions:
1. Does refinement still help beyond training kappa range?
2. How many iterations before performance degrades?
3. Is failure due to wrong direction or wrong magnitude?

Usage:
    python experiments/section4_bonus/extrapolation.py --model_path path/to/model.pt
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
from data.spd_sampler import sample_spd


@dataclass
class ExtrapolationConfig:
    """Configuration for extrapolation tests."""
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4

    num_context: int = 5
    batch_size: int = 64
    test_batches: int = 50

    # Training distribution (for reference)
    train_kappa_max: float = 100.0

    # Extrapolation tests
    extrap_kappa_values: List[float] = None  # Test beyond training
    extrap_iterations: List[int] = None       # Test more iterations

    output_dir: str = "results/section4"
    device: str = "cuda"

    def __post_init__(self):
        if self.extrap_kappa_values is None:
            # Include in-distribution and out-of-distribution
            self.extrap_kappa_values = [10, 50, 100, 150, 200, 300, 500, 1000]
        if self.extrap_iterations is None:
            self.extrap_iterations = [5, 10, 15, 20, 30, 50]


class ExtrapolationTester:
    """Tests model's extrapolation capabilities."""

    def __init__(
        self,
        model: ComponentTransformerModel,
        config: ExtrapolationConfig,
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

        matrix_role, bias_role = self._get_role('matrix'), self._get_role('bias')
        output_role = self._get_role('output')

        A_emb = embedders.matrix(A) + matrix_role
        b_flat = b_ctx.reshape(B * K, d)
        x_flat = x_ctx.reshape(B * K, d)
        n_embd = embedders.vector(b_flat).shape[-1]

        b_emb = embedders.vector(b_flat).reshape(B, K, n_embd) + bias_role
        x_emb = embedders.vector(x_flat).reshape(B, K, n_embd) + output_role
        b_q_emb = embedders.vector(b_query) + bias_role

        seq_len = 3 * K + 5
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
        tokens[:, q_idx + 2] = mask
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

    def test_kappa_extrapolation(
        self,
        kappa: float,
        num_iterations: int = 5,
    ) -> Dict:
        """
        Test model at a specific condition number (potentially beyond training).

        Returns:
            Dict with:
            - mse_by_iteration: MSE at each iteration
            - improved_fraction: fraction of samples that improved
            - is_extrapolation: whether kappa > train_kappa_max
        """
        B, K, d = self.config.batch_size, self.config.num_context, self.d
        device = self.device
        self.model.eval()

        all_mse = {i: [] for i in range(num_iterations)}
        improvements = []

        with torch.no_grad():
            for _ in range(self.config.test_batches):
                # Sample with exact kappa (use narrow range)
                A = sample_spd(B, d, device, kappa * 0.95, kappa * 1.05)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Iteration 0: Direct prediction
                tokens, ex_pos, mask_pos = self._build_tokens_standard(A, b_ctx, x_ctx, b_query)
                x_current = self.model(tokens, ex_pos, mask_pos).vector_output
                mse_0 = F.mse_loss(x_current, x_target, reduction='none').mean(dim=-1)
                all_mse[0].append(mse_0.mean().item())

                # Iterations 1+: Residual refinement
                for i in range(1, num_iterations):
                    tokens_r, ex_pos_r, mask_pos_r = self._build_tokens_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    residual = self.model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                    x_current = x_current + residual
                    mse_i = F.mse_loss(x_current, x_target, reduction='none').mean(dim=-1)
                    all_mse[i].append(mse_i.mean().item())

                # Track per-batch improvement
                mse_final = all_mse[num_iterations - 1][-1]
                mse_init = all_mse[0][-1]
                improvements.append(mse_init / mse_final if mse_final > 1e-12 else 1.0)

        mse_summary = {i: np.mean(m) for i, m in all_mse.items()}
        improved_frac = sum(1 for imp in improvements if imp > 1) / len(improvements)
        improvement_ratio = np.mean(improvements)

        return {
            "kappa": kappa,
            "mse_by_iteration": mse_summary,
            "improved_fraction": improved_frac,
            "improvement_ratio": improvement_ratio,
            "is_extrapolation": kappa > self.config.train_kappa_max,
        }

    def test_iteration_extrapolation(
        self,
        kappa_min: float,
        kappa_max: float,
        num_iterations: int,
    ) -> Dict:
        """
        Test model with more iterations than used during training.

        Key question: Does performance plateau, continue improving, or degrade?

        Returns:
            Dict with:
            - mse_curve: MSE over all iterations
            - optimal_iterations: iteration count with lowest MSE
            - degradation_start: iteration where MSE starts increasing (if any)
        """
        B, K, d = self.config.batch_size, self.config.num_context, self.d
        device = self.device
        self.model.eval()

        all_mse = {i: [] for i in range(num_iterations)}

        with torch.no_grad():
            for _ in range(self.config.test_batches):
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Iteration 0
                tokens, ex_pos, mask_pos = self._build_tokens_standard(A, b_ctx, x_ctx, b_query)
                x_current = self.model(tokens, ex_pos, mask_pos).vector_output
                all_mse[0].append(F.mse_loss(x_current, x_target).item())

                # Iterations 1+
                for i in range(1, num_iterations):
                    tokens_r, ex_pos_r, mask_pos_r = self._build_tokens_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    residual = self.model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                    x_current = x_current + residual
                    all_mse[i].append(F.mse_loss(x_current, x_target).item())

        mse_curve = [np.mean(all_mse[i]) for i in range(num_iterations)]
        optimal_iterations = int(np.argmin(mse_curve))

        # Find degradation start: first iteration where MSE increases consistently
        degradation_start = None
        for i in range(1, num_iterations):
            if mse_curve[i] > mse_curve[i - 1]:
                # Check if it keeps increasing
                if i + 1 < num_iterations and mse_curve[i + 1] >= mse_curve[i]:
                    degradation_start = i
                    break

        return {
            "kappa_range": (kappa_min, kappa_max),
            "mse_curve": mse_curve,
            "optimal_iterations": optimal_iterations,
            "degradation_start": degradation_start,
        }

    def analyze_failure_modes(
        self,
        kappa: float,
        num_iterations: int,
    ) -> Dict:
        """
        Analyze how/why refinement fails at high kappa.

        Computes:
        - Cosine similarity between model correction and true correction
        - Magnitude ratio (model correction norm / true correction norm)
        """
        B, K, d = self.config.batch_size, self.config.num_context, self.d
        device = self.device
        self.model.eval()

        cosine_sims = {i: [] for i in range(1, num_iterations)}
        magnitude_ratios = {i: [] for i in range(1, num_iterations)}

        with torch.no_grad():
            for _ in range(self.config.test_batches):
                A = sample_spd(B, d, device, kappa * 0.95, kappa * 1.05)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Iteration 0
                tokens, ex_pos, mask_pos = self._build_tokens_standard(A, b_ctx, x_ctx, b_query)
                x_current = self.model(tokens, ex_pos, mask_pos).vector_output

                # Analyze each refinement iteration
                for i in range(1, num_iterations):
                    true_correction = x_target - x_current

                    tokens_r, ex_pos_r, mask_pos_r = self._build_tokens_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    model_correction = self.model(tokens_r, ex_pos_r, mask_pos_r).vector_output

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(model_correction, true_correction, dim=-1)
                    cosine_sims[i].extend(cos_sim.cpu().tolist())

                    # Magnitude ratio
                    model_norm = model_correction.norm(dim=-1)
                    true_norm = true_correction.norm(dim=-1)
                    ratio = model_norm / (true_norm + 1e-8)
                    magnitude_ratios[i].extend(ratio.cpu().tolist())

                    x_current = x_current + model_correction

        return {
            "kappa": kappa,
            "cosine_similarity": {i: np.mean(v) for i, v in cosine_sims.items()},
            "magnitude_ratio": {i: np.mean(v) for i, v in magnitude_ratios.items()},
            "direction_correct": np.mean(cosine_sims[1]) > 0.5,
            "magnitude_correct": 0.5 < np.mean(magnitude_ratios[1]) < 2.0,
        }


def run_extrapolation_experiments(
    model_path: str,
    config: ExtrapolationConfig,
) -> Dict:
    """
    Run full extrapolation experiment suite.

    Tests:
    1. Kappa extrapolation: test at kappa values beyond training
    2. Iteration extrapolation: test with more iterations
    3. Failure mode analysis at high kappa
    """
    device = torch.device(config.device)

    # Load model
    model_config = ComponentModelConfig(
        d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    tester = ExtrapolationTester(model, config, device)
    results = {"config": config.__dict__}

    # 1. Kappa extrapolation
    print("\n" + "=" * 60)
    print("KAPPA EXTRAPOLATION")
    print("=" * 60)
    kappa_results = {}
    for kappa in config.extrap_kappa_values:
        res = tester.test_kappa_extrapolation(kappa, num_iterations=5)
        kappa_results[kappa] = res
        status = "EXTRAP" if res["is_extrapolation"] else "IN-DIST"
        print(f"kappa={kappa:4.0f} [{status}]: "
              f"MSE {res['mse_by_iteration'][0]:.6f} -> {res['mse_by_iteration'][4]:.6f}, "
              f"improvement={res['improvement_ratio']:.2f}x")
    results["kappa_extrapolation"] = kappa_results

    # 2. Iteration extrapolation (in-distribution kappa)
    print("\n" + "=" * 60)
    print("ITERATION EXTRAPOLATION")
    print("=" * 60)
    iteration_results = {}
    for max_iter in config.extrap_iterations:
        res = tester.test_iteration_extrapolation(1.0, config.train_kappa_max, max_iter)
        iteration_results[max_iter] = res
        print(f"iterations={max_iter:2d}: optimal={res['optimal_iterations']}, "
              f"degradation_start={res['degradation_start']}")
    results["iteration_extrapolation"] = iteration_results

    # 3. Failure mode analysis at high kappa
    print("\n" + "=" * 60)
    print("FAILURE MODE ANALYSIS")
    print("=" * 60)
    failure_results = {}
    for kappa in [100, 300, 500, 1000]:
        res = tester.analyze_failure_modes(kappa, num_iterations=5)
        failure_results[kappa] = res
        print(f"kappa={kappa:4.0f}: cos_sim={res['cosine_similarity'][1]:.3f}, "
              f"mag_ratio={res['magnitude_ratio'][1]:.3f}, "
              f"dir_ok={res['direction_correct']}, mag_ok={res['magnitude_correct']}")
    results["failure_modes"] = failure_results

    return results


def plot_kappa_extrapolation(
    results: Dict,
    output_path: Path,
    train_kappa_max: float = 100.0,
) -> None:
    """
    ??? TODO: IMPLEMENT

    Plot performance as kappa goes beyond training distribution.

    Expected plot:
    - X-axis: kappa (log scale)
    - Y-axis: improvement factor or MSE
    - Vertical line at train_kappa_max
    - Shaded region for extrapolation zone
    """
    raise NotImplementedError()


def plot_iteration_extrapolation(
    results: Dict,
    output_path: Path,
) -> None:
    """
    ??? TODO: IMPLEMENT

    Plot performance as iteration count increases.

    Expected plot:
    - X-axis: iteration number
    - Y-axis: MSE (log scale)
    - Multiple lines for different kappa ranges
    - Mark optimal iteration count
    """
    raise NotImplementedError()


def generate_extrapolation_report(results: Dict) -> str:
    """
    ??? TODO: IMPLEMENT

    Generate report on extrapolation capabilities.

    Key findings to report:
    - Kappa threshold for extrapolation failure
    - Optimal iteration count
    - Failure mode analysis
    """
    raise NotImplementedError()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test extrapolation capabilities")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="results/section4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batches", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("SECTION 4: EXTRAPOLATION TESTS")
    print("=" * 60)

    config = ExtrapolationConfig(
        batch_size=args.batch_size,
        test_batches=args.test_batches,
        output_dir=args.output_dir,
        device=args.device,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_extrapolation_experiments(args.model_path, config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find kappa threshold where refinement fails
    kappa_res = results["kappa_extrapolation"]
    failure_threshold = None
    for kappa in sorted(kappa_res.keys()):
        if kappa_res[kappa]["improvement_ratio"] < 1.0:
            failure_threshold = kappa
            break
    print(f"Kappa failure threshold: {failure_threshold}")

    # Find optimal iteration count
    iter_res = results["iteration_extrapolation"]
    max_iters = max(iter_res.keys())
    optimal = iter_res[max_iters]["optimal_iterations"]
    degrade = iter_res[max_iters]["degradation_start"]
    print(f"Optimal iterations: {optimal}")
    print(f"Degradation starts at: {degrade}")

    # Save results
    with open(output_dir / "extrapolation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'extrapolation_results.json'}")


if __name__ == "__main__":
    main()
