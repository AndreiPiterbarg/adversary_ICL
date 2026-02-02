"""
Comprehensive evaluation of a trained Role-Disambiguated Residual model.

Runs after training completes. Loads the saved model and measures:

1. Per-iteration MSE curves across kappa ranges (does refinement help?)
2. Per-iteration MSE curves with MORE iterations (does it keep improving or diverge?)
3. Residual norm analysis (does ||residual|| shrink each step? contraction check)
4. Comparison to classical solvers at each iteration

Usage:
    python experiments/section2_solution/evaluate_refinement.py --device cuda
    python experiments/section2_solution/evaluate_refinement.py --model_path path/to/model.pt --device cuda
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import time

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
class EvalConfig:
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    num_context: int = 5
    batch_size: int = 64
    test_batches: int = 50
    max_iterations: int = 15  # Test more iterations than trained on
    kappa_ranges: List[Tuple[float, float]] = None
    model_path: str = "results/section2/model.pt"
    output_dir: str = "results/section2/evaluation"
    device: str = "cuda"

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]


class TokenBuilder:
    """Minimal token builder for evaluation."""

    def __init__(self, model, d, device):
        self.model = model
        self.d = d
        self.device = device
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),
        }

    def _get_role(self, name):
        return self.model.role_embedding(self._role_indices[name])

    def build_standard(self, A, b_ctx, x_ctx, b_query):
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device
        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
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

    def build_with_estimate(self, A, b_ctx, x_ctx, b_query, x_estimate):
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


def load_model(config: EvalConfig):
    device = torch.device(config.device)
    model_config = ComponentModelConfig(
        d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


# ============================================================
# TEST 1: Per-iteration MSE curves
# ============================================================

def test_mse_curves(model, config: EvalConfig, device):
    """MSE at each refinement iteration, per kappa range."""
    print(f"\n{'='*60}")
    print("TEST 1: PER-ITERATION MSE CURVES")
    print(f"{'='*60}")

    builder = TokenBuilder(model, config.d, device)
    results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        all_mse = {i: [] for i in range(config.max_iterations)}

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d
            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Iteration 0
                tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                x_current = model(tokens, ex_pos, mask_pos).vector_output
                all_mse[0].append(F.mse_loss(x_current, x_target).item())

                # Iterations 1+
                for i in range(1, config.max_iterations):
                    tokens_r, ep, mp = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_current)
                    residual = model(tokens_r, ep, mp).vector_output
                    x_current = x_current + residual
                    all_mse[i].append(F.mse_loss(x_current, x_target).item())

        mse_means = [np.mean(all_mse[i]) for i in range(config.max_iterations)]
        best_iter = int(np.argmin(mse_means))
        results[kappa_key] = {
            "mse_per_iteration": {i: {"mean": np.mean(all_mse[i]), "std": np.std(all_mse[i])}
                                  for i in range(config.max_iterations)},
            "best_iteration": best_iter,
            "mse_iter0": mse_means[0],
            "mse_best": mse_means[best_iter],
            "improvement_at_best": mse_means[0] / mse_means[best_iter] if mse_means[best_iter] > 0 else 0,
        }

        print(f"\nkappa [{kappa_min}, {kappa_max}]:")
        for i in range(min(config.max_iterations, 10)):
            marker = " <-- best" if i == best_iter else ""
            print(f"  iter {i:2d}: MSE = {mse_means[i]:.6f}{marker}")
        if config.max_iterations > 10:
            print(f"  iter {config.max_iterations-1:2d}: MSE = {mse_means[-1]:.6f}")
        print(f"  Best: iter {best_iter}, {results[kappa_key]['improvement_at_best']:.2f}x improvement")

    return results


# ============================================================
# TEST 2: Contraction analysis (does ||correction|| shrink?)
# ============================================================

def test_contraction(model, config: EvalConfig, device):
    """Check if the model's correction magnitude shrinks each step."""
    print(f"\n{'='*60}")
    print("TEST 2: CONTRACTION ANALYSIS")
    print(f"{'='*60}")

    builder = TokenBuilder(model, config.d, device)
    results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        correction_norms = {i: [] for i in range(config.max_iterations)}
        error_norms = {i: [] for i in range(config.max_iterations)}

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d
            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                x_current = model(tokens, ex_pos, mask_pos).vector_output

                error_norms[0].extend(
                    (x_current - x_target).norm(dim=-1).cpu().tolist()
                )

                for i in range(1, config.max_iterations):
                    tokens_r, ep, mp = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_current)
                    residual = model(tokens_r, ep, mp).vector_output
                    correction_norms[i].extend(residual.norm(dim=-1).cpu().tolist())
                    x_current = x_current + residual
                    error_norms[i].extend(
                        (x_current - x_target).norm(dim=-1).cpu().tolist()
                    )

        results[kappa_key] = {
            "correction_norm_per_iter": {
                i: {"mean": np.mean(correction_norms[i]), "std": np.std(correction_norms[i])}
                for i in range(1, config.max_iterations)
            },
            "error_norm_per_iter": {
                i: {"mean": np.mean(error_norms[i]), "std": np.std(error_norms[i])}
                for i in range(config.max_iterations)
            },
        }

        print(f"\nkappa [{kappa_min}, {kappa_max}]:")
        print(f"  {'iter':<6} {'||error||':<14} {'||correction||':<14} {'ratio':<10}")
        print(f"  {'-'*44}")
        prev_err = np.mean(error_norms[0])
        print(f"  {0:<6} {prev_err:.6f}     {'--':<14} {'--':<10}")
        for i in range(1, min(config.max_iterations, 8)):
            err = np.mean(error_norms[i])
            corr = np.mean(correction_norms[i])
            ratio = err / prev_err if prev_err > 0 else float('inf')
            shrinking = "Y" if ratio < 1 else "N"
            print(f"  {i:<6} {err:.6f}     {corr:.6f}     {ratio:.4f} {shrinking}")
            prev_err = err

    return results


# ============================================================
# TEST 3: Comparison to classical solvers
# ============================================================

def test_vs_classical(model, config: EvalConfig, device):
    """Compare model refinement to classical iterative solvers."""
    print(f"\n{'='*60}")
    print("TEST 3: MODEL vs CLASSICAL SOLVERS")
    print(f"{'='*60}")

    builder = TokenBuilder(model, config.d, device)
    results = {}
    n_iters = min(config.max_iterations, 10)

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        model_mse = {i: [] for i in range(n_iters)}
        richardson_mse = {i: [] for i in range(n_iters)}
        jacobi_mse = {i: [] for i in range(n_iters)}

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d
            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Model: iteration 0
                tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                x_model = model(tokens, ex_pos, mask_pos).vector_output
                x_rich = x_model.clone()
                x_jac = x_model.clone()

                model_mse[0].append(F.mse_loss(x_model, x_target).item())
                richardson_mse[0].append(F.mse_loss(x_rich, x_target).item())
                jacobi_mse[0].append(F.mse_loss(x_jac, x_target).item())

                # Precompute for classical solvers
                D_inv = 1.0 / torch.diagonal(A, dim1=-2, dim2=-1)
                # Richardson step size: 2 / (lambda_max + lambda_min)
                # Conservative: use 1 / lambda_max ≈ 1 / trace(A) * d
                rich_alpha = config.d / A.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)

                for i in range(1, n_iters):
                    # Model refinement
                    tokens_r, ep, mp = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_model)
                    x_model = x_model + model(tokens_r, ep, mp).vector_output
                    model_mse[i].append(F.mse_loss(x_model, x_target).item())

                    # Richardson: x += alpha * (b - Ax)
                    r_rich = b_query - torch.bmm(A, x_rich.unsqueeze(-1)).squeeze(-1)
                    x_rich = x_rich + rich_alpha * r_rich
                    richardson_mse[i].append(F.mse_loss(x_rich, x_target).item())

                    # Jacobi: x += D^{-1} * (b - Ax)
                    r_jac = b_query - torch.bmm(A, x_jac.unsqueeze(-1)).squeeze(-1)
                    x_jac = x_jac + D_inv * r_jac
                    jacobi_mse[i].append(F.mse_loss(x_jac, x_target).item())

        results[kappa_key] = {
            "model": {i: np.mean(model_mse[i]) for i in range(n_iters)},
            "richardson": {i: np.mean(richardson_mse[i]) for i in range(n_iters)},
            "jacobi": {i: np.mean(jacobi_mse[i]) for i in range(n_iters)},
        }

        print(f"\nkappa [{kappa_min}, {kappa_max}]:")
        print(f"  {'iter':<6} {'Model':<14} {'Richardson':<14} {'Jacobi':<14}")
        print(f"  {'-'*48}")
        for i in range(n_iters):
            m = np.mean(model_mse[i])
            r = np.mean(richardson_mse[i])
            j = np.mean(jacobi_mse[i])
            best = min(m, r, j)
            m_str = f"{m:.6f}" + (" *" if m == best else "")
            r_str = f"{r:.6f}" + (" *" if r == best else "")
            j_str = f"{j:.6f}" + (" *" if j == best else "")
            print(f"  {i:<6} {m_str:<14} {r_str:<14} {j_str:<14}")

    return results


# ============================================================
# TEST 4: Algorithm identification (what is the model doing?)
# ============================================================

def test_algorithm_id(model, config: EvalConfig, device):
    """Identify what algorithm the model's correction most resembles."""
    print(f"\n{'='*60}")
    print("TEST 4: ALGORITHM IDENTIFICATION")
    print(f"{'='*60}")

    builder = TokenBuilder(model, config.d, device)
    results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        cos_sims = {algo: [] for algo in ["richardson", "newton", "jacobi", "steepest"]}

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d
            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
                b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                # Get initial prediction
                tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                x_current = model(tokens, ex_pos, mask_pos).vector_output

                # Get model's correction at step 1
                tokens_r, ep, mp = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_current)
                delta = model(tokens_r, ep, mp).vector_output

                # Residual
                r = b_query - torch.bmm(A, x_current.unsqueeze(-1)).squeeze(-1)

                # Candidate corrections
                candidates = {
                    "richardson": r,
                    "newton": torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1),
                    "jacobi": r / torch.diagonal(A, dim1=-2, dim2=-1),
                    "steepest": r,  # direction same as richardson, just different scale
                }

                for algo, cand in candidates.items():
                    # Per-sample cosine similarity
                    cos = F.cosine_similarity(delta, cand, dim=-1)
                    cos_sims[algo].extend(cos.cpu().tolist())

        results[kappa_key] = {
            algo: {"mean": np.mean(vals), "std": np.std(vals)}
            for algo, vals in cos_sims.items()
        }

        print(f"\nkappa [{kappa_min}, {kappa_max}] — cosine similarity with model's correction:")
        sorted_algos = sorted(results[kappa_key].items(), key=lambda x: -x[1]["mean"])
        for algo, stats in sorted_algos:
            print(f"  {algo:<15} cos_sim = {stats['mean']:.4f} ± {stats['std']:.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained refinement model")
    parser.add_argument("--model_path", type=str, default="results/section2/model.pt")
    parser.add_argument("--output_dir", type=str, default="results/section2/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_iterations", type=int, default=15)
    parser.add_argument("--test_batches", type=int, default=50)
    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        max_iterations=args.max_iterations,
        test_batches=args.test_batches,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE REFINEMENT EVALUATION")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Max iterations: {config.max_iterations}")

    model, device = load_model(config)

    all_results = {}
    all_results["mse_curves"] = test_mse_curves(model, config, device)
    all_results["contraction"] = test_contraction(model, config, device)
    all_results["vs_classical"] = test_vs_classical(model, config, device)
    all_results["algorithm_id"] = test_algorithm_id(model, config, device)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for kappa_key, data in all_results["mse_curves"].items():
        best = data["best_iteration"]
        imp = data["improvement_at_best"]
        print(f"  kappa {kappa_key}: best at iter {best}, {imp:.2f}x improvement")

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
