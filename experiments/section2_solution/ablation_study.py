"""
Section 2: Ablation Study - Role Embedding vs Dual Objective

This experiment isolates the contributions of the two key components:
1. Role embedding (VEC_SECONDARY for estimates vs OUTPUT for ground truth)
2. Dual training objective (direct + residual loss)

Ablation Matrix:
                        | No Dual Obj | Dual Obj
-------------------------------------------------
No Role Embedding       |  Baseline   |   ???
Role Embedding          |    ???      | Role-Disambiguated Residual

Expected Results to Generate:
- No role embedding, no dual objective: 1600x degradation (naive baseline)
- Role embedding only: ???
- Dual objective only: ???
- Both (Role-Disambiguated Residual): Working solution

Key Questions:
- Is role embedding necessary? Can dual objective alone work?
- Is dual objective necessary? Can role embedding alone work?
- Are they complementary or is one dominant?

Usage:
    python experiments/section2_solution/ablation_study.py --device cuda
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import time
import numpy as np

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
class AblationConfig:
    """Configuration for ablation study."""
    # Ablation settings
    use_role_embedding: bool = True   # Whether to use separate role for estimates
    use_dual_objective: bool = True   # Whether to use residual loss

    # Model
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4

    # Training
    training_steps: int = 50000
    batch_size: int = 64
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    residual_weight: float = 0.5
    noise_scale: float = 0.5

    # Data
    num_context: int = 5
    kappa_min: float = 1.0
    kappa_max: float = 100.0

    # Testing
    test_iterations: int = 5
    test_batches: int = 50
    kappa_ranges: List[Tuple[float, float]] = None

    # Output
    output_dir: str = "results/section2/ablation"
    device: str = "cuda"
    log_every: int = 500

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]

    def get_name(self) -> str:
        """Get descriptive name for this ablation setting."""
        if self.use_role_embedding and self.use_dual_objective:
            return "full_role_disambiguated_residual"
        elif self.use_role_embedding and not self.use_dual_objective:
            return "role_embedding_only"
        elif not self.use_role_embedding and self.use_dual_objective:
            return "dual_objective_only"
        else:
            return "naive_baseline"


class AblationTrainer:
    """
    Trainer that can selectively enable/disable:
    - Role embedding: Use same vs different role for estimates
    - Dual objective: Include vs exclude residual loss
    """

    def __init__(
        self,
        model: ComponentTransformerModel,
        config: AblationConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.d = config.d
        self.current_step = 0

        # Cache role indices
        # KEY ABLATION: If use_role_embedding=False, estimate uses OUTPUT role (same as ground truth)
        #               If use_role_embedding=True, estimate uses VEC_SECONDARY role (different)
        estimate_role_value = Role.VEC_SECONDARY.value if config.use_role_embedding else Role.OUTPUT.value

        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(estimate_role_value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        return self.model.role_embedding(self._role_indices[name])

    def build_tokens_standard(
        self, A: torch.Tensor, b_ctx: torch.Tensor, x_ctx: torch.Tensor, b_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard ICL tokens for direct prediction (iteration 0)."""
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

    def build_tokens_with_estimate(
        self, A: torch.Tensor, b_ctx: torch.Tensor, x_ctx: torch.Tensor,
        b_query: torch.Tensor, x_estimate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokens with current estimate for residual prediction (iterations 1+).

        Format: [SEP, A, SEP, b_0, x_0, ..., SEP, b_query, x_estimate, MASK]

        KEY ABLATION:
        - If use_role_embedding=True: x_estimate uses VEC_SECONDARY role (disambiguation)
        - If use_role_embedding=False: x_estimate uses OUTPUT role (no disambiguation)
        """
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device

        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
        output_role = self._get_role('output')
        estimate_role = self._get_role('estimate')  # Uses OUTPUT or VEC_SECONDARY based on config

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

    def training_step(self, optimizer: Adam) -> Dict[str, float]:
        """
        Training step with selective objectives:
        - Always use direct prediction loss
        - Only use residual loss if use_dual_objective=True
        """
        B, K, d = self.config.batch_size, self.config.num_context, self.d
        device = self.device

        A = sample_spd(B, d, device, self.config.kappa_min, self.config.kappa_max)
        self.current_step += 1

        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        b_ctx = b_all[:, :K]
        x_ctx = x_all[:, :K]
        b_query = b_all[:, K]
        x_target = x_all[:, K]

        total_loss = 0.0
        losses = {}

        # Part 1: Direct prediction loss (always used)
        tokens, ex_pos, mask_pos = self.build_tokens_standard(A, b_ctx, x_ctx, b_query)
        output = self.model(tokens, ex_pos, mask_pos)
        pred_0 = output.vector_output
        loss_direct = F.mse_loss(pred_0, x_target)

        if self.config.use_dual_objective:
            # Mix direct and residual loss
            total_loss = total_loss + (1 - self.config.residual_weight) * loss_direct
        else:
            # Only direct loss
            total_loss = total_loss + loss_direct

        losses["direct"] = loss_direct.item()

        # Part 2: Residual prediction loss (only if use_dual_objective=True)
        if self.config.use_dual_objective:
            with torch.no_grad():
                alpha = torch.rand(B, 1, device=device) ** 0.5
                noise = torch.randn_like(pred_0) * (
                    torch.rand(B, 1, device=device) * self.config.noise_scale
                )
                x_estimate = alpha * pred_0.detach() + (1 - alpha) * x_target + noise
                true_residual = x_target - x_estimate

            tokens_r, ex_pos_r, mask_pos_r = self.build_tokens_with_estimate(
                A, b_ctx, x_ctx, b_query, x_estimate
            )
            pred_residual = self.model(tokens_r, ex_pos_r, mask_pos_r).vector_output

            loss_residual = F.mse_loss(pred_residual, true_residual)
            total_loss = total_loss + self.config.residual_weight * loss_residual
            losses["residual"] = loss_residual.item()
        else:
            losses["residual"] = 0.0

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        optimizer.step()

        losses["total"] = total_loss.item()
        return losses


def run_ablation_experiment(config: AblationConfig) -> Dict:
    """
    Run a single ablation experiment with the given configuration.

    Returns:
        Dict with training history and test results
    """
    device = torch.device(config.device)

    print(f"\nTraining: {config.get_name()}")
    print(f"  use_role_embedding: {config.use_role_embedding}")
    print(f"  use_dual_objective: {config.use_dual_objective}")

    # Create model
    model_config = ComponentModelConfig(
        d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # LR scheduler
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.training_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = AblationTrainer(model, config, device)
    history = []
    start_time = time.time()

    # Training loop
    model.train()
    for step in range(config.training_steps):
        losses = trainer.training_step(optimizer)
        scheduler.step()

        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:5d} | Total: {losses['total']:.6f} | "
                  f"Direct: {losses['direct']:.6f} | Residual: {losses['residual']:.6f} | "
                  f"Time: {elapsed:.1f}s")
            history.append({"step": step, **losses})

    print(f"  Training complete in {time.time() - start_time:.1f}s")

    # Testing with iterative refinement
    model.eval()
    test_results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"

        all_mse = {i: [] for i in range(config.test_iterations)}
        improvements = []

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d

            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx = b_all[:, :K]
                x_ctx = x_all[:, :K]
                b_query, x_target = b_all[:, K], x_all[:, K]

                mse_history = []

                # Iteration 0: Direct prediction
                tokens, ex_pos, mask_pos = trainer.build_tokens_standard(A, b_ctx, x_ctx, b_query)
                x_current = model(tokens, ex_pos, mask_pos).vector_output
                mse = F.mse_loss(x_current, x_target).item()
                mse_history.append(mse)
                all_mse[0].append(mse)

                # Iterations 1+: Residual refinement
                for i in range(1, config.test_iterations):
                    tokens_r, ex_pos_r, mask_pos_r = trainer.build_tokens_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    residual = model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                    x_current = x_current + residual

                    mse = F.mse_loss(x_current, x_target).item()
                    mse_history.append(mse)
                    all_mse[i].append(mse)

                improvements.append(mse_history[0] / mse_history[-1] if mse_history[-1] > 0 else 0)

        mse_summary = {i: {"mean": float(np.mean(m)), "std": float(np.std(m))} for i, m in all_mse.items()}
        improved_frac = sum(1 for imp in improvements if imp > 1) / len(improvements)

        test_results[kappa_key] = {
            "mse_by_iteration": mse_summary,
            "improvement_ratio": {"mean": float(np.mean(improvements)), "std": float(np.std(improvements))},
            "improved_fraction": improved_frac,
        }

        print(f"  kappa [{kappa_min}, {kappa_max}]: "
              f"MSE {mse_summary[0]['mean']:.6f} -> {mse_summary[config.test_iterations-1]['mean']:.6f}, "
              f"Improved: {improved_frac*100:.1f}%")

    return {
        "config": asdict(config),
        "training_history": history,
        "test_results": test_results,
    }


def run_full_ablation_study(base_output_dir: str, device: str = "cuda", training_steps: int = 50000) -> Dict:
    """
    Run the complete 2x2 ablation study:
    1. Neither (naive baseline)
    2. Role embedding only
    3. Dual objective only
    4. Both (full Role-Disambiguated Residual)

    Args:
        base_output_dir: Directory to save results
        device: CUDA or CPU
        training_steps: Number of training steps per configuration

    Returns:
        Dict mapping ablation names to results
    """
    output_dir = Path(base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_configs = [
        AblationConfig(use_role_embedding=False, use_dual_objective=False, device=device, training_steps=training_steps),
        AblationConfig(use_role_embedding=True, use_dual_objective=False, device=device, training_steps=training_steps),
        AblationConfig(use_role_embedding=False, use_dual_objective=True, device=device, training_steps=training_steps),
        AblationConfig(use_role_embedding=True, use_dual_objective=True, device=device, training_steps=training_steps),
    ]

    results = {}
    for config in ablation_configs:
        name = config.get_name()
        print(f"\n{'='*60}")
        print(f"ABLATION: {name}")
        print(f"  Role embedding: {config.use_role_embedding}")
        print(f"  Dual objective: {config.use_dual_objective}")
        print(f"{'='*60}")

        config.output_dir = str(output_dir / name)
        results[name] = run_ablation_experiment(config)

        # Save individual result
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(results[name], f, indent=2, default=str)

    # Save combined results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def analyze_ablation_results(results: Dict) -> Dict:
    """
    Analyze ablation results to determine:
    - Contribution of role embedding
    - Contribution of dual objective
    - Whether they are additive or synergistic
    """
    summary = {}

    for name, data in results.items():
        # Average improvement fraction across all kappa ranges
        test_results = data["test_results"]
        avg_improved_frac = np.mean([r["improved_fraction"] for r in test_results.values()])
        avg_improvement_ratio = np.mean([r["improvement_ratio"]["mean"] for r in test_results.values()])

        summary[name] = {
            "avg_improved_fraction": float(avg_improved_frac),
            "avg_improvement_ratio": float(avg_improvement_ratio),
            "use_role_embedding": data["config"]["use_role_embedding"],
            "use_dual_objective": data["config"]["use_dual_objective"],
        }

    # Compute contributions
    baseline = summary.get("naive_baseline", {}).get("avg_improved_fraction", 0)
    role_only = summary.get("role_embedding_only", {}).get("avg_improved_fraction", 0)
    dual_only = summary.get("dual_objective_only", {}).get("avg_improved_fraction", 0)
    full = summary.get("full_role_disambiguated_residual", {}).get("avg_improved_fraction", 0)

    analysis = {
        "summary": summary,
        "contributions": {
            "role_embedding_contribution": role_only - baseline,
            "dual_objective_contribution": dual_only - baseline,
            "combined_contribution": full - baseline,
            "synergy": (full - baseline) - (role_only - baseline) - (dual_only - baseline),
        }
    }

    return analysis


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation Study: Role Embedding vs Dual Objective")
    parser.add_argument("--output_dir", type=str, default="results/section2/ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--training_steps", type=int, default=50000)
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION STUDY: Role Embedding vs Dual Objective")
    print("=" * 60)

    results = run_full_ablation_study(args.output_dir, args.device, args.training_steps)

    # Analyze results
    analysis = analyze_ablation_results(results)

    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    print("\nImprovement Fractions (% of samples that improved):")
    for name, data in analysis["summary"].items():
        print(f"  {name}: {data['avg_improved_fraction']*100:.1f}%")

    print("\nContributions:")
    contrib = analysis["contributions"]
    print(f"  Role embedding alone: {contrib['role_embedding_contribution']*100:+.1f}%")
    print(f"  Dual objective alone: {contrib['dual_objective_contribution']*100:+.1f}%")
    print(f"  Combined: {contrib['combined_contribution']*100:+.1f}%")
    print(f"  Synergy: {contrib['synergy']*100:+.1f}%")

    # Save analysis
    output_dir = Path(args.output_dir)
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
