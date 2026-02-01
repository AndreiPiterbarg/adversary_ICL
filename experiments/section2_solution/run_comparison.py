"""
Section 2: Run Comparison - Baseline vs Iterative Supervision vs Role-Disambiguated Residual

Comprehensive comparison of all approaches:
- Baseline: Standard ICL with naive refinement
- Iterative Supervision: Multi-step supervision (add predictions to context)
- Role-Disambiguated Residual: Residual prediction with role disambiguation

Usage:
    python experiments/section2_solution/run_comparison.py --device cuda
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
class Config:
    """Configuration for comparison experiment."""
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

    # Iterative Supervision config
    b_train_iterations: int = 3
    b_iteration_weights: Tuple[float, ...] = (0.2, 0.3, 0.5)

    # Role-Disambiguated Residual config
    c_residual_weight: float = 0.5
    c_noise_scale: float = 0.5

    # Data
    num_context: int = 5
    kappa_min: float = 1.0
    kappa_max: float = 100.0

    # Testing
    test_iterations: int = 5
    test_batches: int = 50
    kappa_ranges: List[Tuple[float, float]] = None

    # Output
    output_dir: str = "results/section2/comparison"
    device: str = "cuda"
    log_every: int = 1000

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]

    def forwards_per_step(self, approach: str) -> int:
        """Number of forward passes per training step for each approach."""
        if approach == "baseline":
            return 1
        elif approach == "iterative_supervision":
            return self.b_train_iterations
        elif approach == "role_disambiguated_residual":
            return 2  # direct + residual
        raise ValueError(f"Unknown approach: {approach}")

    def steps_for(self, approach: str) -> int:
        """Training steps for an approach, equalized by total forward passes.

        The most expensive approach sets the forward-pass budget
        (its forwards_per_step * training_steps). Every other approach
        gets enough steps so its total forwards match that budget.
        """
        max_fps = max(
            self.forwards_per_step(a)
            for a in ["baseline", "iterative_supervision", "role_disambiguated_residual"]
        )
        budget = max_fps * self.training_steps
        return budget // self.forwards_per_step(approach)


def create_model(config: Config, device: torch.device) -> ComponentTransformerModel:
    model_config = ComponentModelConfig(
        d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    return ComponentTransformerModel(model_config).to(device)


class TokenBuilder:
    """Shared token building utilities."""

    def __init__(self, model: ComponentTransformerModel, d: int, device: torch.device):
        self.model = model
        self.d = d
        self.device = device
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        return self.model.role_embedding(self._role_indices[name])

    def build_standard(self, A, b_ctx, x_ctx, b_query):
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device
        embedders, special = self.model.embedders, self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
        output_role = self._get_role('output')

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
            tokens[:, idx:idx + 3] = torch.stack([sep, b_emb[:, i], x_emb[:, i]], dim=1)
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx:q_idx + 3] = torch.stack([sep, b_q_emb, mask], dim=1)
        ex_pos[:, q_idx:q_idx + 3] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)

    def build_with_estimate(self, A, b_ctx, x_ctx, b_query, x_estimate):
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device
        embedders, special = self.model.embedders, self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
        output_role = self._get_role('output')
        estimate_role = self._get_role('estimate')

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
            tokens[:, idx:idx + 3] = torch.stack([sep, b_emb[:, i], x_emb[:, i]], dim=1)
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx:q_idx + 4] = torch.stack([sep, b_q_emb, x_est_emb, mask], dim=1)
        ex_pos[:, q_idx:q_idx + 4] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)


def train_baseline(config: Config, device: torch.device) -> ComponentTransformerModel:
    """Train standard ICL baseline (steps scaled to match total forward passes)."""
    total_steps = config.steps_for("baseline")

    print(f"\n{'='*60}")
    print(f"TRAINING: BASELINE ({total_steps} steps, {config.forwards_per_step('baseline')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    builder = TokenBuilder(model, config.d, device)
    model.train()
    start = time.time()

    for step in range(total_steps):
        B, K, d = config.batch_size, config.num_context, config.d
        A = sample_spd(B, d, device, config.kappa_min, config.kappa_max)
        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        tokens, ex_pos, mask_pos = builder.build_standard(A, b_all[:, :K], x_all[:, :K], b_all[:, K])
        pred = model(tokens, ex_pos, mask_pos).vector_output
        loss = F.mse_loss(pred, x_all[:, K])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % (config.log_every * 2) == 0:
            print(f"Step {step:5d}/{total_steps} | Loss: {loss.item():.6f}")

    print(f"Baseline training complete in {time.time()-start:.1f}s")
    return model


def train_iterative_supervision(config: Config, device: torch.device) -> ComponentTransformerModel:
    """Train Iterative Supervision model."""
    total_steps = config.steps_for("iterative_supervision")

    print(f"\n{'='*60}")
    print(f"TRAINING: ITERATIVE SUPERVISION ({total_steps} steps, {config.forwards_per_step('iterative_supervision')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    builder = TokenBuilder(model, config.d, device)
    weights = list(config.b_iteration_weights)
    weights = [w / sum(weights) for w in weights]

    model.train()
    start = time.time()

    for step in range(total_steps):
        B, K, d = config.batch_size, config.num_context, config.d

        A = sample_spd(B, d, device, config.kappa_min, config.kappa_max)
        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        b_ctx, x_ctx = b_all[:, :K].clone(), x_all[:, :K].clone()
        b_query, x_target = b_all[:, K], x_all[:, K]

        total_loss = 0.0
        for i in range(config.b_train_iterations):
            tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
            pred = model(tokens, ex_pos, mask_pos).vector_output
            total_loss = total_loss + weights[i] * F.mse_loss(pred, x_target)

            with torch.no_grad():
                b_ctx = torch.cat([b_ctx, b_query.unsqueeze(1)], dim=1)
                x_ctx = torch.cat([x_ctx, pred.unsqueeze(1)], dim=1)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % config.log_every == 0:
            print(f"Step {step:5d} | Loss: {total_loss.item():.6f}")

    print(f"Iterative Supervision training complete in {time.time()-start:.1f}s")
    return model


def train_role_disambiguated_residual(config: Config, device: torch.device) -> ComponentTransformerModel:
    """Train Role-Disambiguated Residual model."""
    total_steps = config.steps_for("role_disambiguated_residual")

    print(f"\n{'='*60}")
    print(f"TRAINING: ROLE-DISAMBIGUATED RESIDUAL ({total_steps} steps, {config.forwards_per_step('role_disambiguated_residual')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    builder = TokenBuilder(model, config.d, device)
    model.train()
    start = time.time()

    for step in range(total_steps):
        B, K, d = config.batch_size, config.num_context, config.d

        A = sample_spd(B, d, device, config.kappa_min, config.kappa_max)
        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        b_ctx, x_ctx = b_all[:, :K], x_all[:, :K]
        b_query, x_target = b_all[:, K], x_all[:, K]

        # Direct prediction
        tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
        pred_direct = model(tokens, ex_pos, mask_pos).vector_output
        loss_direct = F.mse_loss(pred_direct, x_target)

        # Residual prediction loss
        with torch.no_grad():
            alpha = torch.rand(B, 1, device=device) ** 0.5
            noise = torch.randn_like(pred_direct) * (
                torch.rand(B, 1, device=device) * config.c_noise_scale
            )
            x_estimate = alpha * pred_direct.detach() + (1 - alpha) * x_target + noise
            true_residual = x_target - x_estimate

        tokens_r, ex_pos_r, mask_pos_r = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_estimate)
        pred_residual = model(tokens_r, ex_pos_r, mask_pos_r).vector_output
        loss_residual = F.mse_loss(pred_residual, true_residual)

        w = config.c_residual_weight
        total_loss = (1 - w) * loss_direct + w * loss_residual

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % config.log_every == 0:
            print(f"Step {step:5d} | Total: {total_loss.item():.6f} | Direct: {loss_direct.item():.6f} | Residual: {loss_residual.item():.6f}")

    print(f"Role-Disambiguated Residual training complete in {time.time()-start:.1f}s")
    return model


def test_model(model: ComponentTransformerModel, config: Config, device: torch.device, approach: str) -> Dict:
    """Test a model with iterative refinement."""
    model.eval()
    builder = TokenBuilder(model, config.d, device)
    results = {}

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

                b_ctx, x_ctx = b_all[:, :K].clone(), x_all[:, :K].clone()
                b_query, x_target = b_all[:, K], x_all[:, K]
                mse_history = []

                if approach in ["baseline", "iterative_supervision"]:
                    # Test by adding predictions to context
                    for i in range(config.test_iterations):
                        tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                        pred = model(tokens, ex_pos, mask_pos).vector_output
                        mse = F.mse_loss(pred, x_target).item()
                        mse_history.append(mse)
                        all_mse[i].append(mse)
                        b_ctx = torch.cat([b_ctx, b_query.unsqueeze(1)], dim=1)
                        x_ctx = torch.cat([x_ctx, pred.unsqueeze(1)], dim=1)

                elif approach == "role_disambiguated_residual":
                    # Test with residual refinement
                    tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                    x_current = model(tokens, ex_pos, mask_pos).vector_output
                    mse = F.mse_loss(x_current, x_target).item()
                    mse_history.append(mse)
                    all_mse[0].append(mse)

                    for i in range(1, config.test_iterations):
                        tokens_r, ex_pos_r, mask_pos_r = builder.build_with_estimate(A, b_ctx, x_ctx, b_query, x_current)
                        residual = model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                        x_current = x_current + residual
                        mse = F.mse_loss(x_current, x_target).item()
                        mse_history.append(mse)
                        all_mse[i].append(mse)

                improvements.append(mse_history[0] / mse_history[-1] if mse_history[-1] > 0 else 0)

        mse_summary = {i: {"mean": float(np.mean(m)), "std": float(np.std(m))} for i, m in all_mse.items()}
        improved_frac = sum(1 for imp in improvements if imp > 1) / len(improvements)

        results[kappa_key] = {
            "mse_by_iteration": mse_summary,
            "improvement_ratio": {"mean": float(np.mean(improvements)), "std": float(np.std(improvements))},
            "improved_fraction": improved_frac,
        }

    return results


def print_results_table(all_results: Dict[str, Dict]):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Approach':<15} {'Kappa':<12} {'MSE_0':<12} {'MSE_final':<12} {'Improvement':<12} {'% Improved'}")
    print("-" * 80)

    for approach, results in all_results.items():
        for kappa_key, stats in results.items():
            mse_0 = stats["mse_by_iteration"][0]["mean"]
            mse_final = stats["mse_by_iteration"][max(stats["mse_by_iteration"].keys())]["mean"]
            imp = stats["improvement_ratio"]["mean"]
            frac = stats["improved_fraction"] * 100
            print(f"{approach:<15} {kappa_key:<12} {mse_0:<12.6f} {mse_final:<12.6f} {imp:<12.2f}x {frac:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare all approaches")
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="results/section2/comparison")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config(training_steps=args.training_steps, output_dir=args.output_dir, device=args.device)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    all_results = {}

    # Train all models
    models["baseline"] = train_baseline(config, device)
    torch.save(models["baseline"].state_dict(), output_dir / "baseline_model.pt")

    models["iterative_supervision"] = train_iterative_supervision(config, device)
    torch.save(models["iterative_supervision"].state_dict(), output_dir / "iterative_supervision_model.pt")

    models["role_disambiguated_residual"] = train_role_disambiguated_residual(config, device)
    torch.save(models["role_disambiguated_residual"].state_dict(), output_dir / "role_disambiguated_residual_model.pt")

    # Test all models
    print(f"\n{'='*60}")
    print("TESTING ALL APPROACHES")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\nTesting {name}...")
        all_results[name] = test_model(model, config, device, name)

    print_results_table(all_results)

    # Save results
    results = {"config": asdict(config), "results": all_results}
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
