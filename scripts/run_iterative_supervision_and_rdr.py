"""
Combined Test Suite for Iterative Supervision and Role-Disambiguated Residual

Runs both iterative supervision and residual prediction approaches,
then compares results against a standard baseline.

Expected runtime: ~2 hours
- Baseline training: ~45 min
- Iterative Supervision training: ~45 min
- Role-Disambiguated Residual training: ~45 min
- Testing: ~15 min

Usage:
    python scripts/run_approaches_b_c.py
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
_src_dir = Path(__file__).parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import (
    ComponentTransformerModel,
    ComponentModelConfig,
)
from curriculum_model.roles import Role


@dataclass
class Config:
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
    output_dir: str = "experiment_results/approaches_b_c"
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
        """Training steps for an approach, equalized by total forward passes."""
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


def create_scheduler(optimizer, config: Config, total_steps: int):
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_spd(B: int, d: int, device: torch.device, kappa_min: float, kappa_max: float) -> torch.Tensor:
    log_min, log_max = np.log(kappa_min), np.log(kappa_max)
    u = torch.rand(B, device=device)
    kappas = torch.exp(torch.tensor(log_min, device=device) + u * (log_max - log_min))
    u_eigs = torch.rand(B, d, device=device)
    eigs = torch.exp(u_eigs * kappas.unsqueeze(-1).log())
    G = torch.randn(B, d, d, device=device)
    Q, _ = torch.linalg.qr(G)
    A = Q @ torch.diag_embed(eigs) @ Q.transpose(-2, -1)
    return 0.5 * (A + A.transpose(-2, -1))


class TokenBuilder:
    """Shared token building utilities."""

    def __init__(self, model: ComponentTransformerModel, d: int, device: torch.device):
        self.model = model
        self.d = d
        self.device = device

        # Cache role indices (not embeddings) for efficient lookup
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        """Get role embedding (with gradient flow)."""
        return self.model.role_embedding(self._role_indices[name])

    def build_standard(self, A, b_ctx, x_ctx, b_query):
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device
        embedders, special = self.model.embedders, self.model.special_tokens

        # Get role embeddings (with gradient flow)
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

        # Get role embeddings (with gradient flow)
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


# =============================================================================
# BASELINE TRAINING
# =============================================================================

def train_baseline(config: Config, device: torch.device) -> ComponentTransformerModel:
    total_steps = config.steps_for("baseline")

    print(f"\n{'='*60}")
    print(f"TRAINING: BASELINE ({total_steps} steps, {config.forwards_per_step('baseline')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Adjust scheduler for longer training
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

        tokens, ex_pos, mask_pos = builder.build_standard(
            A, b_all[:, :K], x_all[:, :K], b_all[:, K]
        )
        pred = model(tokens, ex_pos, mask_pos).vector_output
        loss = F.mse_loss(pred, x_all[:, K])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        log_every = config.log_every
        if step % log_every == 0:
            print(f"Step {step:5d}/{total_steps} | Loss: {loss.item():.6f} | Time: {time.time()-start:.1f}s")

    print(f"Baseline training complete ({total_steps} steps) in {time.time()-start:.1f}s")
    return model


# =============================================================================
# ITERATIVE SUPERVISION
# =============================================================================

def train_iterative_supervision(config: Config, device: torch.device) -> ComponentTransformerModel:
    total_steps = config.steps_for("iterative_supervision")

    print(f"\n{'='*60}")
    print(f"TRAINING: ITERATIVE SUPERVISION ({total_steps} steps, {config.forwards_per_step('iterative_supervision')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config, total_steps)
    builder = TokenBuilder(model, config.d, device)

    weights = config.b_iteration_weights
    weights = [w / sum(weights) for w in weights]

    model.train()
    start = time.time()

    for step in range(total_steps):
        B, K, d = config.batch_size, config.num_context, config.d
        n_iter = config.b_train_iterations

        A = sample_spd(B, d, device, config.kappa_min, config.kappa_max)
        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        b_ctx, x_ctx = b_all[:, :K].clone(), x_all[:, :K].clone()
        b_query, x_target = b_all[:, K], x_all[:, K]

        total_loss = 0.0
        for i in range(n_iter):
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
            print(f"Step {step:5d} | Loss: {total_loss.item():.6f} | Time: {time.time()-start:.1f}s")

    print(f"Iterative Supervision training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# ROLE-DISAMBIGUATED RESIDUAL
# =============================================================================

def train_role_disambiguated_residual(config: Config, device: torch.device) -> ComponentTransformerModel:
    total_steps = config.steps_for("role_disambiguated_residual")

    print(f"\n{'='*60}")
    print(f"TRAINING: ROLE-DISAMBIGUATED RESIDUAL ({total_steps} steps, {config.forwards_per_step('role_disambiguated_residual')} fwd/step)")
    print(f"{'='*60}")

    model = create_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config, total_steps)
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

        # Direct prediction loss
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

        tokens_r, ex_pos_r, mask_pos_r = builder.build_with_estimate(
            A, b_ctx, x_ctx, b_query, x_estimate
        )
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
            print(f"Step {step:5d} | Total: {total_loss.item():.6f} | "
                  f"Direct: {loss_direct.item():.6f} | Residual: {loss_residual.item():.6f} | "
                  f"Time: {time.time()-start:.1f}s")

    print(f"Role-Disambiguated Residual training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# TESTING
# =============================================================================

def test_model(
    model: ComponentTransformerModel,
    config: Config,
    device: torch.device,
    approach: str,
) -> Dict:
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

                if approach == "baseline" or approach == "iterative_supervision":
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
                        tokens_r, ex_pos_r, mask_pos_r = builder.build_with_estimate(
                            A, b_ctx, x_ctx, b_query, x_current
                        )
                        residual = model(tokens_r, ex_pos_r, mask_pos_r).vector_output
                        x_current = x_current + residual

                        mse = F.mse_loss(x_current, x_target).item()
                        mse_history.append(mse)
                        all_mse[i].append(mse)

                improvements.append(mse_history[0] / mse_history[-1] if mse_history[-1] > 0 else 0)

        mse_summary = {i: {"mean": np.mean(m), "std": np.std(m)} for i, m in all_mse.items()}
        improved_frac = sum(1 for imp in improvements if imp > 1) / len(improvements)

        results[kappa_key] = {
            "mse_by_iteration": mse_summary,
            "improvement_ratio": {"mean": np.mean(improvements), "std": np.std(improvements)},
            "improved_fraction": improved_frac,
        }

    return results


def print_results_table(all_results: Dict[str, Dict]):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Approach':<15} {'Kappa':<12} {'MSE_0':<12} {'MSE_final':<12} {'Improvement':<12} {'Frac Improved'}")
    print("-" * 80)

    for approach, results in all_results.items():
        for kappa_key, stats in results.items():
            mse_0 = stats["mse_by_iteration"][0]["mean"]
            mse_final = stats["mse_by_iteration"][max(stats["mse_by_iteration"].keys())]["mean"]
            imp = stats["improvement_ratio"]["mean"]
            frac = stats["improved_fraction"] * 100
            print(f"{approach:<15} {kappa_key:<12} {mse_0:<12.6f} {mse_final:<12.6f} {imp:<12.2f}x {frac:.1f}%")


def print_summary(all_results: Dict[str, Dict]):
    """Print final summary."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for approach, results in all_results.items():
        fracs = [r["improved_fraction"] for r in results.values()]
        imps = [r["improvement_ratio"]["mean"] for r in results.values()]
        avg_frac = np.mean(fracs) * 100
        avg_imp = np.mean(imps)

        status = "PASS" if avg_frac >= 50 else "FAIL"
        print(f"\n{approach}:")
        print(f"  {status} Avg fraction improved: {avg_frac:.1f}%")
        print(f"    Avg improvement ratio: {avg_imp:.2f}x")

    # Determine winner
    best_approach = max(all_results.keys(),
                       key=lambda a: np.mean([r["improved_fraction"] for r in all_results[a].values()]))
    best_frac = np.mean([r["improved_fraction"] for r in all_results[best_approach].values()]) * 100

    print(f"\n{'='*80}")
    print(f"BEST APPROACH: {best_approach} ({best_frac:.1f}% improved)")
    print(f"{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="experiment_results/approaches_b_c")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline if model exists")
    args = parser.parse_args()

    config = Config(
        training_steps=args.training_steps,
        output_dir=args.output_dir,
        device=args.device,
    )

    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    models = {}

    # Load or train baseline
    baseline_path = output_dir / "baseline_model.pt"
    if args.skip_baseline and baseline_path.exists():
        print(f"Loading baseline from {baseline_path}")
        models["baseline"] = create_model(config, device)
        models["baseline"].load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    else:
        models["baseline"] = train_baseline(config, device)
        torch.save(models["baseline"].state_dict(), baseline_path)

    # Train Iterative Supervision
    iterative_supervision_path = output_dir / "iterative_supervision_model.pt"
    if iterative_supervision_path.exists():
        print(f"Loading Iterative Supervision from {iterative_supervision_path}")
        models["iterative_supervision"] = create_model(config, device)
        models["iterative_supervision"].load_state_dict(torch.load(iterative_supervision_path, map_location=device, weights_only=True))
    else:
        models["iterative_supervision"] = train_iterative_supervision(config, device)
        torch.save(models["iterative_supervision"].state_dict(), iterative_supervision_path)

    # Train Role-Disambiguated Residual
    rdr_path = output_dir / "role_disambiguated_residual_model.pt"
    if rdr_path.exists():
        print(f"Loading Role-Disambiguated Residual from {rdr_path}")
        models["role_disambiguated_residual"] = create_model(config, device)
        models["role_disambiguated_residual"].load_state_dict(torch.load(rdr_path, map_location=device, weights_only=True))
    else:
        models["role_disambiguated_residual"] = train_role_disambiguated_residual(config, device)
        torch.save(models["role_disambiguated_residual"].state_dict(), rdr_path)

    # Test all models
    print(f"\n{'='*60}")
    print("TESTING ALL APPROACHES")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\nTesting {name}...")
        all_results[name] = test_model(model, config, device, name)

    # Print results
    print_results_table(all_results)
    print_summary(all_results)

    # Save results
    results = {
        "config": asdict(config),
        "results": all_results,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
