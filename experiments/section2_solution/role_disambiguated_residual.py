"""
Section 2: Solution - Role-Disambiguated Residual Prediction

This is the main architectural contribution: a principled fix that enables successful
iterative self-refinement through:

1. Role-based disambiguation: Use VEC_SECONDARY role for current estimates vs OUTPUT
   role for ground-truth solutions, allowing the model to distinguish between them.

2. Dual training objective: Mix direct prediction loss and residual prediction loss,
   teaching the model to output corrections when given the current estimate.

Refinement Algorithm:
    x_0 = f(context, query)                 # Initial prediction (standard ICL)
    x_{k+1} = x_k + f(context, query, x_k)  # Refinement iterations

Usage:
    python experiments/section2_solution/role_disambiguated_residual.py --device cuda
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
    """Configuration for Role-Disambiguated Residual experiment."""
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

    # Residual training (KEY HYPERPARAMETERS)
    residual_weight: float = 0.5  # Mix of direct vs residual loss
    noise_scale: float = 0.5      # Noise added to estimates for robustness

    # Data
    num_context: int = 5
    kappa_min: float = 1.0
    kappa_max: float = 100.0

    # Curriculum training (optional)
    curriculum: bool = False
    curriculum_warmup: int = 10000

    # Testing
    test_iterations: int = 5
    test_batches: int = 50
    kappa_ranges: List[Tuple[float, float]] = None

    # Output
    output_dir: str = "results/section2"
    device: str = "cuda"
    log_every: int = 500

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]


class ResidualTrainer:
    """Trains model to predict residuals/corrections using role-based disambiguation."""

    def __init__(self, model: ComponentTransformerModel, config: Config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.d = config.d
        self.current_step = 0

        # Cache role indices (gradient flow through role_embedding)
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
            'estimate': torch.tensor(Role.VEC_SECONDARY.value, device=device),  # KEY: separate role!
        }

    def _get_role(self, name: str) -> torch.Tensor:
        return self.model.role_embedding(self._role_indices[name])

    def get_kappa_range(self, step: int) -> Tuple[float, float]:
        """Get kappa range (supports curriculum learning)."""
        if not self.config.curriculum:
            return self.config.kappa_min, self.config.kappa_max

        if step < self.config.curriculum_warmup:
            return 1.0, 10.0

        progress = (step - self.config.curriculum_warmup) / (
            self.config.training_steps - self.config.curriculum_warmup
        )
        progress = min(progress, 1.0)
        kappa_max = 10.0 + progress * 190.0
        return 1.0, kappa_max

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

        The x_estimate uses VEC_SECONDARY role (not OUTPUT), enabling the model
        to distinguish current estimates from ground-truth solutions.
        """
        B, K = b_ctx.shape[:2]
        d, device = self.d, self.device

        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        bias_role = self._get_role('bias')
        output_role = self._get_role('output')
        estimate_role = self._get_role('estimate')  # KEY: different role!

        A_emb = embedders.matrix(A) + matrix_role
        b_flat = b_ctx.reshape(B * K, d)
        x_flat = x_ctx.reshape(B * K, d)
        n_embd = embedders.vector(b_flat).shape[-1]

        b_emb = embedders.vector(b_flat).reshape(B, K, n_embd) + bias_role
        x_emb = embedders.vector(x_flat).reshape(B, K, n_embd) + output_role
        b_q_emb = embedders.vector(b_query) + bias_role
        x_est_emb = embedders.vector(x_estimate) + estimate_role  # Uses ESTIMATE role!

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
        """One training step with dual objective (direct + residual loss)."""
        B, K, d = self.config.batch_size, self.config.num_context, self.d
        device = self.device

        kappa_min, kappa_max = self.get_kappa_range(self.current_step)
        A = sample_spd(B, d, device, kappa_min, kappa_max)
        self.current_step += 1

        b_all = torch.randn(B, K + 1, d, device=device)
        x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

        b_ctx = b_all[:, :K]
        x_ctx = x_all[:, :K]
        b_query = b_all[:, K]
        x_target = x_all[:, K]

        total_loss = 0.0
        losses = {}

        # Part 1: Direct prediction loss (iteration 0)
        tokens, ex_pos, mask_pos = self.build_tokens_standard(A, b_ctx, x_ctx, b_query)
        output = self.model(tokens, ex_pos, mask_pos)
        pred_0 = output.vector_output
        loss_direct = F.mse_loss(pred_0, x_target)
        total_loss = total_loss + (1 - self.config.residual_weight) * loss_direct
        losses["direct"] = loss_direct.item()

        # Part 2: Residual prediction loss
        if self.config.residual_weight > 0:
            with torch.no_grad():
                # Interpolate between pred_0 (far from x*) and x_target (at x*)
                # so the model sees the full range of estimate qualities:
                # alpha≈1: learns initial correction, alpha≈0: learns to output ~0
                # Squaring biases toward pred_0 (most samples have large alpha)
                # while still covering near-x* estimates for fixed-point stability
                alpha = torch.rand(B, 1, device=device) ** 0.5
                noise = torch.randn_like(pred_0) * (
                    torch.rand(B, 1, device=device) * self.config.noise_scale
                )
                x_estimate = alpha * pred_0.detach() + (1 - alpha) * x_target + noise
                true_residual = x_target - x_estimate

            tokens_r, ex_pos_r, mask_pos_r = self.build_tokens_with_estimate(
                A, b_ctx, x_ctx, b_query, x_estimate
            )
            output_r = self.model(tokens_r, ex_pos_r, mask_pos_r)
            pred_residual = output_r.vector_output

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


def train(config: Config) -> Tuple[ComponentTransformerModel, List[Dict]]:
    """Train Role-Disambiguated Residual model."""
    device = torch.device(config.device)

    print(f"\n{'='*60}")
    print("ROLE-DISAMBIGUATED RESIDUAL: RESIDUAL PREDICTION WITH ROLE DISAMBIGUATION")
    print(f"{'='*60}")
    print(f"Residual weight: {config.residual_weight}")
    print(f"Noise scale: {config.noise_scale}")

    model_config = ComponentModelConfig(
        d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    model = ComponentTransformerModel(model_config).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.training_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = ResidualTrainer(model, config, device)
    history = []
    start_time = time.time()

    model.train()
    for step in range(config.training_steps):
        losses = trainer.training_step(optimizer)
        scheduler.step()

        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Total: {losses['total']:.6f} | "
                  f"Direct: {losses['direct']:.6f} | Residual: {losses['residual']:.6f} | "
                  f"Time: {elapsed:.1f}s")
            history.append({"step": step, **losses})

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")
    return model, history


def test(model: ComponentTransformerModel, config: Config) -> Dict:
    """Test Role-Disambiguated Residual with iterative residual refinement."""
    device = torch.device(config.device)
    model.eval()

    print(f"\n{'='*60}")
    print("TESTING RESIDUAL REFINEMENT")
    print(f"{'='*60}")

    trainer = ResidualTrainer(model, config, device)
    results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        print(f"\nkappa in [{kappa_min}, {kappa_max}]")

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

        mse_summary = {i: {"mean": np.mean(m), "std": np.std(m)} for i, m in all_mse.items()}
        improved_frac = sum(1 for imp in improvements if imp > 1) / len(improvements)

        results[kappa_key] = {
            "mse_by_iteration": mse_summary,
            "improvement_ratio": {"mean": np.mean(improvements), "std": np.std(improvements)},
            "improved_fraction": improved_frac,
        }

        print(f"  MSE: {mse_summary[0]['mean']:.6f} -> {mse_summary[config.test_iterations-1]['mean']:.6f}")
        print(f"  Improvement: {np.mean(improvements):.2f}x, Fraction improved: {improved_frac*100:.1f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Role-Disambiguated Residual Prediction")
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--residual_weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--kappa_min", type=float, default=1.0)
    parser.add_argument("--kappa_max", type=float, default=100.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_warmup", type=int, default=10000)
    parser.add_argument("--test_iterations", type=int, default=5)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/section2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config(
        training_steps=args.training_steps,
        residual_weight=args.residual_weight,
        test_iterations=args.test_iterations,
        output_dir=args.output_dir,
        device=args.device,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        lr=args.lr,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        noise_scale=args.noise_scale,
        curriculum=args.curriculum,
        curriculum_warmup=args.curriculum_warmup,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    if args.test_only:
        model_path = args.model_path or output_dir / "model.pt"
        print(f"Loading model from {model_path}")
        model_config = ComponentModelConfig(
            d=config.d, n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
            n_positions=128, max_examples=64, dropout=0.0
        )
        model = ComponentTransformerModel(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        test_results = test(model, config)
        results = {"config": asdict(config), "testing": test_results}
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nTest results saved to {output_dir / 'test_results.json'}")
        return

    model, train_history = train(config)
    torch.save(model.state_dict(), output_dir / "model.pt")

    test_results = test(model, config)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_fracs = [r["improved_fraction"] for r in test_results.values()]
    avg_frac = np.mean(all_fracs)
    if avg_frac >= 0.5:
        print(f"SUCCESS: {avg_frac*100:.1f}% of samples improved (>=50%)")
    else:
        print(f"BELOW THRESHOLD: {avg_frac*100:.1f}% of samples improved (<50%)")

    results = {"config": asdict(config), "training": train_history, "testing": test_results}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
