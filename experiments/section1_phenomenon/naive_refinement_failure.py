"""
Section 1: Phenomenon - Naive ICL Self-Refinement Fails Catastrophically

This experiment demonstrates the core problem: when a standard ICL model
attempts iterative self-refinement, performance degrades drastically.

Key Results to Generate:
- Standard ICL:           MSE = ~5.93e-05
- Naive refinement (K=1): MSE = ~0.095      (1600x worse!)
- Naive refinement (K=2): MSE = ???
- Naive refinement (K=3): MSE = ???

The model confuses its own predictions (added back as context) with ground truth,
leading to catastrophic error accumulation.

Usage:
    python experiments/section1_phenomenon/naive_refinement_failure.py --device cuda
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
    """Configuration for naive refinement failure experiment."""
    # Model
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4

    # Training (standard ICL - no refinement training)
    training_steps: int = 50000
    batch_size: int = 64
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Data
    num_context: int = 5
    kappa_min: float = 1.0
    kappa_max: float = 100.0

    # Testing
    test_iterations: int = 5  # How many refinement iterations to test
    test_batches: int = 50
    kappa_ranges: List[Tuple[float, float]] = None

    # Output
    output_dir: str = "results/section1"
    device: str = "cuda"
    log_every: int = 500

    def __post_init__(self):
        if self.kappa_ranges is None:
            self.kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]


class TokenBuilder:
    """Builds token sequences for standard ICL."""

    def __init__(self, model: ComponentTransformerModel, d: int, device: torch.device):
        self.model = model
        self.d = d
        self.device = device
        self._role_indices = {
            'matrix': torch.tensor(Role.MATRIX.value, device=device),
            'bias': torch.tensor(Role.VEC_BIAS.value, device=device),
            'output': torch.tensor(Role.OUTPUT.value, device=device),
        }

    def _get_role(self, name: str) -> torch.Tensor:
        return self.model.role_embedding(self._role_indices[name])

    def build_standard(
        self, A: torch.Tensor, b_ctx: torch.Tensor, x_ctx: torch.Tensor, b_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build standard ICL tokens: [SEP, A, SEP, b_1, x_1, ..., SEP, b_query, MASK]"""
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


def train_standard_icl(config: Config) -> ComponentTransformerModel:
    """Train a standard ICL model (no refinement training)."""
    device = torch.device(config.device)

    print(f"\n{'='*60}")
    print("TRAINING: STANDARD ICL BASELINE")
    print(f"{'='*60}")

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

    builder = TokenBuilder(model, config.d, device)
    start_time = time.time()

    model.train()
    for step in range(config.training_steps):
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

        if step % config.log_every == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.6f} | Time: {time.time()-start_time:.1f}s")

    print(f"\nTraining complete in {time.time() - start_time:.1f}s")
    return model


def test_naive_refinement(model: ComponentTransformerModel, config: Config) -> Dict:
    """
    Test naive refinement: add model predictions back as context.
    This demonstrates the catastrophic failure phenomenon.
    """
    device = torch.device(config.device)
    model.eval()
    builder = TokenBuilder(model, config.d, device)

    print(f"\n{'='*60}")
    print("TESTING: NAIVE REFINEMENT (EXPECTED TO FAIL)")
    print(f"{'='*60}")

    results = {}

    for kappa_min, kappa_max in config.kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        print(f"\nkappa in [{kappa_min}, {kappa_max}]")

        all_mse = {i: [] for i in range(config.test_iterations)}

        for _ in range(config.test_batches):
            B, K, d = config.batch_size, config.num_context, config.d

            with torch.no_grad():
                A = sample_spd(B, d, device, kappa_min, kappa_max)
                b_all = torch.randn(B, K + 1, d, device=device)
                x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)

                b_ctx = b_all[:, :K].clone()
                x_ctx = x_all[:, :K].clone()
                b_query = b_all[:, K]
                x_target = x_all[:, K]

                # Naive refinement: add predictions to context
                for i in range(config.test_iterations):
                    tokens, ex_pos, mask_pos = builder.build_standard(A, b_ctx, x_ctx, b_query)
                    pred = model(tokens, ex_pos, mask_pos).vector_output

                    mse = F.mse_loss(pred, x_target).item()
                    all_mse[i].append(mse)

                    # Add prediction to context (this is the NAIVE approach)
                    b_ctx = torch.cat([b_ctx, b_query.unsqueeze(1)], dim=1)
                    x_ctx = torch.cat([x_ctx, pred.unsqueeze(1)], dim=1)

        # Compute statistics
        mse_summary = {}
        for i, mses in all_mse.items():
            mse_summary[i] = {"mean": float(np.mean(mses)), "std": float(np.std(mses))}

        results[kappa_key] = {
            "mse_by_iteration": mse_summary,
            "degradation_factor": mse_summary[config.test_iterations - 1]["mean"] / mse_summary[0]["mean"]
        }

        # Print results
        print(f"  Iteration 0 (standard ICL): MSE = {mse_summary[0]['mean']:.6e}")
        for i in range(1, config.test_iterations):
            deg_factor = mse_summary[i]["mean"] / mse_summary[0]["mean"]
            print(f"  Iteration {i} (naive refine): MSE = {mse_summary[i]['mean']:.6e} ({deg_factor:.0f}x degradation)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demonstrate naive refinement failure")
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--test_iterations", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/section1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config(
        training_steps=args.training_steps,
        test_iterations=args.test_iterations,
        output_dir=args.output_dir,
        device=args.device,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    # Train standard ICL model
    model = train_standard_icl(config)

    # Save model
    torch.save(model.state_dict(), output_dir / "standard_icl_model.pt")

    # Test naive refinement (demonstrate failure)
    results = test_naive_refinement(model, config)

    # Summary
    print(f"\n{'='*60}")
    print("PHENOMENON SUMMARY")
    print(f"{'='*60}")

    # Aggregate across kappa ranges
    all_deg_factors = [r["degradation_factor"] for r in results.values()]
    avg_degradation = np.mean(all_deg_factors)

    print(f"\nAverage degradation factor: {avg_degradation:.0f}x")
    print(f"\nCONCLUSION: Naive ICL self-refinement fails catastrophically!")
    print(f"Adding model predictions to context degrades performance by ~{avg_degradation:.0f}x")

    # Save results
    full_results = {
        "config": asdict(config),
        "phenomenon": "naive_refinement_failure",
        "testing": results,
        "summary": {
            "avg_degradation_factor": float(avg_degradation),
            "conclusion": "Naive ICL self-refinement fails catastrophically"
        }
    }

    with open(output_dir / "naive_refinement_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
