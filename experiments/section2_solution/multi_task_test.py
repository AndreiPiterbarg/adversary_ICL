"""
Test HIGH_RESID_WT (residual_weight=0.8) across multiple ICL tasks.

Each task: train a fresh model with rw=0.8, evaluate iterative refinement.
All tasks use the same architecture—only the data distribution changes.

Tasks:
1. SPD (baseline)      - Ax=b, A is SPD, kappa 1-100
2. General invertible   - Ax=b, A is random invertible (not symmetric)
3. Ill-conditioned     - Ax=b, A is SPD, kappa 100-1000
4. Noisy context       - Ax=b but context x_i have Gaussian noise
5. Diagonal systems    - A is diagonal
6. Symmetric indef     - A is symmetric but not positive definite

Usage:
    python experiments/section2_solution/multi_task_test.py --device cuda
    python experiments/section2_solution/multi_task_test.py --device cuda --tasks 1 2 5
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
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

from curriculum_model.component_model import ComponentTransformerModel, ComponentModelConfig
from curriculum_model.roles import Role
from data.spd_sampler import sample_spd


# ============================================================
# Matrix samplers for each task
# ============================================================

def sample_general_invertible(B, d, device):
    """Random invertible matrix (not symmetric). Condition number ~1-100."""
    A = torch.randn(B, d, d, device=device)
    # Add scaled identity to ensure invertibility and control conditioning
    # Without this, random matrices can be near-singular
    A = A + 3.0 * torch.eye(d, device=device).unsqueeze(0)
    return A


def sample_diagonal(B, d, device):
    """Diagonal matrix with entries in [0.1, 10]."""
    diag_vals = torch.exp(torch.randn(B, d, device=device))  # log-normal
    diag_vals = diag_vals.clamp(0.1, 10.0)
    return torch.diag_embed(diag_vals)


def sample_symmetric_indefinite(B, d, device):
    """Symmetric but not positive definite. Eigenvalues in [-5, -0.5] ∪ [0.5, 5]."""
    G = torch.randn(B, d, d, device=device)
    Q, _ = torch.linalg.qr(G)
    # Eigenvalues: half negative, half positive
    eigs = torch.randn(B, d, device=device) * 2.0
    eigs = torch.where(eigs.abs() < 0.5, torch.sign(eigs) * 0.5, eigs)
    A = Q @ torch.diag_embed(eigs) @ Q.transpose(-2, -1)
    return 0.5 * (A + A.transpose(-2, -1))


# ============================================================
# Task definitions
# ============================================================

def make_batch_spd(B, K, d, device):
    A = sample_spd(B, d, device, 1.0, 100.0)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_general(B, K, d, device):
    A = sample_general_invertible(B, d, device)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_illcond(B, K, d, device):
    A = sample_spd(B, d, device, 100.0, 1000.0)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_noisy(B, K, d, device):
    A = sample_spd(B, d, device, 1.0, 100.0)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    # Add noise to context solutions only (not query target)
    x_ctx_noisy = x_all[:, :K] + torch.randn(B, K, d, device=device) * 0.1
    return A, b_all[:, :K], x_ctx_noisy, b_all[:, K], x_all[:, K]


def make_batch_diagonal(B, K, d, device):
    A = sample_diagonal(B, d, device)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_sym_indef(B, K, d, device):
    A = sample_symmetric_indefinite(B, d, device)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


TASKS = {
    1: {"name": "SPD", "desc": "SPD kappa 1-100 (baseline)", "batch_fn": make_batch_spd},
    2: {"name": "GENERAL", "desc": "General invertible (non-symmetric)", "batch_fn": make_batch_general},
    3: {"name": "ILL_COND", "desc": "SPD kappa 100-1000", "batch_fn": make_batch_illcond},
    4: {"name": "NOISY_CTX", "desc": "SPD with noisy context solutions", "batch_fn": make_batch_noisy},
    5: {"name": "DIAGONAL", "desc": "Diagonal matrices", "batch_fn": make_batch_diagonal},
    6: {"name": "SYM_INDEF", "desc": "Symmetric indefinite", "batch_fn": make_batch_sym_indef},
}


# ============================================================
# Training infrastructure (same as overnight_theories.py)
# ============================================================

class Trainer:
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


def make_model(d=4, device='cuda'):
    model_config = ComponentModelConfig(
        d=d, n_embd=128, n_layer=6, n_head=4,
        n_positions=128, max_examples=64, dropout=0.0
    )
    return ComponentTransformerModel(model_config).to(device)


# ============================================================
# Train + evaluate for one task
# ============================================================

def train_task(task_id, task, device_str, training_steps=50000):
    device = torch.device(device_str)
    d, K, B = 4, 5, 64
    batch_fn = task["batch_fn"]

    print(f"\n{'#'*60}")
    print(f"# TASK {task_id}: {task['name']} — {task['desc']}")
    print(f"{'#'*60}")

    model = make_model(d=d, device=device_str)
    trainer = Trainer(model, d, device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    warmup = 1000
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (training_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # HIGH_RESID_WT config
    residual_weight = 0.8
    noise_scale = 0.5
    alpha_power = 0.5

    model.train()
    start = time.time()
    for step in range(training_steps):
        A, b_ctx, x_ctx, b_query, x_target = batch_fn(B, K, d, device)

        # Direct prediction
        tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
        output = model(tokens, ex_pos, mask_pos)
        pred_0 = output.vector_output
        loss_direct = F.mse_loss(pred_0, x_target)

        # Residual prediction
        with torch.no_grad():
            alpha = torch.rand(B, 1, device=device) ** alpha_power
            noise = torch.randn_like(pred_0) * (
                torch.rand(B, 1, device=device) * noise_scale
            )
            x_estimate = alpha * pred_0.detach() + (1 - alpha) * x_target + noise
            true_residual = x_target - x_estimate

        tokens_r, ep_r, mp_r = trainer.build_with_estimate(
            A, b_ctx, x_ctx, b_query, x_estimate
        )
        pred_residual = model(tokens_r, ep_r, mp_r).vector_output
        loss_residual = F.mse_loss(pred_residual, true_residual)

        total_loss = (1 - residual_weight) * loss_direct + residual_weight * loss_residual

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 5000 == 0:
            elapsed = time.time() - start
            print(f"  Step {step:6d}/{training_steps} | "
                  f"total={total_loss.item():.6f} | "
                  f"direct={loss_direct.item():.6f} | "
                  f"residual={loss_residual.item():.6f} | "
                  f"{elapsed:.0f}s")

    train_time = time.time() - start
    print(f"  Training done in {train_time:.0f}s")

    # Evaluate
    results = evaluate_task(model, trainer, batch_fn, d, device)
    return results, train_time


def evaluate_task(model, trainer, batch_fn, d, device, max_iters=10,
                  test_batches=30, B=64, K=5):
    model.eval()
    all_mse = {i: [] for i in range(max_iters)}

    for _ in range(test_batches):
        with torch.no_grad():
            A, b_ctx, x_ctx, b_query, x_target = batch_fn(B, K, d, device)

            tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
            x_current = model(tokens, ex_pos, mask_pos).vector_output
            all_mse[0].append(F.mse_loss(x_current, x_target).item())

            for i in range(1, max_iters):
                tokens_r, ep, mp = trainer.build_with_estimate(
                    A, b_ctx, x_ctx, b_query, x_current
                )
                correction = model(tokens_r, ep, mp).vector_output
                x_current = x_current + correction
                all_mse[i].append(F.mse_loss(x_current, x_target).item())

    mse_means = [np.mean(all_mse[i]) for i in range(max_iters)]
    best_iter = int(np.argmin(mse_means))
    improvement = mse_means[0] / mse_means[best_iter] if mse_means[best_iter] > 0 else 0

    mono_improving = 0
    for i in range(1, max_iters):
        if mse_means[i] < mse_means[i - 1]:
            mono_improving = i
        else:
            break

    # Also evaluate with damping=0.5
    all_mse_d = {i: [] for i in range(max_iters)}
    for _ in range(test_batches):
        with torch.no_grad():
            A, b_ctx, x_ctx, b_query, x_target = batch_fn(B, K, d, device)

            tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
            x_current = model(tokens, ex_pos, mask_pos).vector_output
            all_mse_d[0].append(F.mse_loss(x_current, x_target).item())

            for i in range(1, max_iters):
                tokens_r, ep, mp = trainer.build_with_estimate(
                    A, b_ctx, x_ctx, b_query, x_current
                )
                correction = model(tokens_r, ep, mp).vector_output
                x_current = x_current + 0.5 * correction
                all_mse_d[i].append(F.mse_loss(x_current, x_target).item())

    mse_means_d = [np.mean(all_mse_d[i]) for i in range(max_iters)]
    best_iter_d = int(np.argmin(mse_means_d))
    improvement_d = mse_means_d[0] / mse_means_d[best_iter_d] if mse_means_d[best_iter_d] > 0 else 0

    mono_d = 0
    for i in range(1, max_iters):
        if mse_means_d[i] < mse_means_d[i - 1]:
            mono_d = i
        else:
            break

    model.train()
    return {
        "undamped": {
            "mse_per_iter": mse_means,
            "best_iter": best_iter,
            "improvement": improvement,
            "mono_improving": mono_improving,
        },
        "damped_0.5": {
            "mse_per_iter": mse_means_d,
            "best_iter": best_iter_d,
            "improvement": improvement_d,
            "mono_improving": mono_d,
        },
    }


def print_results(task_name, results):
    for mode in ["undamped", "damped_0.5"]:
        r = results[mode]
        tag = "" if mode == "undamped" else " (damped 0.5)"
        print(f"\n  {task_name}{tag}:")
        print(f"    {'iter':<6} {'MSE':<12}")
        print(f"    {'-'*20}")
        for i, mse in enumerate(r["mse_per_iter"]):
            marker = " <-- best" if i == r["best_iter"] else ""
            print(f"    {i:<6} {mse:.6f}{marker}")
        print(f"    Improvement: {r['improvement']:.2f}x | "
              f"Best @ iter {r['best_iter']} | "
              f"Mono improving: {r['mono_improving']} steps")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-task ICL test with HIGH_RESID_WT")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", type=int, nargs="*", default=None,
                        help="Which tasks to run (1-6). Default: all")
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="results/multi_task")
    args = parser.parse_args()

    task_ids = args.tasks or list(TASKS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-TASK ICL TEST — HIGH_RESID_WT (rw=0.8)")
    print("=" * 60)
    print(f"Tasks: {task_ids}")
    for tid in task_ids:
        t = TASKS[tid]
        print(f"  {tid}. {t['name']}: {t['desc']}")

    all_results = {}
    total_start = time.time()

    for tid in task_ids:
        task = TASKS[tid]
        results, train_time = train_task(tid, task, args.device, args.training_steps)
        all_results[tid] = {
            "task": {k: v for k, v in task.items() if k != "batch_fn"},
            "train_time": train_time,
            "results": results,
        }
        print_results(task["name"], results)

        # Save after each task
        with open(output_dir / "results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON — HIGH_RESID_WT across ICL tasks")
    print(f"{'='*60}")

    print(f"\n{'Task':<15} {'Mode':<12} {'iter0 MSE':<12} {'best MSE':<12} "
          f"{'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 75)
    for tid in task_ids:
        name = TASKS[tid]["name"]
        for mode in ["undamped", "damped_0.5"]:
            r = all_results[tid]["results"][mode]
            mode_short = "full" if mode == "undamped" else "damp"
            print(f"{name:<15} {mode_short:<12} "
                  f"{r['mse_per_iter'][0]:.6f}     "
                  f"{r['mse_per_iter'][r['best_iter']]:.6f}     "
                  f"{r['best_iter']:<6} "
                  f"{r['improvement']:.2f}x    "
                  f"{r['mono_improving']}")
        print()

    total_time = time.time() - total_start
    print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
