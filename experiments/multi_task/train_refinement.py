"""
Test role-disambiguated residual refinement across different ICL tasks.

Tasks:
1. LINEAR_SYSTEM  - Ax=b, clean SPD matrix (baseline)
2. NOISY_MATRIX   - Ax=b, model gets A_noisy != A_true (denoising/robustness)
3. KERNEL_REGRESS - Kernel regression, K_reg as matrix token (K=d)

Each task: train a fresh model with rw=0.8, evaluate iterative refinement.
All tasks use the same token format—only the data generation changes.

Task 1: clean inverse problem. Task 2: inverse problem where the matrix token
is a noisy version of the true matrix, forcing the model to reconcile matrix
hint with context evidence. Task 3: kernel regression where the matrix token is
the regularized kernel matrix K+λI encoding pairwise context relationships.

Usage:
    python experiments/multi_task/train_refinement.py --device cuda
    python experiments/multi_task/train_refinement.py --device cuda --tasks 2 3
    python experiments/multi_task/train_refinement.py --device cuda --training_steps 30000
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json
import math
import time
import sys

_src_dir = Path(__file__).parent.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import ComponentTransformerModel, ComponentModelConfig
from curriculum_model.roles import Role
from data.spd_sampler import sample_spd


# ============================================================
# Batch functions — all return (matrix, ctx_inputs, ctx_outputs, query_input, query_target)
# ============================================================

def make_batch_linear_system(B: int, K: int, d: int, device: torch.device):
    """Ax=b with SPD matrix A. Identical to existing SPD task."""
    A = sample_spd(B, d, device, 1.0, 100.0)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_noisy_matrix(B: int, K: int, d: int, device: torch.device):
    """Solve Ax=b where the model receives a NOISY version of A.

    Matrix token: A_noisy = A_true + symmetric noise (SPD, close to A_true).
    Context: (b_i, x_i) where A_true @ x_i = b_i (perfectly consistent
             with the TRUE matrix, not the noisy one).
    Query: b_query -> predict x_query = A_true^{-1} @ b_query.

    The model must reconcile the noisy matrix token with the clean context
    evidence. Context pairs are inconsistent with A_noisy (A_noisy @ x_i != b_i),
    forcing the model to learn implicit denoising of the matrix.

    Per-sample noise scale varies so each instance has a different
    matrix-context discrepancy.
    """
    A_true = sample_spd(B, d, device, 1.0, 100.0)

    # Per-sample symmetric noise (scale in [0.1, 1.0])
    noise_scale = 0.1 + 0.9 * torch.rand(B, 1, 1, device=device)
    raw_noise = torch.randn(B, d, d, device=device)
    sym_noise = 0.5 * (raw_noise + raw_noise.transpose(-2, -1))
    A_noisy = A_true + noise_scale * sym_noise

    # Ensure A_noisy stays SPD: shift eigenvalues above 0.1
    eigs = torch.linalg.eigvalsh(A_noisy)
    min_eig = eigs[:, 0].view(B, 1, 1)
    shift = torch.clamp(0.1 - min_eig, min=0)
    A_noisy = A_noisy + shift * torch.eye(d, device=device).unsqueeze(0)

    # Clean linear system with A_true
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A_true, b_all.transpose(-2, -1)).transpose(-2, -1)

    return A_noisy, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


def make_batch_kernel_regress(B: int, K: int, d: int, device: torch.device,
                              noise_std: float = 0.1):
    """Kernel regression: predict f(x_query) from context (x_i, y_i).

    Uses K=d context points so the kernel matrix K_reg is d×d, fitting
    the matrix embedder exactly.

    Matrix token: K_reg = K_ctx + λI (d×d regularized RBF kernel matrix).
    This encodes pairwise distances between context points — information the
    model needs for kernel-weighted prediction but cannot easily compute from
    the context vector tokens alone.

    Ground truth: random Fourier features f(x) = W_out @ sin(W_in @ x + bias).
    Target: y_query = k_qc^T @ K_reg^{-1} @ Y_ctx (kernel prediction).

    Bandwidth and lambda sampled per-sample so each instance has different
    kernel structure.
    """
    assert K == d, f"Kernel regression requires K=d, got K={K}, d={d}"

    n_features = 2 * d

    # Per-sample hyperparameters (log-uniform)
    bandwidth = torch.exp(
        torch.rand(B, device=device) * (math.log(4.0) - math.log(1.0)) + math.log(1.0)
    )                                                                      # (B,) in [1, 4]
    lambda_reg = torch.exp(
        torch.rand(B, device=device) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
    )                                                                      # (B,) in [0.001, 0.1]

    # Random nonlinear function via Fourier features
    W_in = torch.randn(B, n_features, d, device=device) / bandwidth.view(B, 1, 1)
    bias = torch.rand(B, n_features, device=device) * (2 * math.pi)
    W_out = torch.randn(B, d, n_features, device=device) / math.sqrt(n_features)

    def f(x):
        """x: (B, N, d) -> (B, N, d)"""
        projected = x @ W_in.transpose(-2, -1)                            # (B, N, 2d)
        features = torch.sin(projected + bias.unsqueeze(1))                # (B, N, 2d)
        return features @ W_out.transpose(-2, -1)                          # (B, N, d)

    # Context data (K=d points)
    X_ctx = torch.randn(B, K, d, device=device)                           # (B, d, d)
    Y_ctx = f(X_ctx) + noise_std * torch.randn(B, K, d, device=device)    # (B, d, d)

    # Query
    x_query = torch.randn(B, d, device=device)                            # (B, d)

    # RBF kernel matrix between context points: K_ij = exp(-||x_i-x_j||^2 / 2σ^2)
    diff_ctx = X_ctx.unsqueeze(2) - X_ctx.unsqueeze(1)                    # (B, d, d, d)
    bw_sq = (2.0 * bandwidth.pow(2)).view(B, 1, 1)
    K_ctx = torch.exp(-diff_ctx.pow(2).sum(-1) / bw_sq)                   # (B, d, d)

    # Regularized kernel matrix (this IS the matrix token)
    lam = lambda_reg.view(B, 1, 1) * torch.eye(K, device=device).unsqueeze(0)
    K_reg = K_ctx + lam                                                    # (B, d, d)

    # Kernel vector: k(x_query, x_i) for each context point
    diff_q = x_query.unsqueeze(1) - X_ctx                                 # (B, d, d)
    bw_sq_k = (2.0 * bandwidth.pow(2)).view(B, 1)
    k_qc = torch.exp(-diff_q.pow(2).sum(-1) / bw_sq_k)                   # (B, d)

    # Kernel prediction: y = k_qc^T @ K_reg^{-1} @ Y_ctx
    alpha = torch.linalg.solve(K_reg, Y_ctx)                              # (B, d, d)
    y_target = (k_qc.unsqueeze(-1) * alpha).sum(1)                        # (B, d)

    return K_reg, X_ctx, Y_ctx, x_query, y_target


TASKS = {
    1: {"name": "LINEAR_SYSTEM",  "desc": "Ax=b, clean SPD (baseline)",              "batch_fn": make_batch_linear_system,  "K": 5},
    2: {"name": "NOISY_MATRIX",   "desc": "Ax=b, noisy A (matrix-context mismatch)", "batch_fn": make_batch_noisy_matrix,   "K": 5},
    3: {"name": "KERNEL_REGRESS", "desc": "Kernel regression, K_reg matrix (K=d)",   "batch_fn": make_batch_kernel_regress, "K": 4},
}


# ============================================================
# Training infrastructure (same pattern as multi_task_test.py)
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

    def build_standard(self, matrix, ctx_in, ctx_out, query_in):
        """Build token sequence: [SEP, matrix, SEP, in_1, out_1, ..., SEP, query_in, MASK]"""
        B, K = ctx_in.shape[:2]
        d, device = self.d, self.device
        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        in_role = self._get_role('bias')
        out_role = self._get_role('output')

        mat_emb = embedders.matrix(matrix) + matrix_role
        in_flat = ctx_in.reshape(B * K, d)
        out_flat = ctx_out.reshape(B * K, d)
        n_embd = embedders.vector(in_flat).shape[-1]

        in_emb = embedders.vector(in_flat).reshape(B, K, n_embd) + in_role
        out_emb = embedders.vector(out_flat).reshape(B, K, n_embd) + out_role
        q_emb = embedders.vector(query_in) + in_role

        seq_len = 3 * K + 5
        tokens = torch.zeros(B, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        sep, mask = special.get_sep_batch(B), special.get_mask_batch(B)
        tokens[:, 0], tokens[:, 1] = sep, mat_emb

        for i in range(K):
            idx = 2 + i * 3
            tokens[:, idx] = sep
            tokens[:, idx + 1] = in_emb[:, i]
            tokens[:, idx + 2] = out_emb[:, i]
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx] = sep
        tokens[:, q_idx + 1] = q_emb
        tokens[:, q_idx + 2] = mask
        ex_pos[:, q_idx:q_idx + 3] = K + 1

        return tokens, ex_pos, torch.full((B,), seq_len - 1, dtype=torch.long, device=device)

    def build_with_estimate(self, matrix, ctx_in, ctx_out, query_in, estimate):
        """Build token sequence with estimate: [SEP, matrix, ..., SEP, query_in, estimate, MASK]"""
        B, K = ctx_in.shape[:2]
        d, device = self.d, self.device
        embedders = self.model.embedders
        special = self.model.special_tokens

        matrix_role = self._get_role('matrix')
        in_role = self._get_role('bias')
        out_role = self._get_role('output')
        estimate_role = self._get_role('estimate')

        mat_emb = embedders.matrix(matrix) + matrix_role
        in_flat = ctx_in.reshape(B * K, d)
        out_flat = ctx_out.reshape(B * K, d)
        n_embd = embedders.vector(in_flat).shape[-1]

        in_emb = embedders.vector(in_flat).reshape(B, K, n_embd) + in_role
        out_emb = embedders.vector(out_flat).reshape(B, K, n_embd) + out_role
        q_emb = embedders.vector(query_in) + in_role
        est_emb = embedders.vector(estimate) + estimate_role

        seq_len = 3 * K + 6
        tokens = torch.zeros(B, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        sep, mask = special.get_sep_batch(B), special.get_mask_batch(B)
        tokens[:, 0], tokens[:, 1] = sep, mat_emb

        for i in range(K):
            idx = 2 + i * 3
            tokens[:, idx] = sep
            tokens[:, idx + 1] = in_emb[:, i]
            tokens[:, idx + 2] = out_emb[:, i]
            ex_pos[:, idx:idx + 3] = i + 1

        q_idx = 2 + K * 3
        tokens[:, q_idx] = sep
        tokens[:, q_idx + 1] = q_emb
        tokens[:, q_idx + 2] = est_emb
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
    d, B = 4, 64
    K = task["K"]
    batch_fn = task["batch_fn"]

    print(f"\n{'#'*60}")
    print(f"# TASK {task_id}: {task['name']} — {task['desc']}")
    print(f"{'#'*60}")
    print(f"  K={K}, d={d}, B={B}")

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

    rw_max = 0.8
    rw_warmup = training_steps // 4  # ramp residual weight over first 25%
    noise_scale = 0.5
    alpha_power = 0.5

    model.train()
    start = time.time()
    for step in range(training_steps):
        matrix, ctx_in, ctx_out, q_in, q_target = batch_fn(B, K, d, device)

        # Residual weight warmup: 0 -> rw_max over first 25% of training
        rw = min(rw_max, rw_max * step / max(rw_warmup, 1))

        # Direct prediction
        tokens, ex_pos, mask_pos = trainer.build_standard(matrix, ctx_in, ctx_out, q_in)
        output = model(tokens, ex_pos, mask_pos)
        pred_0 = output.vector_output
        loss_direct = F.mse_loss(pred_0, q_target)

        # Residual prediction
        with torch.no_grad():
            alpha = torch.rand(B, 1, device=device) ** alpha_power
            noise = torch.randn_like(pred_0) * (
                torch.rand(B, 1, device=device) * noise_scale
            )
            x_estimate = alpha * pred_0.detach() + (1 - alpha) * q_target + noise
            true_residual = q_target - x_estimate

        tokens_r, ep_r, mp_r = trainer.build_with_estimate(
            matrix, ctx_in, ctx_out, q_in, x_estimate
        )
        pred_residual = model(tokens_r, ep_r, mp_r).vector_output
        loss_residual = F.mse_loss(pred_residual, true_residual)

        total_loss = (1 - rw) * loss_direct + rw * loss_residual

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
                  f"rw={rw:.3f} | {elapsed:.0f}s")

    train_time = time.time() - start
    print(f"  Training done in {train_time:.0f}s")

    results = evaluate_task(model, trainer, batch_fn, d, device, K=K)
    return results, train_time


def evaluate_task(model, trainer, batch_fn, d, device, max_iters=10,
                  test_batches=30, B=64, K=4):
    model.eval()
    all_mse = {i: [] for i in range(max_iters)}

    for _ in range(test_batches):
        with torch.no_grad():
            matrix, ctx_in, ctx_out, q_in, q_target = batch_fn(B, K, d, device)

            tokens, ex_pos, mask_pos = trainer.build_standard(matrix, ctx_in, ctx_out, q_in)
            x_current = model(tokens, ex_pos, mask_pos).vector_output
            all_mse[0].append(F.mse_loss(x_current, q_target).item())

            for i in range(1, max_iters):
                tokens_r, ep, mp = trainer.build_with_estimate(
                    matrix, ctx_in, ctx_out, q_in, x_current
                )
                correction = model(tokens_r, ep, mp).vector_output
                x_current = x_current + correction
                all_mse[i].append(F.mse_loss(x_current, q_target).item())

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
            matrix, ctx_in, ctx_out, q_in, q_target = batch_fn(B, K, d, device)

            tokens, ex_pos, mask_pos = trainer.build_standard(matrix, ctx_in, ctx_out, q_in)
            x_current = model(tokens, ex_pos, mask_pos).vector_output
            all_mse_d[0].append(F.mse_loss(x_current, q_target).item())

            for i in range(1, max_iters):
                tokens_r, ep, mp = trainer.build_with_estimate(
                    matrix, ctx_in, ctx_out, q_in, x_current
                )
                correction = model(tokens_r, ep, mp).vector_output
                x_current = x_current + 0.5 * correction
                all_mse_d[i].append(F.mse_loss(x_current, q_target).item())

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
    parser = argparse.ArgumentParser(
        description="Multi-task ICL refinement: clean, noisy matrix, kernel regression"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", type=int, nargs="*", default=None,
                        help="Which tasks to run (1-3). Default: all")
    parser.add_argument("--training_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="results/multi_task")
    args = parser.parse_args()

    task_ids = args.tasks or list(TASKS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-TASK ICL REFINEMENT — Clean / Noisy Matrix / Kernel")
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

        with open(output_dir / "results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison table
    print(f"\n{'='*60}")
    print("FINAL COMPARISON — Refinement across ICL task types")
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
