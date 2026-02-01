"""
Non-Linear Least Squares — Nonlinear Refinement Experiment

Fit y = theta[2]*sigmoid(theta[0]*t + theta[1]) + theta[3] with d=4 parameters
and N=4 data points.  Matrix token encodes data as (t, t^2, y, 1) per row.
Context demonstrates (theta_init -> theta_star) convergence.

Hypothesis test: compare model corrections against analytical Gauss-Newton steps
    delta = -(J^T J)^{-1} J^T r
at each refinement step and measure cosine similarity.

Usage:
    python experiments/nonlinear/nlls_refinement.py --device cuda
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
import json
import time
import sys
import argparse

_src_dir = Path(__file__).parent.parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import ComponentTransformerModel, ComponentModelConfig
from curriculum_model.roles import Role
from data.nlls_sampler import make_batch_nlls, gauss_newton_correction


# ============================================================
# Training infrastructure
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
    config = ComponentModelConfig(
        d=d, n_embd=128, n_layer=6, n_head=4,
        n_positions=128, max_examples=64, dropout=0.0
    )
    return ComponentTransformerModel(config).to(device)


# ============================================================
# Training
# ============================================================

def train(device_str, training_steps=100000):
    device = torch.device(device_str)
    d, B, K = 4, 64, 5

    print("=" * 60)
    print("NON-LINEAR LEAST SQUARES — Nonlinear Refinement")
    print("=" * 60)
    print(f"  d={d}, B={B}, K={K}, steps={training_steps}")

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
    rw_warmup = training_steps // 4
    noise_scale = 0.5
    alpha_power = 0.5

    model.train()
    start = time.time()
    for step in range(training_steps):
        matrix, ctx_in, ctx_out, q_in, q_target = make_batch_nlls(
            B, K, d, device)

        rw = min(rw_max, rw_max * step / max(rw_warmup, 1))

        # Direct prediction
        tokens, ex_pos, mask_pos = trainer.build_standard(
            matrix, ctx_in, ctx_out, q_in)
        pred_0 = model(tokens, ex_pos, mask_pos).vector_output
        loss_direct = F.mse_loss(pred_0, q_target)

        # Residual prediction
        with torch.no_grad():
            alpha = torch.rand(B, 1, device=device) ** alpha_power
            noise = torch.randn_like(pred_0) * (
                torch.rand(B, 1, device=device) * noise_scale)
            x_estimate = alpha * pred_0.detach() + (1 - alpha) * q_target + noise
            true_residual = q_target - x_estimate

        tokens_r, ep_r, mp_r = trainer.build_with_estimate(
            matrix, ctx_in, ctx_out, q_in, x_estimate)
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
    return model, trainer, train_time


# ============================================================
# Evaluation with Gauss-Newton hypothesis testing
# ============================================================

def evaluate(model, trainer, device, max_iters=20, test_batches=30, B=64):
    d, K = 4, 5
    model.eval()

    results = {}
    for damping in [1.0, 0.5]:
        tag = "undamped" if damping == 1.0 else "damped_0.5"
        all_mse = {i: [] for i in range(max_iters)}
        all_cosine = {i: [] for i in range(1, max_iters)}

        for _ in range(test_batches):
            with torch.no_grad():
                matrix, ctx_in, ctx_out, q_in, q_target = \
                    make_batch_nlls(B, K, d, device)

                tokens, ex_pos, mask_pos = trainer.build_standard(
                    matrix, ctx_in, ctx_out, q_in)
                x_current = model(tokens, ex_pos, mask_pos).vector_output
                all_mse[0].append(F.mse_loss(x_current, q_target).item())

                for i in range(1, max_iters):
                    # Analytical Gauss-Newton correction
                    gn_delta = gauss_newton_correction(
                        matrix, x_current, device)

                    # Model correction
                    tokens_r, ep, mp = trainer.build_with_estimate(
                        matrix, ctx_in, ctx_out, q_in, x_current)
                    model_delta = model(tokens_r, ep, mp).vector_output

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(
                        model_delta.reshape(-1, d),
                        gn_delta.reshape(-1, d), dim=-1)
                    all_cosine[i].append(cos_sim.mean().item())

                    x_current = x_current + damping * model_delta
                    all_mse[i].append(
                        F.mse_loss(x_current, q_target).item())

        mse_means = [np.mean(all_mse[i]) for i in range(max_iters)]
        cosine_means = [np.mean(all_cosine[i]) for i in range(1, max_iters)]
        best_iter = int(np.argmin(mse_means))
        improvement = (mse_means[0] / mse_means[best_iter]
                       if mse_means[best_iter] > 0 else 0)

        mono = 0
        for i in range(1, max_iters):
            if mse_means[i] < mse_means[i - 1]:
                mono = i
            else:
                break

        results[tag] = {
            "mse_per_iter": mse_means,
            "gn_cosine_per_iter": [None] + cosine_means,
            "best_iter": best_iter,
            "improvement": improvement,
            "mono_improving": mono,
        }

    model.train()
    return results


def print_results(results):
    for mode in ["undamped", "damped_0.5"]:
        r = results[mode]
        tag = "" if mode == "undamped" else " (damped 0.5)"
        print(f"\n  NLLS{tag}:")
        print(f"    {'iter':<6} {'MSE':<14} {'GN cos':<12}")
        print(f"    {'-'*35}")
        for i, mse in enumerate(r["mse_per_iter"]):
            marker = " <-- best" if i == r["best_iter"] else ""
            cos_str = (f"{r['gn_cosine_per_iter'][i]:.4f}"
                       if r['gn_cosine_per_iter'][i] is not None
                       else "  --")
            print(f"    {i:<6} {mse:.8f}   {cos_str}{marker}")
        print(f"    Improvement: {r['improvement']:.2f}x | "
              f"Best @ iter {r['best_iter']} | "
              f"Mono improving: {r['mono_improving']} steps")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Non-linear least squares — nonlinear refinement")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--training_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str,
                        default="results/nonlinear")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, trainer, train_time = train(args.device, args.training_steps)
    results = evaluate(model, trainer, torch.device(args.device))
    print_results(results)

    output = {
        "task": "NONLINEAR_LEAST_SQUARES",
        "desc": "4-param sigmoid fit, Gauss-Newton convergence",
        "train_time": train_time,
        "training_steps": args.training_steps,
        "results": results,
    }
    out_path = output_dir / "nlls_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
