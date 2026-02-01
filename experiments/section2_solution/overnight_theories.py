"""
Overnight test suite: try different theories to get true multi-step iterative refinement.

Core problem: the model does 1 good correction step, then outputs ~0 correction.
It can't distinguish "close but not at x*" from "at x*".

Theories:
1. BASELINE        - Current approach (alpha^0.5, rw=0.5) as reference
2. HIGH_RESID_WT   - residual_weight=0.8 (stronger correction pathway)
3. ALPHA_UNIFORM   - alpha^1.0 (uniform distribution, more near-x* training)
4. ALPHA_SHARP     - alpha^0.3 (heavily biased toward pred_0)
5. SELF_PLAY       - Phase 1: normal 30k, Phase 2: 20k on model's own step-1 output
6. DAMPED_TEST     - Same as baseline but test with x += 0.5 * correction (no retrain)
7. BIGGER_MODEL    - n_layer=10, n_embd=192
8. LONGER_TRAIN    - 100k steps instead of 50k

Usage:
    python experiments/section2_solution/overnight_theories.py --device cuda
    python experiments/section2_solution/overnight_theories.py --device cuda --theories 1 5 6
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
# Training infrastructure (reuses existing logic)
# ============================================================

class Trainer:
    """Minimal trainer that supports all theory variants."""

    def __init__(self, model, d, device, role_indices=None):
        self.model = model
        self.d = d
        self.device = device
        self._role_indices = role_indices or {
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


def make_model(d=4, n_embd=128, n_layer=6, n_head=4, device='cuda'):
    model_config = ComponentModelConfig(
        d=d, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        n_positions=128, max_examples=64, dropout=0.0
    )
    return ComponentTransformerModel(model_config).to(device)


def generate_batch(B, K, d, device, kappa_min=1.0, kappa_max=100.0):
    A = sample_spd(B, d, device, kappa_min, kappa_max)
    b_all = torch.randn(B, K + 1, d, device=device)
    x_all = torch.linalg.solve(A, b_all.transpose(-2, -1)).transpose(-2, -1)
    return A, b_all[:, :K], x_all[:, :K], b_all[:, K], x_all[:, K]


# ============================================================
# Training step variants
# ============================================================

def train_step_standard(model, trainer, optimizer, config, alpha_power=0.5,
                        residual_weight=0.5, noise_scale=0.5, self_play=False):
    """One training step. Supports all theory variants via parameters."""
    B, K, d = 64, 5, 4
    device = next(model.parameters()).device

    A, b_ctx, x_ctx, b_query, x_target = generate_batch(
        B, K, d, device, config.get('kappa_min', 1.0), config.get('kappa_max', 100.0)
    )

    total_loss = 0.0
    losses = {}

    # Part 1: Direct prediction
    tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
    output = model(tokens, ex_pos, mask_pos)
    pred_0 = output.vector_output
    loss_direct = F.mse_loss(pred_0, x_target)
    total_loss = (1 - residual_weight) * loss_direct
    losses["direct"] = loss_direct.item()

    # Part 2: Residual prediction
    if residual_weight > 0:
        with torch.no_grad():
            if self_play:
                # Use model's own step-1 output as estimate
                tokens_r, ep, mp = trainer.build_with_estimate(
                    A, b_ctx, x_ctx, b_query, pred_0.detach()
                )
                residual_0 = model(tokens_r, ep, mp).vector_output
                x_estimate = (pred_0 + residual_0).detach()
                # Add small noise for robustness
                x_estimate = x_estimate + torch.randn_like(x_estimate) * noise_scale * 0.1
            else:
                alpha = torch.rand(B, 1, device=device) ** alpha_power
                noise = torch.randn_like(pred_0) * (
                    torch.rand(B, 1, device=device) * noise_scale
                )
                x_estimate = alpha * pred_0.detach() + (1 - alpha) * x_target + noise

            true_residual = x_target - x_estimate

        tokens_r, ex_pos_r, mask_pos_r = trainer.build_with_estimate(
            A, b_ctx, x_ctx, b_query, x_estimate
        )
        output_r = model(tokens_r, ex_pos_r, mask_pos_r)
        pred_residual = output_r.vector_output
        loss_residual = F.mse_loss(pred_residual, true_residual)
        total_loss = total_loss + residual_weight * loss_residual
        losses["residual"] = loss_residual.item()

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses["total"] = total_loss.item()
    return losses


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, device, d=4, num_context=5, max_iters=10, test_batches=30,
             kappa_ranges=None, damping=1.0):
    """Evaluate model refinement. Returns per-kappa MSE curves."""
    if kappa_ranges is None:
        kappa_ranges = [(1, 10), (10, 50), (50, 100), (100, 200)]

    model.eval()
    trainer = Trainer(model, d, device)
    results = {}

    for kappa_min, kappa_max in kappa_ranges:
        kappa_key = f"{kappa_min}-{kappa_max}"
        all_mse = {i: [] for i in range(max_iters)}

        for _ in range(test_batches):
            B, K = 64, num_context
            with torch.no_grad():
                A, b_ctx, x_ctx, b_query, x_target = generate_batch(
                    B, K, d, device, kappa_min, kappa_max
                )

                tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
                x_current = model(tokens, ex_pos, mask_pos).vector_output
                all_mse[0].append(F.mse_loss(x_current, x_target).item())

                for i in range(1, max_iters):
                    tokens_r, ep, mp = trainer.build_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    correction = model(tokens_r, ep, mp).vector_output
                    x_current = x_current + damping * correction
                    all_mse[i].append(F.mse_loss(x_current, x_target).item())

        mse_means = [np.mean(all_mse[i]) for i in range(max_iters)]
        best_iter = int(np.argmin(mse_means))
        improvement = mse_means[0] / mse_means[best_iter] if mse_means[best_iter] > 0 else 0

        # Check if monotonically improving for first N steps
        mono_improving = 0
        for i in range(1, max_iters):
            if mse_means[i] < mse_means[i - 1]:
                mono_improving = i
            else:
                break

        results[kappa_key] = {
            "mse_per_iter": mse_means,
            "best_iter": best_iter,
            "improvement": improvement,
            "mono_improving_until": mono_improving,
        }

    model.train()
    return results


def print_eval(results, theory_name):
    print(f"\n  {'kappa':<12} {'iter0 MSE':<12} {'best MSE':<12} {'best@':<6} {'improv':<8} {'mono':<6}")
    print(f"  {'-'*56}")
    for kappa_key, data in results.items():
        print(f"  {kappa_key:<12} {data['mse_per_iter'][0]:.6f}     "
              f"{data['mse_per_iter'][data['best_iter']]:.6f}     "
              f"{data['best_iter']:<6} {data['improvement']:.2f}x    "
              f"{data['mono_improving_until']}")


# ============================================================
# Theory definitions
# ============================================================

THEORIES = {
    1: {
        "name": "BASELINE",
        "desc": "Current approach (alpha^0.5, rw=0.5)",
        "alpha_power": 0.5,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    2: {
        "name": "HIGH_RESID_WT",
        "desc": "residual_weight=0.8 (stronger correction pathway)",
        "alpha_power": 0.5,
        "residual_weight": 0.8,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    3: {
        "name": "ALPHA_UNIFORM",
        "desc": "alpha^1.0 (uniform, more near-x* training)",
        "alpha_power": 1.0,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    4: {
        "name": "ALPHA_SHARP",
        "desc": "alpha^0.3 (heavily biased toward pred_0)",
        "alpha_power": 0.3,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    5: {
        "name": "SELF_PLAY",
        "desc": "Phase 1: 30k normal, Phase 2: 20k on model's own outputs",
        "alpha_power": 0.5,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "self_play_phase2_start": 30000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    6: {
        "name": "DAMPED_TEST",
        "desc": "Same as baseline but test with x += 0.5 * correction",
        "alpha_power": 0.5,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "test_damping": 0.5,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
    7: {
        "name": "BIGGER_MODEL",
        "desc": "n_layer=10, n_embd=192 (more capacity)",
        "alpha_power": 0.5,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 50000,
        "n_layer": 10,
        "n_embd": 192,
        "n_head": 4,
    },
    8: {
        "name": "LONGER_TRAIN",
        "desc": "100k steps instead of 50k",
        "alpha_power": 0.5,
        "residual_weight": 0.5,
        "noise_scale": 0.5,
        "training_steps": 100000,
        "n_layer": 6,
        "n_embd": 128,
        "n_head": 4,
    },
}


# ============================================================
# Main runner
# ============================================================

def run_theory(theory_id, theory, device_str):
    device = torch.device(device_str)
    name = theory["name"]
    d = 4

    print(f"\n{'#'*60}")
    print(f"# THEORY {theory_id}: {name}")
    print(f"# {theory['desc']}")
    print(f"{'#'*60}")

    # Create model
    model = make_model(
        d=d,
        n_embd=theory["n_embd"],
        n_layer=theory["n_layer"],
        n_head=theory["n_head"],
        device=device_str,
    )
    trainer = Trainer(model, d, device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # LR schedule
    total_steps = theory["training_steps"]
    warmup = 1000

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    self_play_start = theory.get("self_play_phase2_start", None)
    config = {"kappa_min": 1.0, "kappa_max": 100.0}

    # Train
    model.train()
    start = time.time()
    for step in range(total_steps):
        use_self_play = self_play_start is not None and step >= self_play_start

        losses = train_step_standard(
            model, trainer, optimizer, config,
            alpha_power=theory["alpha_power"],
            residual_weight=theory["residual_weight"],
            noise_scale=theory["noise_scale"],
            self_play=use_self_play,
        )
        scheduler.step()

        if step % 5000 == 0:
            phase = " [SELF-PLAY]" if use_self_play else ""
            elapsed = time.time() - start
            print(f"  Step {step:6d}/{total_steps} | "
                  f"total={losses['total']:.6f} | "
                  f"direct={losses['direct']:.6f} | "
                  f"residual={losses.get('residual', 0):.6f} | "
                  f"{elapsed:.0f}s{phase}")

    train_time = time.time() - start
    print(f"  Training done in {train_time:.0f}s")

    # Evaluate
    damping = theory.get("test_damping", 1.0)
    results = evaluate(model, device, d=d, damping=damping)
    print_eval(results, name)

    # Also test with damping=0.5 for all theories (cheap, informative)
    if damping == 1.0:
        results_damped = evaluate(model, device, d=d, damping=0.5)
        print(f"\n  (Also with damping=0.5:)")
        print_eval(results_damped, name + " (damped)")
    else:
        results_damped = None

    return {
        "theory": theory,
        "train_time": train_time,
        "results": results,
        "results_damped": {k: v for k, v in (results_damped or {}).items()},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Overnight theory testing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--theories", type=int, nargs="*", default=None,
                        help="Which theories to run (1-8). Default: all")
    parser.add_argument("--output_dir", type=str, default="results/overnight_theories")
    args = parser.parse_args()

    theory_ids = args.theories or list(THEORIES.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OVERNIGHT THEORY TESTING")
    print("=" * 60)
    print(f"Theories to test: {theory_ids}")
    print(f"Device: {args.device}")
    for tid in theory_ids:
        t = THEORIES[tid]
        print(f"  {tid}. {t['name']}: {t['desc']}")

    all_results = {}
    total_start = time.time()

    for tid in theory_ids:
        theory = THEORIES[tid]
        result = run_theory(tid, theory, args.device)
        all_results[tid] = result

        # Save after each theory (in case of crash)
        with open(output_dir / "results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    print(f"\n{'Theory':<20} {'kappa':<12} {'iter0':<10} {'best':<10} {'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 78)
    for tid in theory_ids:
        name = THEORIES[tid]["name"]
        res = all_results[tid]["results"]
        for kappa_key, data in res.items():
            print(f"{name:<20} {kappa_key:<12} "
                  f"{data['mse_per_iter'][0]:.6f}   "
                  f"{data['mse_per_iter'][data['best_iter']]:.6f}   "
                  f"{data['best_iter']:<6} "
                  f"{data['improvement']:.2f}x    "
                  f"{data['mono_improving_until']}")
        print()

    # Summary: which theory has best multi-step improvement?
    print(f"\n{'='*60}")
    print("WINNER ANALYSIS")
    print(f"{'='*60}")

    # Best by: most iterations of monotonic improvement
    print("\nBest by monotonic improvement depth:")
    for tid in theory_ids:
        name = THEORIES[tid]["name"]
        res = all_results[tid]["results"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        print(f"  {name:<20} avg_mono={avg_mono:.1f}  avg_improvement={avg_imp:.2f}x")

    # Check damped variants
    print("\nDamped (0.5) results:")
    for tid in theory_ids:
        name = THEORIES[tid]["name"]
        res_d = all_results[tid].get("results_damped", {})
        if res_d:
            avg_mono = np.mean([d["mono_improving_until"] for d in res_d.values()])
            avg_imp = np.mean([d["improvement"] for d in res_d.values()])
            print(f"  {name:<20} avg_mono={avg_mono:.1f}  avg_improvement={avg_imp:.2f}x")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
