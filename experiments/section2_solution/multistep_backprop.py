"""
Multi-step backprop: run K refinement steps during training and backprop through ALL of them.

Key ideas:
- Don't detach intermediate predictions during training
- Loss = weighted sum across iterations (later iterations weighted more)
- Iteration embedding tells the model which refinement step it's on
- Test whether the model generalizes beyond K at test time

Uses residual_weight=0.8 as the base (best from overnight experiments).

Variants:
1. K3_UNIFORM   - 3 steps, uniform loss weights
2. K3_WEIGHTED  - 3 steps, increasing weights [0.1, 0.3, 0.6]
3. K5_WEIGHTED  - 5 steps, increasing weights [0.05, 0.1, 0.15, 0.3, 0.4]
4. K5_LAST_ONLY - 5 steps, only loss on last step

Usage:
    python experiments/section2_solution/multistep_backprop.py --device cuda
    python experiments/section2_solution/multistep_backprop.py --device cuda --variants 1 2
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
from data.spd_sampler import sample_spd

_exp_dir = Path(__file__).parent
sys.path.insert(0, str(_exp_dir))
from overnight_theories import (
    Trainer, generate_batch, evaluate, print_eval,
)


VARIANTS = {
    1: {
        "name": "K3_UNIFORM",
        "desc": "3 refinement steps, uniform loss weights",
        "K": 3,
        "weights": [1.0/3, 1.0/3, 1.0/3],
    },
    2: {
        "name": "K3_WEIGHTED",
        "desc": "3 refinement steps, increasing weights",
        "K": 3,
        "weights": [0.1, 0.3, 0.6],
    },
    3: {
        "name": "K5_WEIGHTED",
        "desc": "5 refinement steps, increasing weights",
        "K": 5,
        "weights": [0.05, 0.1, 0.15, 0.3, 0.4],
    },
    4: {
        "name": "K5_LAST_ONLY",
        "desc": "5 refinement steps, loss only on last step",
        "K": 5,
        "weights": [0.0, 0.0, 0.0, 0.0, 1.0],
    },
}


def make_model_with_iter(d=4, n_embd=128, n_layer=6, n_head=4,
                         max_iterations=6, device='cuda'):
    """Create model with iteration embedding enabled."""
    model_config = ComponentModelConfig(
        d=d, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        n_positions=128, max_examples=64, dropout=0.0,
        max_iterations=max_iterations,
    )
    return ComponentTransformerModel(model_config).to(device)


def train_step_multistep(model, trainer, optimizer, K, loss_weights,
                         residual_weight=0.8, noise_scale=0.5):
    """Training step with multi-step backprop through K refinement iterations.

    Does NOT detach intermediate predictions, allowing gradients to flow
    through the full refinement chain.
    """
    B, num_ctx, d = 64, 5, 4
    device = next(model.parameters()).device

    A, b_ctx, x_ctx, b_query, x_target = generate_batch(
        B, num_ctx, d, device, 1.0, 100.0
    )

    total_loss = 0.0
    losses = {}

    # Step 0: Direct prediction (iteration_index=0)
    tokens, ex_pos, mask_pos = trainer.build_standard(A, b_ctx, x_ctx, b_query)
    output = model(tokens, ex_pos, mask_pos, iteration_index=0)
    pred_0 = output.vector_output  # Keep gradient attached
    loss_direct = F.mse_loss(pred_0, x_target)
    total_loss = (1 - residual_weight) * loss_direct
    losses["direct"] = loss_direct.item()

    # Multi-step refinement with backprop through all steps
    x_current = pred_0  # Gradient flows through this
    for k in range(K):
        # Build sequence with current estimate (NOT detached)
        tokens_r, ep, mp = trainer.build_with_estimate(
            A, b_ctx, x_ctx, b_query, x_current
        )
        # iteration_index = k+1 (step 0 was the direct prediction)
        output_r = model(tokens_r, ep, mp, iteration_index=k + 1)
        correction = output_r.vector_output

        # Apply correction (gradient flows through)
        x_current = x_current + correction

        # Loss at this iteration
        loss_k = F.mse_loss(x_current, x_target)
        weight_k = loss_weights[k] * residual_weight
        total_loss = total_loss + weight_k * loss_k
        losses[f"step_{k+1}"] = loss_k.item()

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses["total"] = total_loss.item()
    return losses


def evaluate_with_iter(model, device, d=4, num_context=5, max_iters=15,
                       test_batches=30, kappa_ranges=None, damping=1.0):
    """Evaluate with iteration embeddings. Tests generalization beyond training K."""
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
                x_current = model(tokens, ex_pos, mask_pos, iteration_index=0).vector_output
                all_mse[0].append(F.mse_loss(x_current, x_target).item())

                for i in range(1, max_iters):
                    tokens_r, ep, mp = trainer.build_with_estimate(
                        A, b_ctx, x_ctx, b_query, x_current
                    )
                    correction = model(tokens_r, ep, mp, iteration_index=i).vector_output
                    x_current = x_current + damping * correction
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

        results[kappa_key] = {
            "mse_per_iter": mse_means,
            "best_iter": best_iter,
            "improvement": improvement,
            "mono_improving_until": mono_improving,
        }

    model.train()
    return results


def run_variant(variant_id, variant, device_str):
    device = torch.device(device_str)
    name = variant["name"]
    K = variant["K"]
    weights = variant["weights"]
    d = 4

    print(f"\n{'#'*60}")
    print(f"# VARIANT {variant_id}: {name}")
    print(f"# {variant['desc']}")
    print(f"# K={K}, weights={weights}")
    print(f"{'#'*60}")

    # max_iterations = K+1 (step 0 = direct, steps 1..K = refinement)
    model = make_model_with_iter(
        d=d, n_embd=128, n_layer=6, n_head=4,
        max_iterations=K + 1, device=device_str,
    )
    trainer = Trainer(model, d, device)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    total_steps = 50000
    warmup = 1000

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    start = time.time()
    for step in range(total_steps):
        losses = train_step_multistep(
            model, trainer, optimizer, K, weights,
            residual_weight=0.8, noise_scale=0.5,
        )
        scheduler.step()

        if step % 5000 == 0:
            elapsed = time.time() - start
            step_losses = " | ".join(
                f"s{k+1}={losses.get(f'step_{k+1}', 0):.6f}" for k in range(K)
            )
            print(f"  Step {step:6d}/{total_steps} | "
                  f"total={losses['total']:.6f} | "
                  f"direct={losses['direct']:.6f} | "
                  f"{step_losses} | {elapsed:.0f}s")

    train_time = time.time() - start
    print(f"  Training done in {train_time:.0f}s")

    # Evaluate with iteration embeddings, max_iters=15 to test beyond K
    print(f"\n  Evaluating (with iteration embeddings, max_iters=15)...")
    results = evaluate_with_iter(model, device, d=d, max_iters=15, damping=1.0)
    print_eval(results, name)

    results_damped = evaluate_with_iter(model, device, d=d, max_iters=15, damping=0.5)
    print(f"\n  (Also with damping=0.5:)")
    print_eval(results_damped, name + " (damped)")

    # Also evaluate WITHOUT iteration embeddings to see if they matter
    print(f"\n  Evaluating (WITHOUT iteration embeddings)...")
    results_no_iter = evaluate(model, device, d=d, max_iters=15, damping=1.0)
    print_eval(results_no_iter, name + " (no iter emb)")

    return {
        "variant": variant,
        "train_time": train_time,
        "results_with_iter": results,
        "results_with_iter_damped": results_damped,
        "results_no_iter": results_no_iter,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-step backprop experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=int, nargs="*", default=None,
                        help="Which variants to run (1-4). Default: all")
    parser.add_argument("--output_dir", type=str, default="results/direction1")
    args = parser.parse_args()

    variant_ids = args.variants or list(VARIANTS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MULTI-STEP BACKPROP EXPERIMENT")
    print("=" * 60)
    print(f"Variants: {variant_ids}")
    print(f"Device: {args.device}")
    for vid in variant_ids:
        v = VARIANTS[vid]
        print(f"  {vid}. {v['name']}: {v['desc']}")

    all_results = {}
    total_start = time.time()

    for vid in variant_ids:
        variant = VARIANTS[vid]
        result = run_variant(vid, variant, args.device)
        all_results[vid] = result

        with open(output_dir / "multistep_backprop_progress.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - MULTI-STEP BACKPROP")
    print(f"{'='*60}")

    print(f"\n{'Variant':<16} {'mode':<12} {'kappa':<12} {'iter0':<10} {'best':<10} "
          f"{'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 86)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        K = VARIANTS[vid]["K"]
        for mode, key in [("iter_emb", "results_with_iter"), ("no_iter", "results_no_iter")]:
            res = all_results[vid][key]
            for kappa_key, data in res.items():
                print(f"{name:<16} {mode:<12} {kappa_key:<12} "
                      f"{data['mse_per_iter'][0]:.6f}   "
                      f"{data['mse_per_iter'][data['best_iter']]:.6f}   "
                      f"{data['best_iter']:<6} "
                      f"{data['improvement']:.2f}x    "
                      f"{data['mono_improving_until']}")
            print()

    # Generalization analysis: do models improve beyond training K?
    print(f"\n{'='*60}")
    print("GENERALIZATION BEYOND TRAINING K")
    print(f"{'='*60}")
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        K = VARIANTS[vid]["K"]
        res = all_results[vid]["results_with_iter"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        max_mono = max(d["mono_improving_until"] for d in res.values())
        generalizes = "YES" if max_mono > K else "no"
        print(f"  {name:<16} K={K}  avg_mono={avg_mono:.1f}  avg_improv={avg_imp:.2f}x  "
              f"max_mono={max_mono}  generalizes_beyond_K={generalizes}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "multistep_backprop.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'multistep_backprop.json'}")


if __name__ == "__main__":
    main()
