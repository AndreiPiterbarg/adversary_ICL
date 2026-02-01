"""
Sweep residual_weight to find the phase transition for multi-step convergence.

The overnight experiment showed that rw=0.8 broke the single-step barrier (2.4x improvement
over 4-9 iterations). This script sweeps {0.6, 0.7, 0.8, 0.9, 0.95, 1.0} to find exactly
where multi-step convergence kicks in.

Usage:
    python experiments/section2_solution/rw_sweep.py --device cuda
    python experiments/section2_solution/rw_sweep.py --device cuda --variants 1 3 5
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict
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

# Reuse infrastructure from overnight_theories
_exp_dir = Path(__file__).parent
sys.path.insert(0, str(_exp_dir))
from overnight_theories import (
    Trainer, make_model, generate_batch, train_step_standard,
    evaluate, print_eval,
)


VARIANTS = {
    1: {"name": "RW_0.60", "residual_weight": 0.60},
    2: {"name": "RW_0.70", "residual_weight": 0.70},
    3: {"name": "RW_0.80", "residual_weight": 0.80},
    4: {"name": "RW_0.90", "residual_weight": 0.90},
    5: {"name": "RW_0.95", "residual_weight": 0.95},
    6: {"name": "RW_1.00", "residual_weight": 1.00},
}


def run_variant(variant_id, variant, device_str):
    device = torch.device(device_str)
    name = variant["name"]
    rw = variant["residual_weight"]
    d = 4

    print(f"\n{'#'*60}")
    print(f"# VARIANT {variant_id}: {name} (residual_weight={rw})")
    print(f"{'#'*60}")

    model = make_model(d=d, n_embd=128, n_layer=6, n_head=4, device=device_str)
    trainer = Trainer(model, d, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    total_steps = 50000
    warmup = 1000

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    config = {"kappa_min": 1.0, "kappa_max": 100.0}

    model.train()
    start = time.time()
    for step in range(total_steps):
        losses = train_step_standard(
            model, trainer, optimizer, config,
            alpha_power=0.5,
            residual_weight=rw,
            noise_scale=0.5,
        )
        scheduler.step()

        if step % 5000 == 0:
            elapsed = time.time() - start
            print(f"  Step {step:6d}/{total_steps} | "
                  f"total={losses['total']:.6f} | "
                  f"direct={losses['direct']:.6f} | "
                  f"residual={losses.get('residual', 0):.6f} | "
                  f"{elapsed:.0f}s")

    train_time = time.time() - start
    print(f"  Training done in {train_time:.0f}s")

    # Evaluate with max_iters=15
    results = evaluate(model, device, d=d, max_iters=15, damping=1.0)
    print_eval(results, name)

    results_damped = evaluate(model, device, d=d, max_iters=15, damping=0.5)
    print(f"\n  (Also with damping=0.5:)")
    print_eval(results_damped, name + " (damped)")

    return {
        "variant": variant,
        "train_time": train_time,
        "results": results,
        "results_damped": results_damped,
    }


def main():
    parser = argparse.ArgumentParser(description="Residual weight sweep")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=int, nargs="*", default=None,
                        help="Which variants to run (1-6). Default: all")
    parser.add_argument("--output_dir", type=str, default="results/direction1")
    args = parser.parse_args()

    variant_ids = args.variants or list(VARIANTS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RESIDUAL WEIGHT SWEEP")
    print("=" * 60)
    print(f"Variants: {variant_ids}")
    print(f"Device: {args.device}")
    for vid in variant_ids:
        v = VARIANTS[vid]
        print(f"  {vid}. {v['name']}: rw={v['residual_weight']}")

    all_results = {}
    total_start = time.time()

    for vid in variant_ids:
        variant = VARIANTS[vid]
        result = run_variant(vid, variant, args.device)
        all_results[vid] = result

        with open(output_dir / "rw_sweep_progress.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - RESIDUAL WEIGHT SWEEP")
    print(f"{'='*60}")

    print(f"\n{'Variant':<12} {'kappa':<12} {'iter0':<10} {'best':<10} {'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 70)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results"]
        for kappa_key, data in res.items():
            print(f"{name:<12} {kappa_key:<12} "
                  f"{data['mse_per_iter'][0]:.6f}   "
                  f"{data['mse_per_iter'][data['best_iter']]:.6f}   "
                  f"{data['best_iter']:<6} "
                  f"{data['improvement']:.2f}x    "
                  f"{data['mono_improving_until']}")
        print()

    # Phase transition analysis
    print(f"\n{'='*60}")
    print("PHASE TRANSITION ANALYSIS")
    print(f"{'='*60}")
    print(f"\n{'rw':<8} {'avg_mono':<10} {'avg_improv':<12} {'max_mono':<10}")
    print("-" * 42)
    for vid in variant_ids:
        rw = VARIANTS[vid]["residual_weight"]
        res = all_results[vid]["results"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        max_mono = max(d["mono_improving_until"] for d in res.values())
        print(f"{rw:<8.2f} {avg_mono:<10.1f} {avg_imp:<12.2f}x {max_mono:<10}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "rw_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'rw_sweep.json'}")


if __name__ == "__main__":
    main()
