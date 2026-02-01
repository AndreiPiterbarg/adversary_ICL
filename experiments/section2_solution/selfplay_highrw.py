"""
Self-play + high residual_weight: combine the two best strategies from overnight experiments.

HIGH_RESID_WT (rw=0.8) gave 2.4x improvement with mono_improving up to 9 iterations.
SELF_PLAY gave good results on hard kappa ranges. Combining them may push further.

Variants:
1. SP20K_RW0.8       - Self-play at 20k, rw=0.8, 50k total
2. SP30K_RW0.8       - Self-play at 30k, rw=0.8, 50k total
3. SP20K_RW0.9       - Self-play at 20k, rw=0.9, 50k total
4. SP30K_RW0.8_75K   - Self-play at 30k, rw=0.8, 75k total (more training)

Usage:
    python experiments/section2_solution/selfplay_highrw.py --device cuda
    python experiments/section2_solution/selfplay_highrw.py --device cuda --variants 1 2
"""

import torch
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
    Trainer, make_model, generate_batch, train_step_standard,
    evaluate, print_eval,
)


VARIANTS = {
    1: {
        "name": "SP20K_RW0.8",
        "desc": "Self-play at 20k, rw=0.8, 50k total",
        "residual_weight": 0.8,
        "self_play_start": 20000,
        "training_steps": 50000,
    },
    2: {
        "name": "SP30K_RW0.8",
        "desc": "Self-play at 30k, rw=0.8, 50k total",
        "residual_weight": 0.8,
        "self_play_start": 30000,
        "training_steps": 50000,
    },
    3: {
        "name": "SP20K_RW0.9",
        "desc": "Self-play at 20k, rw=0.9, 50k total",
        "residual_weight": 0.9,
        "self_play_start": 20000,
        "training_steps": 50000,
    },
    4: {
        "name": "SP30K_RW0.8_75K",
        "desc": "Self-play at 30k, rw=0.8, 75k total",
        "residual_weight": 0.8,
        "self_play_start": 30000,
        "training_steps": 75000,
    },
}


def run_variant(variant_id, variant, device_str):
    device = torch.device(device_str)
    name = variant["name"]
    rw = variant["residual_weight"]
    sp_start = variant["self_play_start"]
    total_steps = variant["training_steps"]
    d = 4

    print(f"\n{'#'*60}")
    print(f"# VARIANT {variant_id}: {name}")
    print(f"# {variant['desc']}")
    print(f"{'#'*60}")

    model = make_model(d=d, n_embd=128, n_layer=6, n_head=4, device=device_str)
    trainer = Trainer(model, d, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

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
        use_self_play = step >= sp_start

        losses = train_step_standard(
            model, trainer, optimizer, config,
            alpha_power=0.5,
            residual_weight=rw,
            noise_scale=0.5,
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
    parser = argparse.ArgumentParser(description="Self-play + high residual weight")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=int, nargs="*", default=None,
                        help="Which variants to run (1-4). Default: all")
    parser.add_argument("--output_dir", type=str, default="results/direction1")
    args = parser.parse_args()

    variant_ids = args.variants or list(VARIANTS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SELF-PLAY + HIGH RESIDUAL WEIGHT")
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

        with open(output_dir / "selfplay_highrw_progress.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - SELF-PLAY + HIGH RW")
    print(f"{'='*60}")

    print(f"\n{'Variant':<20} {'kappa':<12} {'iter0':<10} {'best':<10} {'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 78)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results"]
        for kappa_key, data in res.items():
            print(f"{name:<20} {kappa_key:<12} "
                  f"{data['mse_per_iter'][0]:.6f}   "
                  f"{data['mse_per_iter'][data['best_iter']]:.6f}   "
                  f"{data['best_iter']:<6} "
                  f"{data['improvement']:.2f}x    "
                  f"{data['mono_improving_until']}")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Variant':<20} {'avg_mono':<10} {'avg_improv':<12} {'max_mono':<10}")
    print("-" * 54)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        max_mono = max(d["mono_improving_until"] for d in res.values())
        print(f"{name:<20} {avg_mono:<10.1f} {avg_imp:<12.2f}x {max_mono:<10}")

    # Compare damped variants
    print(f"\n{'Variant (damped)':<20} {'avg_mono':<10} {'avg_improv':<12} {'max_mono':<10}")
    print("-" * 54)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results_damped"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        max_mono = max(d["mono_improving_until"] for d in res.values())
        print(f"{name:<20} {avg_mono:<10.1f} {avg_imp:<12.2f}x {max_mono:<10}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "selfplay_highrw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'selfplay_highrw.json'}")


if __name__ == "__main__":
    main()
