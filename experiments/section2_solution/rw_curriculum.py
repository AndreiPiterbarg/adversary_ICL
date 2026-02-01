"""
Curriculum residual_weight: start low (strong initial predictor) then increase
(strong correction pathway).

Hypothesis: rw=0.8 works because the model focuses on correction. But it may sacrifice
initial prediction quality. A curriculum schedule could give both: good initial predictor
early, then shift to correction mastery.

Schedules tested:
1. LINEAR_0.3_0.9    - Linear ramp from 0.3 to 0.9 over full training
2. STEP_0.5_0.8_25K  - 0.5 for first 25k, then 0.8
3. COSINE_0.3_0.9    - Cosine ramp from 0.3 to 0.9
4. LINEAR_0.5_1.0    - Linear ramp from 0.5 to 1.0 (aggressive)
5. STEP_0.3_0.9_20K  - 0.3 for first 20k, then 0.9 (sharp shift)

Usage:
    python experiments/section2_solution/rw_curriculum.py --device cuda
    python experiments/section2_solution/rw_curriculum.py --device cuda --variants 1 2 3
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


def rw_schedule_linear(step, total_steps, rw_start, rw_end):
    progress = min(step / total_steps, 1.0)
    return rw_start + (rw_end - rw_start) * progress


def rw_schedule_step(step, total_steps, rw_start, rw_end, switch_step):
    return rw_start if step < switch_step else rw_end


def rw_schedule_cosine(step, total_steps, rw_start, rw_end):
    progress = min(step / total_steps, 1.0)
    # Cosine that goes from 0 to 1 (half cosine)
    t = 0.5 * (1 - np.cos(np.pi * progress))
    return rw_start + (rw_end - rw_start) * t


VARIANTS = {
    1: {
        "name": "LINEAR_0.3_0.9",
        "desc": "Linear ramp 0.3 -> 0.9",
        "schedule": "linear",
        "rw_start": 0.3, "rw_end": 0.9,
    },
    2: {
        "name": "STEP_0.5_0.8_25K",
        "desc": "Step: 0.5 for 25k, then 0.8",
        "schedule": "step",
        "rw_start": 0.5, "rw_end": 0.8, "switch_step": 25000,
    },
    3: {
        "name": "COSINE_0.3_0.9",
        "desc": "Cosine ramp 0.3 -> 0.9",
        "schedule": "cosine",
        "rw_start": 0.3, "rw_end": 0.9,
    },
    4: {
        "name": "LINEAR_0.5_1.0",
        "desc": "Linear ramp 0.5 -> 1.0 (aggressive)",
        "schedule": "linear",
        "rw_start": 0.5, "rw_end": 1.0,
    },
    5: {
        "name": "STEP_0.3_0.9_20K",
        "desc": "Step: 0.3 for 20k, then 0.9 (sharp shift)",
        "schedule": "step",
        "rw_start": 0.3, "rw_end": 0.9, "switch_step": 20000,
    },
}


def get_rw(variant, step, total_steps):
    schedule = variant["schedule"]
    if schedule == "linear":
        return rw_schedule_linear(step, total_steps, variant["rw_start"], variant["rw_end"])
    elif schedule == "step":
        return rw_schedule_step(step, total_steps, variant["rw_start"], variant["rw_end"],
                                variant["switch_step"])
    elif schedule == "cosine":
        return rw_schedule_cosine(step, total_steps, variant["rw_start"], variant["rw_end"])
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def run_variant(variant_id, variant, device_str):
    device = torch.device(device_str)
    name = variant["name"]
    d = 4

    print(f"\n{'#'*60}")
    print(f"# VARIANT {variant_id}: {name}")
    print(f"# {variant['desc']}")
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

    rw_log = []
    model.train()
    start = time.time()
    for step in range(total_steps):
        rw = get_rw(variant, step, total_steps)

        losses = train_step_standard(
            model, trainer, optimizer, config,
            alpha_power=0.5,
            residual_weight=rw,
            noise_scale=0.5,
        )
        scheduler.step()

        if step % 5000 == 0:
            elapsed = time.time() - start
            rw_log.append({"step": step, "rw": rw})
            print(f"  Step {step:6d}/{total_steps} | rw={rw:.3f} | "
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
        "rw_log": rw_log,
        "results": results,
        "results_damped": results_damped,
    }


def main():
    parser = argparse.ArgumentParser(description="Residual weight curriculum")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=int, nargs="*", default=None,
                        help="Which variants to run (1-5). Default: all")
    parser.add_argument("--output_dir", type=str, default="results/direction1")
    args = parser.parse_args()

    variant_ids = args.variants or list(VARIANTS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RESIDUAL WEIGHT CURRICULUM")
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

        with open(output_dir / "rw_curriculum_progress.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - RW CURRICULUM")
    print(f"{'='*60}")

    print(f"\n{'Variant':<22} {'kappa':<12} {'iter0':<10} {'best':<10} {'best@':<6} {'improv':<8} {'mono'}")
    print("-" * 80)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results"]
        for kappa_key, data in res.items():
            print(f"{name:<22} {kappa_key:<12} "
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
    print(f"\n{'Schedule':<22} {'avg_mono':<10} {'avg_improv':<12} {'max_mono':<10}")
    print("-" * 56)
    for vid in variant_ids:
        name = VARIANTS[vid]["name"]
        res = all_results[vid]["results"]
        avg_mono = np.mean([d["mono_improving_until"] for d in res.values()])
        avg_imp = np.mean([d["improvement"] for d in res.values()])
        max_mono = max(d["mono_improving_until"] for d in res.values())
        print(f"{name:<22} {avg_mono:<10.1f} {avg_imp:<12.2f}x {max_mono:<10}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    with open(output_dir / "rw_curriculum.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir / 'rw_curriculum.json'}")


if __name__ == "__main__":
    main()
