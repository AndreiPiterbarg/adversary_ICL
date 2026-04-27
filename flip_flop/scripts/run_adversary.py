"""Run the FFLM adversary.

Usage (from repo root):
    python -m flip_flop.scripts.run_adversary \
        --config flip_flop/configs/adversary_stationary.yaml
    python -m flip_flop.scripts.run_adversary \
        --config flip_flop/configs/adversary_piecewise.yaml --test_run
"""
import argparse
import os

from flip_flop.adversary.run import AdversaryConfig, run_adversary

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "adversary_stationary.yaml",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    cfg = AdversaryConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.test_run:
        # Shrink everything to smoke-test size.
        cfg.search_n = 128
        cfg.final_eval_n = 256
        cfg.n_final_seeds = 1
        cfg.top_k = 5
        if cfg.strategy == "cma":
            cfg.budget = 32
            cfg.pop_size = 8
            cfg.num_restarts = 1
            cfg.K_segments = min(cfg.K_segments, 2)
        elif cfg.strategy == "grid" or cfg.strategy == "planted":
            # Trim each grid axis to its first two values.
            cfg.param_grid = {k: v[:2] for k, v in cfg.param_grid.items()}

    print("[done]", run_adversary(cfg))


if __name__ == "__main__":
    main()
