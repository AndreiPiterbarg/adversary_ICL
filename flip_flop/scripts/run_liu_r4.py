"""Train the Liu R4 control: fresh model on uniform {FFL(0.1), FFL(0.9), FFL(0.98)}.

Used as the paper-published baseline for cross-comparison with our automated
cumulative retrain loop.

Usage:
    python -m flip_flop.scripts.run_liu_r4
    python -m flip_flop.scripts.run_liu_r4 --test_run
"""
import argparse
import os

from flip_flop.adversary.r4_sampler import R4MixSampler, R4_PAPER_PIs
from flip_flop.train import TrainConfig, train

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "liu_r4.yaml",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.test_run:
        cfg.train_steps = 100
        cfg.decay_end_step = 101
        cfg.warmup_steps = 10
        cfg.eval_in_n = 64
        cfg.eval_sparse_n = 256
        cfg.eval_dense_n = 64
        cfg.eval_every = 50
        cfg.save_every = 0

    print(f"[liu_r4] training fresh model from scratch on uniform mixture "
          f"{R4_PAPER_PIs}; selection_enabled={cfg.selection_enabled}")
    sampler = R4MixSampler(T=cfg.seq_len, p_i_values=R4_PAPER_PIs)
    print("[done]", train(cfg, sampler=sampler))


if __name__ == "__main__":
    main()
