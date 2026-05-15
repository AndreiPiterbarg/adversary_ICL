"""Fitness function for the adversary.

Two objective modes:

  "penalty" (legacy):
      fitness = T_glitch - lambda_lstm * max(0, LSTM_glitch - lstm_tolerance)
  "regret" (default for the quick configs):
      fitness = T_glitch - LSTM_glitch

The regret form is the excess-error / gap-to-skyline objective (PAIRED-style
regret; RHO-LOSS reducible loss). It does not reward distributions that are
merely hard for *this* checkpoint when the LSTM (Bayes-feasible skyline) also
struggles, and it gives zero credit to ill-posed regions.

Both modes apply a hard read-density feasibility gate: a candidate whose
expected read positions per sequence is below `min_reads_per_seq` is rejected
(is_valid=False). This blocks the degenerate `p_r -> 0` exploit where the
glitch metric is estimated over ~1 read position per sequence. The floor is
kept low (≈3) so genuinely sparse-but-valid tails like FFL(0.98) (~3.5
reads/seq) still pass; the regret term does the real selection work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data import R
from ..eval import evaluate_dataset
from .distribution import FFLDistribution


@dataclass
class FitnessResult:
    fitness: float
    T_glitch: float
    lstm_glitch: float
    read_density: float
    n_samples: int
    seed: int
    is_valid: bool = True


def _glitch_rate(model, tokens, batch_size, device) -> float:
    return evaluate_dataset(model, tokens, batch_size=batch_size, device=device)["error_rate"]


def fitness(
    dist: FFLDistribution,
    transformer,
    lstm,
    *,
    n: int,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    lambda_lstm: float = 10.0,
    lstm_tolerance: float = 1e-3,
    objective_mode: str = "penalty",
    min_reads_per_seq: float = 0.0,
) -> FitnessResult:
    """Sample n sequences from `dist`, score on both models, return FitnessResult."""
    seed = int(rng.integers(0, 2**31 - 1))
    sample_rng = np.random.default_rng(seed)
    try:
        tokens = dist.sample(n, sample_rng)
    except AssertionError:
        return FitnessResult(fitness=float("-inf"), T_glitch=0.0, lstm_glitch=0.0,
                             read_density=0.0, n_samples=n, seed=seed, is_valid=False)

    # Expected read positions per sequence, from the already-sampled tokens
    # (no extra forward pass). The clean-mode metric/loss only supervises reads,
    # so a near-zero-read distribution is degenerate regardless of T_glitch.
    read_density = float((tokens == R).sum(dim=1).float().mean().item())
    if min_reads_per_seq > 0.0 and read_density < min_reads_per_seq:
        return FitnessResult(fitness=float("-inf"), T_glitch=0.0, lstm_glitch=0.0,
                             read_density=read_density, n_samples=n, seed=seed,
                             is_valid=False)

    t_glitch = _glitch_rate(transformer, tokens, batch_size, device)
    l_glitch = _glitch_rate(lstm, tokens, batch_size, device) if lstm is not None else 0.0

    if objective_mode == "regret":
        # Excess error over the Bayes-feasible skyline. Degenerate / ill-posed
        # regions (LSTM also fails) get ~0 credit by construction.
        score = t_glitch - l_glitch
    else:
        score = t_glitch - lambda_lstm * max(0.0, l_glitch - lstm_tolerance)

    return FitnessResult(
        fitness=float(score),
        T_glitch=float(t_glitch),
        lstm_glitch=float(l_glitch),
        read_density=read_density,
        n_samples=n,
        seed=seed,
        is_valid=True,
    )


def seed_averaged_fitness(
    dist: FFLDistribution,
    transformer,
    lstm,
    *,
    n: int,
    batch_size: int,
    device: str,
    n_seeds: int = 3,
    base_rng: Optional[np.random.Generator] = None,
    lambda_lstm: float = 10.0,
    lstm_tolerance: float = 1e-3,
    objective_mode: str = "penalty",
    min_reads_per_seq: float = 0.0,
) -> FitnessResult:
    """Average fitness over n_seeds independent data draws.

    Use during final evaluation of top-K to de-noise the ranking (Fig 7 shows
    both data and model seeds matter materially).
    """
    base_rng = base_rng or np.random.default_rng(0)
    results = [
        fitness(dist, transformer, lstm, n=n, batch_size=batch_size, device=device,
                rng=base_rng, lambda_lstm=lambda_lstm, lstm_tolerance=lstm_tolerance,
                objective_mode=objective_mode, min_reads_per_seq=min_reads_per_seq)
        for _ in range(n_seeds)
    ]
    valid = [r for r in results if r.is_valid]
    if not valid:
        return results[0]
    return FitnessResult(
        fitness=float(np.mean([r.fitness for r in valid])),
        T_glitch=float(np.mean([r.T_glitch for r in valid])),
        lstm_glitch=float(np.mean([r.lstm_glitch for r in valid])),
        read_density=float(np.mean([r.read_density for r in valid])),
        n_samples=n * n_seeds,
        seed=-1,
        is_valid=True,
    )
