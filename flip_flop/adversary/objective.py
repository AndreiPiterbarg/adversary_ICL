"""Fitness function for the adversary.

Primary score: the frozen Transformer's glitch rate on samples from the
candidate distribution, penalized when the LSTM skyline also fails on the
same distribution (signal that the distribution is ill-posed, not adversarial).

    fitness = T_glitch - lambda_lstm * max(0, LSTM_glitch - lstm_tolerance)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..eval import evaluate_dataset
from .distribution import FFLDistribution


@dataclass
class FitnessResult:
    fitness: float
    T_glitch: float
    lstm_glitch: float
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
) -> FitnessResult:
    """Sample n sequences from `dist`, score on both models, return FitnessResult."""
    seed = int(rng.integers(0, 2**31 - 1))
    sample_rng = np.random.default_rng(seed)
    try:
        tokens = dist.sample(n, sample_rng)
    except AssertionError:
        return FitnessResult(fitness=float("-inf"), T_glitch=0.0, lstm_glitch=0.0,
                             n_samples=n, seed=seed, is_valid=False)

    t_glitch = _glitch_rate(transformer, tokens, batch_size, device)
    l_glitch = _glitch_rate(lstm, tokens, batch_size, device) if lstm is not None else 0.0
    penalty = lambda_lstm * max(0.0, l_glitch - lstm_tolerance)
    return FitnessResult(
        fitness=float(t_glitch - penalty),
        T_glitch=float(t_glitch),
        lstm_glitch=float(l_glitch),
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
) -> FitnessResult:
    """Average fitness over n_seeds independent data draws.

    Use during final evaluation of top-K to de-noise the ranking (Fig 7 shows
    both data and model seeds matter materially).
    """
    base_rng = base_rng or np.random.default_rng(0)
    results = [
        fitness(dist, transformer, lstm, n=n, batch_size=batch_size, device=device,
                rng=base_rng, lambda_lstm=lambda_lstm, lstm_tolerance=lstm_tolerance)
        for _ in range(n_seeds)
    ]
    valid = [r for r in results if r.is_valid]
    if not valid:
        return results[0]
    return FitnessResult(
        fitness=float(np.mean([r.fitness for r in valid])),
        T_glitch=float(np.mean([r.T_glitch for r in valid])),
        lstm_glitch=float(np.mean([r.lstm_glitch for r in valid])),
        n_samples=n * n_seeds,
        seed=-1,
        is_valid=True,
    )
