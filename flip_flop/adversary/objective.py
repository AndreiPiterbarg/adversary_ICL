"""Fitness function for the adversary.

Two objective modes:

  "penalty" (legacy):
      fitness = T_glitch - lambda_lstm * max(0, LSTM_glitch - lstm_tolerance)
  "regret" (default for the quick configs; PLAN_beat_liu_r4 §1 final form):
      regret  = T_glitch - LSTM_glitch                       # excess over skyline
      fitness = regret - regret_hinge_coef
                         * max(0, LSTM_glitch - regret_hinge_tol)

The regret form is the excess-error / gap-to-skyline objective (PAIRED-style
regret; RHO-LOSS reducible loss). It does not reward distributions that are
merely hard for *this* checkpoint when the LSTM (Bayes-feasible skyline) also
struggles, and it gives zero credit to ill-posed regions. The soft hinge
(coefficient 5.0, tolerance 1e-3) is a near-boundary validity penalty that
preserves signal the old hard reject discarded.

Both modes apply hard feasibility gates on the *already-sampled* tokens (zero
extra forward passes):

  * read density (`min_reads_per_seq`): expected read positions per sequence.
    Blocks the degenerate `p_r -> 0` exploit. Floor ≈3 still admits FFL(0.98)
    (~3.5 reads/seq).
  * write→read gap (`min_gap`, PLAN §1): mean token distance from each read to
    its most-recent write. Floor ≈4 rejects trivially short-range
    distributions that are not glitches.

Two-checkpoint robustness (PLAN §2.D): if a second frozen transformer is
supplied, regret is computed against *both* checkpoints and the score is the
*minimum* of the two regrets (minus the shared LSTM hinge). A distribution is
only rewarded if it breaks both seeds — killing non-transferable, single-
checkpoint-idiosyncratic exploits. Costs +1 forward pass per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data import R, write_read_gaps
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
    gap: float = 0.0          # mean write→read gap (tokens)
    gap_p90: float = 0.0      # p90 write→read gap (tokens) — descriptor-grid axis
    T_glitch2: float = 0.0    # second-checkpoint glitch (§2.D); 0 if no transformer2
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
    min_gap: float = 0.0,
    regret_hinge_coef: float = 5.0,
    regret_hinge_tol: float = 1e-3,
    transformer2=None,
) -> FitnessResult:
    """Sample n sequences from `dist`, score on both models, return FitnessResult."""
    seed = int(rng.integers(0, 2**31 - 1))
    sample_rng = np.random.default_rng(seed)

    def _reject(read_density=0.0, gap=0.0, gap_p90=0.0):
        return FitnessResult(
            fitness=float("-inf"), T_glitch=0.0, lstm_glitch=0.0,
            read_density=read_density, n_samples=n, seed=seed,
            gap=gap, gap_p90=gap_p90, T_glitch2=0.0, is_valid=False,
        )

    try:
        tokens = dist.sample(n, sample_rng)
    except AssertionError:
        return _reject()

    # Free token-level descriptors (no forward pass). The clean-mode metric
    # only supervises reads, so a near-zero-read distribution is degenerate
    # regardless of T_glitch; a near-zero-gap one is trivially short-range.
    read_density = float((tokens == R).sum(dim=1).float().mean().item())
    gap_mean, gap_p90 = write_read_gaps(tokens)

    if min_reads_per_seq > 0.0 and read_density < min_reads_per_seq:
        return _reject(read_density, gap_mean, gap_p90)
    if min_gap > 0.0 and gap_mean < min_gap:
        return _reject(read_density, gap_mean, gap_p90)

    t_glitch = _glitch_rate(transformer, tokens, batch_size, device)
    l_glitch = _glitch_rate(lstm, tokens, batch_size, device) if lstm is not None else 0.0
    t_glitch2 = (_glitch_rate(transformer2, tokens, batch_size, device)
                 if transformer2 is not None else 0.0)

    if objective_mode == "regret":
        # Excess error over the Bayes-feasible skyline. With a second
        # checkpoint, require the gap to replicate on both seeds (min regret).
        regret = t_glitch - l_glitch
        if transformer2 is not None:
            regret = min(regret, t_glitch2 - l_glitch)
        score = regret - regret_hinge_coef * max(0.0, l_glitch - regret_hinge_tol)
    else:
        score = t_glitch - lambda_lstm * max(0.0, l_glitch - lstm_tolerance)

    return FitnessResult(
        fitness=float(score),
        T_glitch=float(t_glitch),
        lstm_glitch=float(l_glitch),
        read_density=read_density,
        n_samples=n,
        seed=seed,
        gap=float(gap_mean),
        gap_p90=float(gap_p90),
        T_glitch2=float(t_glitch2),
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
    min_gap: float = 0.0,
    regret_hinge_coef: float = 5.0,
    regret_hinge_tol: float = 1e-3,
    transformer2=None,
) -> FitnessResult:
    """Average fitness over n_seeds independent data draws.

    Use during final evaluation of top-K to de-noise the ranking (Fig 7 shows
    both data and model seeds matter materially).
    """
    base_rng = base_rng or np.random.default_rng(0)
    results = [
        fitness(dist, transformer, lstm, n=n, batch_size=batch_size, device=device,
                rng=base_rng, lambda_lstm=lambda_lstm, lstm_tolerance=lstm_tolerance,
                objective_mode=objective_mode, min_reads_per_seq=min_reads_per_seq,
                min_gap=min_gap, regret_hinge_coef=regret_hinge_coef,
                regret_hinge_tol=regret_hinge_tol, transformer2=transformer2)
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
        gap=float(np.mean([r.gap for r in valid])),
        gap_p90=float(np.mean([r.gap_p90 for r in valid])),
        T_glitch2=float(np.mean([r.T_glitch2 for r in valid])),
        is_valid=True,
    )
