"""Search strategies: grid and diagonal-CMA-ES.

The CMA-ES class is a verbatim copy of src/adversary/search.py:DiagonalCMAES
to avoid coupling to the legacy Gaussian-regression tree. Minimal, ~100 lines,
self-contained.
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .distribution import FFLDistribution
from .objective import FitnessResult


@dataclass
class EvalResult:
    config: dict              # dist.to_dict()
    descriptor: dict          # dist.descriptor()
    fitness: float
    T_glitch: float
    lstm_glitch: float
    n_samples: int
    seed: int
    step: int = 0
    extra: dict = field(default_factory=dict)
    is_valid: bool = True

    @classmethod
    def from_fitness(cls, dist: FFLDistribution, fr: FitnessResult, step: int = 0,
                     **extra) -> "EvalResult":
        return cls(
            config=dist.to_dict(),
            descriptor=dist.descriptor(),
            fitness=fr.fitness,
            T_glitch=fr.T_glitch,
            lstm_glitch=fr.lstm_glitch,
            n_samples=fr.n_samples,
            seed=fr.seed,
            step=step,
            extra=extra,
            is_valid=fr.is_valid,
        )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def _cartesian(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def grid_search(
    dist_factory: Callable[[dict], FFLDistribution],
    param_grid: dict[str, list],
    objective_fn: Callable[[FFLDistribution], FitnessResult],
    out_dir: str,
    log_every: int = 10,
) -> list[EvalResult]:
    """Evaluate every point in the Cartesian product of param_grid."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "adversary_log.jsonl")
    candidates = _cartesian(param_grid)
    print(f"[grid] {len(candidates)} candidates over {list(param_grid)}")
    results: list[EvalResult] = []
    best = -float("inf")
    with open(log_path, "a") as fh:
        for step, params in enumerate(candidates):
            dist = dist_factory(params)
            fr = objective_fn(dist)
            r = EvalResult.from_fitness(dist, fr, step=step, params=params)
            results.append(r)
            fh.write(json.dumps(asdict(r)) + "\n")
            fh.flush()
            if r.fitness > best:
                best = r.fitness
            if (step + 1) % log_every == 0 or step == len(candidates) - 1:
                print(f"  [grid] {step + 1}/{len(candidates)} "
                      f"best_fitness={best:.4e} last_T_glitch={r.T_glitch:.4e} "
                      f"last_lstm={r.lstm_glitch:.4e}")
    return results


# ---------------------------------------------------------------------------
# Diagonal CMA-ES  (copied from src/adversary/search.py)
# ---------------------------------------------------------------------------
class DiagonalCMAES:
    """Minimal diagonal CMA-ES (sep-CMA-ES).

    Maintains only diagonal covariance (O(d) memory). Good enough for the
    low-dim spaces we search here. Minimizes; we negate fitness externally.
    """

    def __init__(self, x0: np.ndarray, sigma: float, pop_size: int, seed: int = 0):
        self.d = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)

        self.C_diag = np.ones(self.d)
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.d_sigma = (1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1)
                        + self.c_sigma)
        self.c_c = (4 + self.mu_eff / self.d) / (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.d + 2) ** 2 + self.mu_eff),
        )
        self.chi_d = np.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))
        self.mu = mu
        self.generation = 0

    def ask(self) -> list[np.ndarray]:
        std = self.sigma * np.sqrt(self.C_diag)
        return [self.mean + std * self.rng.standard_normal(self.d)
                for _ in range(self.pop_size)]

    def tell(self, solutions: list[np.ndarray], fitnesses: list[float]):
        order = np.argsort(fitnesses)
        selected = [solutions[i] for i in order[: self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum([w * x for w, x in zip(self.weights, selected)], axis=0)

        std = np.sqrt(self.C_diag)
        std_safe = np.where(std > 1e-30, std, 1e-30)

        displacement = (self.mean - old_mean) / (self.sigma * std_safe)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * displacement

        h_sigma = int(
            np.linalg.norm(self.p_sigma)
            / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))
            < (1.4 + 2 / (self.d + 1)) * self.chi_d
        )
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * (self.mean - old_mean) / self.sigma

        artmp = np.array([(x - old_mean) / self.sigma for x in selected])
        C_mu_update = np.sum(
            [w * (a / std_safe) ** 2 for w, a in zip(self.weights, artmp)], axis=0
        )
        self.C_diag = (
            (1 - self.c_1 - self.c_mu) * self.C_diag
            + self.c_1 * (self.p_c ** 2 + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C_diag)
            + self.c_mu * C_mu_update * self.C_diag
        )
        self.C_diag = np.maximum(self.C_diag, 1e-20)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_d - 1)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)
        self.generation += 1


# ---------------------------------------------------------------------------
# Encoder for piecewise distributions
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax3(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class PiecewiseEncoder:
    """Real-vector <-> Piecewise distribution params.

    Layout per segment: 3 logits for (p_w, p_r, p_i) simplex + 1 logit for bit_p1.
    Segment starts are fixed uniformly in [0, 1); only the params vary. This
    keeps dimensionality low and avoids CMA searching over ordering.

    Total dims = 4 * K for K segments.
    """
    def __init__(self, T: int, K: int):
        self.T = T
        self.K = K
        self.n_dims = 4 * K

    def decode(self, x: np.ndarray):
        from .distribution import Piecewise
        segments = []
        for k in range(self.K):
            base = 4 * k
            simplex = _softmax3(x[base : base + 3])
            p_w, p_r, _ = simplex
            bit_p1 = float(_sigmoid(np.array([x[base + 3]]))[0])
            start_frac = k / self.K
            segments.append((start_frac, float(p_w), float(p_r), bit_p1))
        return Piecewise(T=self.T, segments=segments)

    def random_init(self, rng: np.random.Generator) -> np.ndarray:
        # Small-magnitude init centers the simplex near uniform and bit_p1 ~ 0.5.
        return rng.standard_normal(self.n_dims) * 0.3


# ---------------------------------------------------------------------------
# CMA-ES search loop
# ---------------------------------------------------------------------------
def cma_search(
    encoder: PiecewiseEncoder,
    objective_fn: Callable[[FFLDistribution], FitnessResult],
    out_dir: str,
    *,
    budget: int = 3000,
    pop_size: int = 16,
    sigma_init: float = 0.3,
    num_restarts: int = 3,
    seed: int = 0,
    log_every: int = 1,
) -> list[EvalResult]:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "adversary_log.jsonl")
    budget_per = budget // max(1, num_restarts)
    print(f"[cma] n_dims={encoder.n_dims} budget={budget} restarts={num_restarts} "
          f"pop={pop_size} sigma={sigma_init}")

    all_results: list[EvalResult] = []
    best = -float("inf")
    running_mean_fitness = 0.0
    n_running = 0
    step = 0
    traj_path = os.path.join(out_dir, "cma_trajectory.jsonl")
    with open(log_path, "a") as fh, open(traj_path, "a") as tfh:
        for restart in range(num_restarts):
            rng = np.random.default_rng(seed + restart)
            x0 = encoder.random_init(rng)
            es = DiagonalCMAES(x0, sigma_init, pop_size, seed=seed + restart)
            evals_done = 0
            gen = 0
            while evals_done < budget_per:
                solutions = es.ask()
                fitnesses = []
                gen_fitnesses_signed = []
                for sol in solutions:
                    dist = encoder.decode(sol)
                    fr = objective_fn(dist)
                    r = EvalResult.from_fitness(dist, fr, step=step,
                                                restart=restart, generation=gen)
                    all_results.append(r)
                    fh.write(json.dumps(asdict(r)) + "\n")
                    fh.flush()
                    fitnesses.append(-r.fitness if r.is_valid else 0.0)
                    gen_fitnesses_signed.append(r.fitness if r.is_valid else 0.0)
                    if r.fitness > best:
                        best = r.fitness
                    n_running += 1
                    running_mean_fitness += (r.fitness - running_mean_fitness) / n_running
                    step += 1
                es.tell(solutions, fitnesses)
                evals_done += len(solutions)
                gen += 1

                # Per-generation trajectory log: lets us verify plateau claims
                # post-hoc without re-running CMA.
                gen_best = max(gen_fitnesses_signed)
                gen_mean = float(np.mean(gen_fitnesses_signed))
                c_diag = es.C_diag
                tfh.write(json.dumps({
                    "restart": restart,
                    "generation": gen,
                    "evals_done": evals_done,
                    "gen_best": gen_best,
                    "gen_mean": gen_mean,
                    "running_best": best,
                    "running_mean": running_mean_fitness,
                    "sigma": float(es.sigma),
                    "c_diag_max": float(c_diag.max()),
                    "c_diag_min": float(c_diag.min()),
                    "c_diag_ratio": float(c_diag.max() / max(c_diag.min(), 1e-30)),
                }) + "\n")
                tfh.flush()

                if gen % log_every == 0:
                    print(f"  [cma r{restart} gen{gen:03d}] evals={evals_done}/{budget_per} "
                          f"gen_best={gen_best:.4e} gen_mean={gen_mean:.4e} "
                          f"best={best:.4e} sigma={es.sigma:.3e}")
    return all_results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_checkpoint(results: list[EvalResult], out_dir: str):
    path = os.path.join(out_dir, "checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump([asdict(r) for r in results], f)
