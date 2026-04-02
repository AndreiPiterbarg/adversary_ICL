import os
import pickle

import numpy as np

from .genome import Genome
from .evaluate import GenomeEvaluator, EvalResult


class DiagonalCMAES:
    """Minimal diagonal CMA-ES (sep-CMA-ES) for high-dimensional spaces.

    Maintains only diagonal covariance (O(d) memory vs O(d^2) for full CMA-ES).
    Good enough for d=481 and avoids external dependencies.
    """

    def __init__(self, x0: np.ndarray, sigma: float, pop_size: int, seed: int = 0):
        self.d = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)

        # Diagonal covariance (variance per dimension)
        self.C_diag = np.ones(self.d)

        # Evolution paths
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)

        # Strategy parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.d) / (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.d + 2) ** 2 + self.mu_eff))

        self.chi_d = np.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))

        self.mu = mu
        self.generation = 0

    def ask(self) -> list[np.ndarray]:
        """Sample pop_size candidates from the current distribution."""
        std = self.sigma * np.sqrt(self.C_diag)
        samples = []
        for _ in range(self.pop_size):
            z = self.rng.standard_normal(self.d)
            x = self.mean + std * z
            samples.append(x)
        return samples

    def tell(self, solutions: list[np.ndarray], fitnesses: list[float]):
        """Update distribution based on evaluated fitnesses (minimization)."""
        # Sort by fitness (ascending = better for minimization)
        order = np.argsort(fitnesses)
        selected = [solutions[i] for i in order[: self.mu]]

        # Weighted recombination
        old_mean = self.mean.copy()
        self.mean = np.sum([w * x for w, x in zip(self.weights, selected)], axis=0)

        std = np.sqrt(self.C_diag)
        std_safe = np.where(std > 1e-30, std, 1e-30)

        # Evolution path for sigma
        displacement = (self.mean - old_mean) / (self.sigma * std_safe)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * displacement

        # Evolution path for covariance
        h_sigma = int(
            np.linalg.norm(self.p_sigma)
            / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))
            < (1.4 + 2 / (self.d + 1)) * self.chi_d
        )
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * (self.mean - old_mean) / self.sigma

        # Covariance update (diagonal only)
        artmp = np.array([(x - old_mean) / self.sigma for x in selected])
        C_mu_update = np.sum(
            [w * (a / std_safe) ** 2 for w, a in zip(self.weights, artmp)], axis=0
        )
        self.C_diag = (
            (1 - self.c_1 - self.c_mu) * self.C_diag
            + self.c_1 * (self.p_c ** 2 + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C_diag)
            + self.c_mu * C_mu_update * self.C_diag
        )

        # Ensure positive
        self.C_diag = np.maximum(self.C_diag, 1e-20)

        # Sigma update
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_d - 1)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)

        self.generation += 1


def _run_single_cma(
    evaluator: GenomeEvaluator,
    n_dims: int,
    budget: int,
    pop_size: int,
    sigma_init: float,
    seed: int,
    restart_id: int,
) -> list[EvalResult]:
    """Run a single CMA-ES restart. Returns list of EvalResults."""
    rng = np.random.default_rng(seed)
    x0 = Genome.random_structured(n_dims, rng).raw

    es = DiagonalCMAES(x0, sigma_init, pop_size, seed)

    results: list[EvalResult] = []
    best_fitness = 0.0
    generation = 0
    total_evals = 0

    while total_evals < budget:
        solutions = es.ask()
        generation += 1

        fitnesses = []
        gen_results = []

        for sol in solutions:
            genome = Genome(n_dims, np.array(sol))
            result = evaluator.evaluate(genome)
            gen_results.append(result)
            # CMA-ES minimizes, so negate fitness (we want to maximize gap)
            fitnesses.append(-result.fitness if result.is_valid else 0.0)

        es.tell(solutions, fitnesses)
        results.extend(gen_results)
        total_evals += len(solutions)

        # Track best
        for r in gen_results:
            if r.is_valid and r.fitness > best_fitness:
                best_fitness = r.fitness

        # Progress report
        valid = [r for r in gen_results if r.is_valid]
        if valid:
            gen_best = max(r.fitness for r in valid)
            gen_mean = np.mean([r.fitness for r in valid])
            print(
                f"  [restart {restart_id}] Gen {generation:4d} | evals {total_evals:6d}/{budget} | "
                f"gen_best={gen_best:.4f} gen_mean={gen_mean:.4f} | "
                f"best={best_fitness:.4f}"
            )

    return results


def cma_search(
    evaluator: GenomeEvaluator,
    n_dims: int,
    budget: int = 50000,
    pop_size: int = 32,
    sigma_init: float = 0.5,
    num_restarts: int = 5,
    save_dir: str | None = None,
    save_interval: int = 50,
    seed: int = 0,
) -> list[EvalResult]:
    """Run multi-restart CMA-ES adversarial search over the genome space.

    The total budget is split evenly across independent restarts, each
    initialized from a different random structured genome. This dramatically
    improves coverage of the fitness landscape vs. a single long run.

    Args:
        evaluator: Evaluates genomes against ICL model + baselines.
        n_dims: Dimensionality of the input space (determines genome size).
        budget: Total number of genome evaluations (split across restarts).
        pop_size: Population size per CMA-ES generation.
        sigma_init: Initial step size for CMA-ES.
        num_restarts: Number of independent CMA-ES runs.
        save_dir: Directory to save checkpoints. None = no saving.
        save_interval: Save checkpoint every N generations.
        seed: Base random seed (each restart uses seed + restart_id).

    Returns:
        List of all EvalResults from all restarts.
    """
    genome_size = Genome.flat_size(n_dims)
    budget_per_restart = budget // num_restarts

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(
        f"Starting multi-restart CMA-ES: genome_size={genome_size}, "
        f"budget={budget}, restarts={num_restarts}, "
        f"budget_per_restart={budget_per_restart}, pop_size={pop_size}"
    )

    all_results: list[EvalResult] = []
    best_fitness = 0.0
    best_result: EvalResult | None = None

    for restart_id in range(num_restarts):
        restart_seed = seed + restart_id
        print(f"\n--- Restart {restart_id + 1}/{num_restarts} (seed={restart_seed}) ---")

        restart_results = _run_single_cma(
            evaluator=evaluator,
            n_dims=n_dims,
            budget=budget_per_restart,
            pop_size=pop_size,
            sigma_init=sigma_init,
            seed=restart_seed,
            restart_id=restart_id,
        )

        all_results.extend(restart_results)

        # Track global best
        for r in restart_results:
            if r.is_valid and r.fitness > best_fitness:
                best_fitness = r.fitness
                best_result = r

        # Report restart summary
        valid = [r for r in restart_results if r.is_valid]
        if valid:
            restart_best = max(r.fitness for r in valid)
            print(
                f"  Restart {restart_id} complete: {len(restart_results)} evals, "
                f"restart_best={restart_best:.4f}, global_best={best_fitness:.4f}"
            )

        # Save after each restart
        if save_dir:
            _save_checkpoint(save_dir, all_results, restart_id + 1, best_result)

    print(f"\nSearch complete. {len(all_results)} total evaluations, best fitness={best_fitness:.4f}")
    if best_result:
        print(f"Best genome: {best_result.genome}")

    return all_results


def _save_checkpoint(save_dir, all_results, restarts_completed, best_result):
    """Save search state to disk."""
    checkpoint = {
        "results": all_results,
        "restarts_completed": restarts_completed,
        "best_result": best_result,
    }
    path = os.path.join(save_dir, "checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  [saved checkpoint: {len(all_results)} results, {restarts_completed} restarts]")


def load_checkpoint(save_dir: str) -> dict:
    """Load a saved checkpoint."""
    path = os.path.join(save_dir, "checkpoint.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
