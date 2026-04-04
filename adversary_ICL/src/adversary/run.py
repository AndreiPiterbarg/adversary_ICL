"""Entry point for adversarial search.

Usage:
    python -m src.adversary.run --config configs/adversary.yaml
    python -m src.adversary.run --config configs/adversary.yaml --analyze-only
"""

import argparse
import os
import sys

import yaml
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.icl.eval import get_model_from_run
from src.adversary.evaluate import GenomeEvaluator
from src.adversary.search import cma_search
from src.adversary.genome import Genome
from src.adversary.pipeline_genome import PipelineGenome
from src.adversary.analyze import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Adversarial ICL search")
    parser.add_argument("--config", type=str, required=True, help="Path to adversary config YAML")
    parser.add_argument("--analyze-only", action="store_true", help="Skip search, only run analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    save_dir = config.get("output", {}).get("save_dir", "results/adversary_runs")

    if args.analyze_only:
        run_analysis(save_dir)
        return

    # Load ICL model
    model_config = config["icl_model"]
    run_path = model_config["run_path"]
    step = model_config.get("step", -1)
    print(f"Loading ICL model from {run_path} (step={step})...")
    model, conf = get_model_from_run(run_path, step)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Task config
    task_config = config.get("task", {})
    n_dims = task_config.get("n_dims", conf.model.n_dims)
    n_points = task_config.get("n_points", 41)
    task_name = task_config.get("name", "noisy_linear_regression")

    # Genome config
    genome_config = config.get("genome", {})
    genome_type = genome_config.get("type", "pipeline")
    genome_cls = PipelineGenome if genome_type == "pipeline" else Genome

    # Parsimony config
    parsimony_config = config.get("parsimony", {})
    parsimony_lambda = parsimony_config.get("lambda", 0.0)
    c_base = parsimony_config.get("c_base", 1.0)
    c_stage = parsimony_config.get("c_stage", 1.0)
    c_affine = parsimony_config.get("c_affine", 1.0)

    # Eval config
    eval_config = config.get("eval", {})
    batch_size = eval_config.get("batch_size", 64)
    num_batches = eval_config.get("num_batches", 10)
    baselines = eval_config.get("baselines", ["ridge", "least_squares", "averaging"])

    # Search config
    search_config = config.get("search", {})
    budget = search_config.get("budget", 50000)
    pop_size = search_config.get("pop_size", 32)
    sigma_init = search_config.get("sigma_init", 0.5)
    num_restarts = search_config.get("num_restarts", 5)
    seed = search_config.get("seed", 0)
    save_interval = search_config.get("save_interval", 50)

    # Build evaluator
    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name=task_name,
        n_dims=n_dims,
        n_points=n_points,
        batch_size=batch_size,
        num_batches=num_batches,
        baseline_names=baselines,
        parsimony_lambda=parsimony_lambda,
        c_base=c_base,
        c_stage=c_stage,
        c_affine=c_affine,
    )

    # Run search
    genome_size = genome_cls.flat_size(n_dims)
    print(f"\nStarting adversarial search (genome_type={genome_type})...")
    print(f"  n_dims={n_dims}, n_points={n_points}, budget={budget}")
    print(f"  genome_size={genome_size}, num_restarts={num_restarts}")
    if parsimony_lambda > 0:
        print(f"  parsimony: lambda={parsimony_lambda}, c_base={c_base}, c_stage={c_stage}, c_affine={c_affine}")
    print(f"  save_dir={save_dir}")

    results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=budget,
        pop_size=pop_size,
        sigma_init=sigma_init,
        num_restarts=num_restarts,
        save_dir=save_dir,
        save_interval=save_interval,
        seed=seed,
        genome_cls=genome_cls,
    )

    # Run analysis (with optional diagnostics)
    print("\n--- Post-hoc Analysis ---")
    diag_config = config.get("diagnostics", {})
    run_analysis(
        save_dir,
        icl_model=model,
        evaluator=evaluator,
        run_diagnostics=diag_config.get("enabled", False),
        diagnostics_top_k=diag_config.get("top_k", 5),
        diagnostics_list=diag_config.get("run", None),
    )


if __name__ == "__main__":
    main()
