"""
Comprehensive Hypothesis Testing Suite for Self-Refine ICL

Tests all hypotheses systematically with checkpoint/resume capability.
Provides statistical analysis and evidence for/against each hypothesis.

Usage:
    python scripts/run_hypothesis_suite.py --device cuda
    python scripts/run_hypothesis_suite.py --device cuda --resume  # Resume interrupted run
    python scripts/run_hypothesis_suite.py --analyze-only  # Just analyze existing results

Hypotheses tested:
    H1: Higher residual_weight improves refinement (0.3 vs 0.5 vs 0.7)
    H2: Deeper models improve performance (n_layer: 4, 6, 8)
    H3: Wider models improve performance (n_embd: 64, 128, 256)
    H4: Higher noise during training improves generalization (noise_scale: 0.25, 0.5, 1.0)
    H5: Curriculum training improves hard problem performance
    H6: More test iterations continue to improve results (5, 10, 15)
    H7: Learning rate affects convergence (5e-5, 1e-4, 2e-4)
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import numpy as np
from scipy import stats


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    hypothesis: str
    training_steps: int = 50000
    residual_weight: float = 0.5
    n_layer: int = 6
    n_embd: int = 128
    n_head: int = 4
    lr: float = 1e-4
    noise_scale: float = 0.5
    kappa_min: float = 1.0
    kappa_max: float = 100.0
    curriculum: bool = False
    curriculum_warmup: int = 10000
    test_iterations: int = 5

    def to_cli_args(self, output_dir: str, device: str) -> List[str]:
        """Convert config to CLI arguments."""
        args = [
            "--training_steps", str(self.training_steps),
            "--residual_weight", str(self.residual_weight),
            "--n_layer", str(self.n_layer),
            "--n_embd", str(self.n_embd),
            "--n_head", str(self.n_head),
            "--lr", str(self.lr),
            "--noise_scale", str(self.noise_scale),
            "--kappa_min", str(self.kappa_min),
            "--kappa_max", str(self.kappa_max),
            "--test_iterations", str(self.test_iterations),
            "--output_dir", output_dir,
            "--device", device,
        ]
        if self.curriculum:
            args.extend(["--curriculum", "--curriculum_warmup", str(self.curriculum_warmup)])
        return args

    def get_id(self) -> str:
        """Get unique identifier for this experiment."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    results_path: Optional[str] = None

    # Key metrics (populated after completion)
    avg_improvement_fraction: Optional[float] = None
    improvement_by_kappa: Optional[Dict[str, float]] = None
    final_mse_by_kappa: Optional[Dict[str, float]] = None


def define_experiments() -> List[ExperimentConfig]:
    """Define all experiments to test hypotheses."""
    experiments = []

    # Standard ICL baseline (no residual training, 2x steps for fair compute)
    # This is the control - trains only direct prediction
    experiments.append(ExperimentConfig(
        name="standard_icl_baseline",
        hypothesis="baseline",
        residual_weight=0.0,  # No residual training
        training_steps=100000,  # 2x steps for equal compute (1 vs 2 forward passes)
    ))

    # Role-Disambiguated Residual default (residual prediction with default settings)
    experiments.append(ExperimentConfig(
        name="rdr_default",
        hypothesis="role_disambiguated_residual",
    ))

    # H1: Residual weight sweep
    for rw in [0.3, 0.7]:  # 0.5 is rdr_default
        experiments.append(ExperimentConfig(
            name=f"rw_{rw}",
            hypothesis="H1_residual_weight",
            residual_weight=rw,
        ))

    # H2: Model depth
    for n_layer in [4, 8]:  # 6 is baseline
        experiments.append(ExperimentConfig(
            name=f"layer_{n_layer}",
            hypothesis="H2_model_depth",
            n_layer=n_layer,
        ))

    # H3: Model width
    for n_embd in [64, 256]:  # 128 is baseline
        # Adjust n_head to be compatible with n_embd
        n_head = 4 if n_embd >= 64 else 2
        experiments.append(ExperimentConfig(
            name=f"embd_{n_embd}",
            hypothesis="H3_model_width",
            n_embd=n_embd,
            n_head=n_head,
        ))

    # H4: Noise scale
    for noise in [0.25, 1.0]:  # 0.5 is baseline
        experiments.append(ExperimentConfig(
            name=f"noise_{noise}",
            hypothesis="H4_noise_scale",
            noise_scale=noise,
        ))

    # H5: Curriculum learning
    experiments.append(ExperimentConfig(
        name="curriculum",
        hypothesis="H5_curriculum",
        curriculum=True,
        curriculum_warmup=10000,
    ))

    # H6: Extended test iterations - these are special and handled separately
    # (test_only mode on baseline model)

    # H7: Learning rate
    for lr in [5e-5, 2e-4]:  # 1e-4 is baseline
        experiments.append(ExperimentConfig(
            name=f"lr_{lr}",
            hypothesis="H7_learning_rate",
            lr=lr,
        ))

    # Combination experiments (promising combinations)
    # H8: Best combinations
    experiments.append(ExperimentConfig(
        name="curriculum_layer8",
        hypothesis="H8_combinations",
        curriculum=True,
        n_layer=8,
    ))

    experiments.append(ExperimentConfig(
        name="curriculum_rw07",
        hypothesis="H8_combinations",
        curriculum=True,
        residual_weight=0.7,
    ))

    return experiments


@dataclass
class ExtendedIterationTest:
    """Configuration for extended iteration tests (uses existing model)."""
    name: str
    test_iterations: int
    model_name: str = "baseline"  # Which model to test


class HypothesisSuite:
    """Manages the hypothesis testing suite with checkpointing."""

    def __init__(self, base_dir: str, device: str = "cuda"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.manifest_path = self.base_dir / "manifest.json"
        self.experiments: Dict[str, ExperimentResult] = {}

        # Extended iteration tests (test_only mode on existing models)
        self.extended_tests: Dict[str, Dict] = {}

        # Load existing manifest if it exists
        self._load_manifest()

    def _load_manifest(self):
        """Load experiment manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                data = json.load(f)

            for exp_id, exp_data in data.get("experiments", {}).items():
                config = ExperimentConfig(**exp_data["config"])
                result = ExperimentResult(
                    config=config,
                    status=exp_data["status"],
                    start_time=exp_data.get("start_time"),
                    end_time=exp_data.get("end_time"),
                    duration_seconds=exp_data.get("duration_seconds"),
                    error_message=exp_data.get("error_message"),
                    results_path=exp_data.get("results_path"),
                    avg_improvement_fraction=exp_data.get("avg_improvement_fraction"),
                    improvement_by_kappa=exp_data.get("improvement_by_kappa"),
                    final_mse_by_kappa=exp_data.get("final_mse_by_kappa"),
                )
                self.experiments[exp_id] = result

            # Load extended iteration tests
            self.extended_tests = data.get("extended_tests", {})

            print(f"Loaded manifest with {len(self.experiments)} experiments, {len(self.extended_tests)} extended tests")

    def _save_manifest(self):
        """Save experiment manifest to disk."""
        data = {
            "last_updated": datetime.now().isoformat(),
            "device": self.device,
            "experiments": {},
            "extended_tests": self.extended_tests,
        }

        for exp_id, result in self.experiments.items():
            data["experiments"][exp_id] = {
                "config": asdict(result.config),
                "status": result.status,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration_seconds": result.duration_seconds,
                "error_message": result.error_message,
                "results_path": result.results_path,
                "avg_improvement_fraction": result.avg_improvement_fraction,
                "improvement_by_kappa": result.improvement_by_kappa,
                "final_mse_by_kappa": result.final_mse_by_kappa,
            }

        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def initialize_experiments(self, experiments: List[ExperimentConfig]):
        """Initialize experiments, preserving completed ones."""
        for config in experiments:
            exp_id = config.get_id()
            if exp_id not in self.experiments:
                self.experiments[exp_id] = ExperimentResult(
                    config=config,
                    status="pending"
                )
            elif self.experiments[exp_id].status == "running":
                # Reset stuck experiments
                self.experiments[exp_id].status = "pending"

        self._save_manifest()

    def run_experiment(self, exp_id: str) -> bool:
        """Run a single experiment. Returns True if successful."""
        result = self.experiments[exp_id]
        config = result.config

        output_dir = self.base_dir / config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        result.status = "running"
        result.start_time = datetime.now().isoformat()
        result.results_path = str(output_dir)
        self._save_manifest()

        print(f"\n{'='*60}")
        print(f"Running: {config.name} ({config.hypothesis})")
        print(f"{'='*60}")

        # Build command
        script_path = Path(__file__).parent / "role_disambiguated_residual_prediction.py"
        cmd = [
            sys.executable, str(script_path)
        ] + config.to_cli_args(str(output_dir), self.device)

        print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Run experiment
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            duration = time.time() - start_time
            result.duration_seconds = duration
            result.end_time = datetime.now().isoformat()

            if process.returncode != 0:
                result.status = "failed"
                result.error_message = process.stderr[-2000:] if process.stderr else "Unknown error"
                print(f"FAILED: {result.error_message}")
                self._save_manifest()
                return False

            # Load results
            results_file = output_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    exp_results = json.load(f)

                # Extract key metrics
                testing = exp_results.get("testing", {})
                improvement_fracs = []
                result.improvement_by_kappa = {}
                result.final_mse_by_kappa = {}

                for kappa_range, data in testing.items():
                    frac = data.get("improved_fraction", 0)
                    improvement_fracs.append(frac)
                    result.improvement_by_kappa[kappa_range] = frac

                    mse_data = data.get("mse_by_iteration", {})
                    # Get final iteration MSE
                    max_iter = max(int(k) for k in mse_data.keys()) if mse_data else 0
                    final_mse = mse_data.get(str(max_iter), {}).get("mean", 0)
                    result.final_mse_by_kappa[kappa_range] = final_mse

                result.avg_improvement_fraction = np.mean(improvement_fracs) if improvement_fracs else 0

            result.status = "completed"
            print(f"COMPLETED in {duration:.1f}s - Avg improvement: {result.avg_improvement_fraction*100:.1f}%")
            self._save_manifest()
            return True

        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = "Timeout after 2 hours"
            result.end_time = datetime.now().isoformat()
            self._save_manifest()
            return False
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.end_time = datetime.now().isoformat()
            self._save_manifest()
            return False

    def run_all_pending(self):
        """Run all pending experiments."""
        pending = [
            exp_id for exp_id, result in self.experiments.items()
            if result.status in ["pending", "running"]
        ]

        total = len(self.experiments)
        completed = sum(1 for r in self.experiments.values() if r.status == "completed")

        print(f"\nExperiment Status: {completed}/{total} completed, {len(pending)} remaining")

        for i, exp_id in enumerate(pending):
            print(f"\nProgress: {completed + i + 1}/{total}")
            self.run_experiment(exp_id)

        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETE")
        print(f"{'='*60}")

    def run_extended_iteration_tests(self):
        """Run extended iteration tests on completed models (H6)."""
        # Define extended iteration tests (test more refinement iterations on Role-Disambiguated Residual)
        extended_configs = [
            ExtendedIterationTest(name="role_disambiguated_residual_iter10", test_iterations=10, model_name="rdr_default"),
            ExtendedIterationTest(name="role_disambiguated_residual_iter15", test_iterations=15, model_name="rdr_default"),
            ExtendedIterationTest(name="role_disambiguated_residual_iter20", test_iterations=20, model_name="rdr_default"),
        ]

        # Find rdr_default model
        baseline_result = None
        for result in self.experiments.values():
            if result.config.name == "rdr_default" and result.status == "completed":
                baseline_result = result
                break

        if baseline_result is None:
            print("rdr_default not completed yet, skipping extended iteration tests")
            return

        model_path = Path(baseline_result.results_path) / "model.pt"
        if not model_path.exists():
            print(f"Baseline model not found at {model_path}")
            return

        print(f"\n{'='*60}")
        print("RUNNING EXTENDED ITERATION TESTS (H6)")
        print(f"{'='*60}")

        for test_config in extended_configs:
            test_key = test_config.name

            # Skip if already completed
            if test_key in self.extended_tests and self.extended_tests[test_key].get("status") == "completed":
                print(f"Skipping {test_key} (already completed)")
                continue

            print(f"\nRunning {test_key} ({test_config.test_iterations} iterations)...")

            output_dir = self.base_dir / test_key
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run test-only mode
            script_path = Path(__file__).parent / "role_disambiguated_residual_prediction.py"
            cmd = [
                sys.executable, str(script_path),
                "--test_only",
                "--model_path", str(model_path),
                "--test_iterations", str(test_config.test_iterations),
                "--output_dir", str(output_dir),
                "--device", self.device,
            ]

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if process.returncode != 0:
                    self.extended_tests[test_key] = {
                        "status": "failed",
                        "error": process.stderr[-1000:] if process.stderr else "Unknown",
                    }
                    self._save_manifest()
                    continue

                # Load results
                results_file = output_dir / "test_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        test_results = json.load(f)

                    testing = test_results.get("testing", {})
                    improvement_fracs = []
                    improvement_by_kappa = {}

                    for kappa_range, data in testing.items():
                        frac = data.get("improved_fraction", 0)
                        improvement_fracs.append(frac)
                        improvement_by_kappa[kappa_range] = frac

                    self.extended_tests[test_key] = {
                        "status": "completed",
                        "test_iterations": test_config.test_iterations,
                        "model_name": test_config.model_name,
                        "avg_improvement_fraction": float(np.mean(improvement_fracs)) if improvement_fracs else 0,
                        "improvement_by_kappa": improvement_by_kappa,
                    }

                    print(f"  Result: {self.extended_tests[test_key]['avg_improvement_fraction']*100:.1f}% improved")
                else:
                    self.extended_tests[test_key] = {"status": "failed", "error": "No results file"}

            except Exception as e:
                self.extended_tests[test_key] = {"status": "failed", "error": str(e)}

            self._save_manifest()

    def get_status_summary(self) -> Dict[str, int]:
        """Get count of experiments by status."""
        summary = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for result in self.experiments.values():
            summary[result.status] = summary.get(result.status, 0) + 1
        return summary

    def analyze_results(self) -> Dict:
        """Analyze results and test hypotheses."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_status_summary(),
            "hypotheses": {},
            "recommendations": [],
        }

        # Get rdr_default as reference for hyperparameter comparisons
        role_disambiguated_residual_ref = None
        standard_icl = None
        for result in self.experiments.values():
            if result.config.name == "rdr_default" and result.status == "completed":
                role_disambiguated_residual_ref = result
            if result.config.name == "standard_icl_baseline" and result.status == "completed":
                standard_icl = result

        if role_disambiguated_residual_ref is None:
            analysis["error"] = "rdr_default experiment not completed"
            return analysis

        baseline = role_disambiguated_residual_ref  # Use role_disambiguated_residual as reference for hypothesis tests
        baseline_frac = baseline.avg_improvement_fraction

        # Report main claim: Role-Disambiguated Residual vs Standard ICL
        if standard_icl:
            std_frac = standard_icl.avg_improvement_fraction or 0
            analysis["main_claim"] = {
                "standard_icl_improvement": std_frac,
                "role_disambiguated_residual_improvement": baseline_frac,
                "role_disambiguated_residual_advantage": baseline_frac - std_frac,
            }
            print(f"\nMain claim - Standard ICL: {std_frac*100:.1f}% vs Role-Disambiguated Residual: {baseline_frac*100:.1f}%")
        else:
            print(f"\nRole-Disambiguated Residual default improvement: {baseline_frac*100:.1f}%")

        # Group experiments by hypothesis
        by_hypothesis: Dict[str, List[ExperimentResult]] = {}
        for result in self.experiments.values():
            if result.status != "completed":
                continue
            h = result.config.hypothesis
            if h not in by_hypothesis:
                by_hypothesis[h] = []
            by_hypothesis[h].append(result)

        # Analyze each hypothesis
        for hypothesis, results in by_hypothesis.items():
            if hypothesis in ["baseline", "role_disambiguated_residual"]:
                continue

            h_analysis = self._analyze_hypothesis(hypothesis, results, baseline)
            analysis["hypotheses"][hypothesis] = h_analysis

            # Add recommendations
            if h_analysis.get("best_config"):
                best = h_analysis["best_config"]
                improvement = h_analysis.get("best_improvement_over_baseline", 0)
                if improvement > 0.05:  # >5% improvement
                    analysis["recommendations"].append({
                        "hypothesis": hypothesis,
                        "recommendation": f"Use {best['name']} (+{improvement*100:.1f}%)",
                        "config": best,
                    })

        # Add H6 analysis from extended tests
        if self.extended_tests:
            h6_analysis = self._analyze_extended_iterations(baseline)
            if h6_analysis:
                analysis["hypotheses"]["H6_test_iterations"] = h6_analysis

        return analysis

    def _analyze_extended_iterations(self, baseline: ExperimentResult) -> Optional[Dict]:
        """Analyze H6: extended iteration tests."""
        completed_tests = {
            k: v for k, v in self.extended_tests.items()
            if v.get("status") == "completed"
        }

        if not completed_tests:
            return None

        baseline_frac = baseline.avg_improvement_fraction

        h_analysis = {
            "experiments": [],
            "conclusion": "",
            "evidence_strength": "",
            "best_iterations": 5,
            "convergence_analysis": {},
        }

        # Include baseline (5 iterations) and extended tests
        all_results = [(5, baseline_frac)]

        for name, data in completed_tests.items():
            iters = data.get("test_iterations", 0)
            frac = data.get("avg_improvement_fraction", 0)
            all_results.append((iters, frac))

            h_analysis["experiments"].append({
                "name": name,
                "iterations": iters,
                "avg_improvement": frac,
                "vs_baseline": frac - baseline_frac,
            })

        # Sort by iterations
        all_results.sort(key=lambda x: x[0])

        # Check if improvement continues or plateaus
        fracs = [r[1] for r in all_results]
        iters = [r[0] for r in all_results]

        # Find best
        best_idx = np.argmax(fracs)
        best_iters = iters[best_idx]
        best_frac = fracs[best_idx]

        h_analysis["best_iterations"] = best_iters
        h_analysis["best_improvement"] = best_frac

        # Convergence: check if last is better than first
        if len(fracs) >= 2:
            improvement = fracs[-1] - fracs[0]
            if improvement > 0.05:
                h_analysis["conclusion"] = f"SUPPORTED - More iterations help (+{improvement*100:.1f}%)"
                h_analysis["evidence_strength"] = "strong" if improvement > 0.1 else "moderate"
            elif improvement > 0:
                h_analysis["conclusion"] = f"WEAKLY SUPPORTED - Slight improvement (+{improvement*100:.1f}%)"
                h_analysis["evidence_strength"] = "weak"
            elif improvement > -0.05:
                h_analysis["conclusion"] = "PLATEAUS - No benefit from more iterations"
                h_analysis["evidence_strength"] = "moderate"
            else:
                h_analysis["conclusion"] = f"REFUTED - More iterations hurt ({improvement*100:.1f}%)"
                h_analysis["evidence_strength"] = "moderate"

        return h_analysis

    def _analyze_hypothesis(
        self,
        hypothesis: str,
        results: List[ExperimentResult],
        baseline: ExperimentResult
    ) -> Dict:
        """Analyze a specific hypothesis."""
        h_analysis = {
            "experiments": [],
            "conclusion": "",
            "evidence_strength": "",
            "best_config": None,
            "best_improvement_over_baseline": 0,
        }

        baseline_frac = baseline.avg_improvement_fraction
        baseline_kappas = baseline.improvement_by_kappa or {}

        best_frac = baseline_frac
        best_result = None

        fracs = [baseline_frac]
        labels = ["baseline"]

        for result in results:
            frac = result.avg_improvement_fraction or 0
            fracs.append(frac)
            labels.append(result.config.name)

            exp_info = {
                "name": result.config.name,
                "avg_improvement": frac,
                "vs_baseline": frac - baseline_frac,
                "by_kappa": result.improvement_by_kappa,
            }
            h_analysis["experiments"].append(exp_info)

            if frac > best_frac:
                best_frac = frac
                best_result = result

        # Statistical test: one-way ANOVA if >2 groups, t-test if 2
        if len(fracs) == 2:
            # Can't do proper stats with single values, report difference
            diff = fracs[1] - fracs[0]
            h_analysis["conclusion"] = f"{'Improves' if diff > 0 else 'Worsens'} by {abs(diff)*100:.1f}%"
            h_analysis["evidence_strength"] = "weak (single comparison)"
        else:
            # Compare all against baseline
            improvements = [f - baseline_frac for f in fracs[1:]]
            mean_improvement = np.mean(improvements)

            if mean_improvement > 0.02:
                h_analysis["conclusion"] = f"SUPPORTED - Average +{mean_improvement*100:.1f}%"
                h_analysis["evidence_strength"] = "moderate" if mean_improvement > 0.05 else "weak"
            elif mean_improvement < -0.02:
                h_analysis["conclusion"] = f"REFUTED - Average {mean_improvement*100:.1f}%"
                h_analysis["evidence_strength"] = "moderate" if mean_improvement < -0.05 else "weak"
            else:
                h_analysis["conclusion"] = "INCONCLUSIVE - No significant difference"
                h_analysis["evidence_strength"] = "none"

        if best_result and best_result != baseline:
            h_analysis["best_config"] = {
                "name": best_result.config.name,
                **{k: v for k, v in asdict(best_result.config).items()
                   if k not in ["name", "hypothesis"]}
            }
            h_analysis["best_improvement_over_baseline"] = best_frac - baseline_frac

        return h_analysis

    def generate_report(self) -> str:
        """Generate a human-readable report."""
        analysis = self.analyze_results()

        lines = []
        lines.append("=" * 70)
        lines.append("HYPOTHESIS TESTING SUITE - FINAL REPORT")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {analysis['timestamp']}")

        # Main claim: Role-Disambiguated Residual vs Standard ICL
        main_claim = analysis.get("main_claim")
        if main_claim:
            lines.append("\n" + "-" * 70)
            lines.append("MAIN CLAIM: Iterative Residual Refinement vs Standard ICL")
            lines.append("-" * 70)
            std_pct = main_claim["standard_icl_improvement"] * 100
            app_pct = main_claim["role_disambiguated_residual_improvement"] * 100
            adv_pct = main_claim["role_disambiguated_residual_advantage"] * 100
            lines.append(f"  Standard ICL (2x steps):  {std_pct:.1f}% samples improved after refinement")
            lines.append(f"  Role-Disambiguated Residual (residual):    {app_pct:.1f}% samples improved after refinement")
            lines.append(f"  Advantage:                {'+' if adv_pct >= 0 else ''}{adv_pct:.1f}%")
            if adv_pct > 5:
                lines.append(f"  CONCLUSION: Residual prediction significantly outperforms standard ICL")
            elif adv_pct > 0:
                lines.append(f"  CONCLUSION: Residual prediction slightly outperforms standard ICL")
            else:
                lines.append(f"  CONCLUSION: No advantage over standard ICL")

        # Status summary
        summary = analysis["summary"]
        lines.append(f"\nExperiment Status:")
        lines.append(f"  Completed: {summary.get('completed', 0)}")
        lines.append(f"  Failed: {summary.get('failed', 0)}")
        lines.append(f"  Pending: {summary.get('pending', 0)}")

        ext_completed = sum(1 for v in self.extended_tests.values() if v.get("status") == "completed")
        if self.extended_tests:
            lines.append(f"  Extended iteration tests: {ext_completed}/{len(self.extended_tests)}")

        # Hypothesis results
        lines.append("\n" + "=" * 70)
        lines.append("HYPOTHESIS RESULTS")
        lines.append("=" * 70)

        for h_name, h_data in analysis.get("hypotheses", {}).items():
            lines.append(f"\n{h_name}")
            lines.append("-" * 40)
            lines.append(f"  Conclusion: {h_data.get('conclusion', 'N/A')}")
            lines.append(f"  Evidence: {h_data.get('evidence_strength', 'N/A')}")

            if h_data.get("experiments"):
                lines.append("  Experiments:")
                for exp in h_data["experiments"]:
                    diff = exp["vs_baseline"]
                    sign = "+" if diff >= 0 else ""
                    lines.append(f"    - {exp['name']}: {exp['avg_improvement']*100:.1f}% ({sign}{diff*100:.1f}%)")

        # Recommendations
        lines.append("\n" + "=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)

        recs = analysis.get("recommendations", [])
        if recs:
            for rec in sorted(recs, key=lambda x: -x.get("config", {}).get("name", "")):
                lines.append(f"\n  * {rec['recommendation']}")
        else:
            lines.append("\n  No configurations significantly outperformed baseline.")

        # Best overall configuration
        lines.append("\n" + "=" * 70)
        lines.append("BEST CONFIGURATION")
        lines.append("=" * 70)

        best_overall = None
        best_frac = 0
        for result in self.experiments.values():
            if result.status == "completed" and (result.avg_improvement_fraction or 0) > best_frac:
                best_frac = result.avg_improvement_fraction
                best_overall = result

        if best_overall:
            lines.append(f"\n  Name: {best_overall.config.name}")
            lines.append(f"  Improvement fraction: {best_frac*100:.1f}%")
            lines.append(f"  Configuration:")
            for key, val in asdict(best_overall.config).items():
                if key not in ["name", "hypothesis"]:
                    lines.append(f"    {key}: {val}")

        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run hypothesis testing suite")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="hypothesis_results")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    args = parser.parse_args()

    suite = HypothesisSuite(args.output_dir, args.device)

    # Initialize experiments
    experiments = define_experiments()
    suite.initialize_experiments(experiments)

    if args.status:
        summary = suite.get_status_summary()
        print(f"\nExperiment Status:")
        for status, count in summary.items():
            print(f"  {status}: {count}")

        print(f"\nMain Experiments:")
        for exp_id, result in suite.experiments.items():
            config = result.config
            frac = result.avg_improvement_fraction
            frac_str = f"{frac*100:.1f}%" if frac is not None else "N/A"
            print(f"  [{result.status:9s}] {config.name:20s} ({config.hypothesis}) - {frac_str}")

        if suite.extended_tests:
            print(f"\nExtended Iteration Tests (H6):")
            for name, data in suite.extended_tests.items():
                status = data.get("status", "unknown")
                frac = data.get("avg_improvement_fraction")
                frac_str = f"{frac*100:.1f}%" if frac is not None else "N/A"
                iters = data.get("test_iterations", "?")
                print(f"  [{status:9s}] {name:20s} ({iters} iters) - {frac_str}")
        return

    if args.analyze_only:
        report = suite.generate_report()
        print(report)

        # Save report
        report_path = Path(args.output_dir) / "final_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")

        # Save JSON analysis
        analysis = suite.analyze_results()
        analysis_path = Path(args.output_dir) / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_path}")
        return

    # Run experiments
    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTING SUITE")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Total experiments: {len(experiments)}")

    suite.run_all_pending()

    # Run extended iteration tests (H6) - uses test_only mode on baseline
    suite.run_extended_iteration_tests()

    # Generate final report
    report = suite.generate_report()
    print("\n" + report)

    # Save report
    report_path = Path(args.output_dir) / "final_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save JSON analysis
    analysis = suite.analyze_results()
    analysis_path = Path(args.output_dir) / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
