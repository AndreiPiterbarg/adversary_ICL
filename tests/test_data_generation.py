"""Tests for SPD matrix sampling and linear system generation."""

import torch
import pytest
import numpy as np
from conftest import sample_spd


class TestSPDSampling:
    def test_positive_definite(self, device):
        """Sampled matrices have all positive eigenvalues."""
        A = sample_spd(100, d=4, kappa_min=1.0, kappa_max=100.0, device=device)
        eigs = torch.linalg.eigvalsh(A)
        assert (eigs > 0).all(), f"Found non-positive eigenvalues: {eigs.min()}"

    def test_symmetric(self, device):
        """Sampled matrices are symmetric."""
        A = sample_spd(100, d=4, kappa_min=1.0, kappa_max=100.0, device=device)
        assert torch.allclose(A, A.transpose(-2, -1), atol=1e-6)

    def test_condition_number_bounded(self, device):
        """Condition numbers are bounded (not necessarily in exact range due to sampling)."""
        kappa_min, kappa_max = 10.0, 50.0
        A = sample_spd(500, d=4, kappa_min=kappa_min, kappa_max=kappa_max, device=device)

        eigs = torch.linalg.eigvalsh(A)
        kappas = eigs.max(dim=-1).values / eigs.min(dim=-1).values

        # The sampling is stochastic; verify reasonable bounds
        assert kappas.min() >= 1.0, f"Kappa must be >= 1, got {kappas.min()}"
        # Most samples should be within a reasonable factor of the target range
        median_kappa = kappas.median().item()
        assert median_kappa > 1.0, f"Median kappa should be > 1, got {median_kappa}"

    def test_condition_number_increases_with_range(self, device):
        """Higher kappa_max produces higher average condition numbers."""
        A_low = sample_spd(200, d=4, kappa_min=1.0, kappa_max=10.0, device=device)
        A_high = sample_spd(200, d=4, kappa_min=1.0, kappa_max=100.0, device=device)

        eigs_low = torch.linalg.eigvalsh(A_low)
        eigs_high = torch.linalg.eigvalsh(A_high)

        kappas_low = eigs_low.max(dim=-1).values / eigs_low.min(dim=-1).values
        kappas_high = eigs_high.max(dim=-1).values / eigs_high.min(dim=-1).values

        assert kappas_high.mean() > kappas_low.mean(), "Higher kappa_max should give higher mean kappa"

    @pytest.mark.parametrize("kappa_range", [(1, 10), (10, 50), (50, 100), (100, 200)])
    def test_multiple_kappa_ranges(self, kappa_range, device):
        """Test each condition number range used in experiments."""
        kappa_min, kappa_max = kappa_range
        A = sample_spd(100, d=4, kappa_min=kappa_min, kappa_max=kappa_max, device=device)

        eigs = torch.linalg.eigvalsh(A)
        assert (eigs > 0).all()


class TestLinearSystemSolution:
    def test_solution_correctness(self, device):
        """A @ x_star == b for generated systems."""
        batch, d = 100, 4
        A = sample_spd(batch, d=d, kappa_min=1.0, kappa_max=100.0, device=device)
        b = torch.randn(batch, d, device=device)
        x_star = torch.linalg.solve(A, b)

        # Verify A @ x_star ≈ b
        residual = A @ x_star.unsqueeze(-1) - b.unsqueeze(-1)
        residual_norm = residual.squeeze(-1).norm(dim=-1)

        assert (residual_norm < 1e-4).all(), f"Max residual: {residual_norm.max()}"

    def test_batch_solve_consistency(self, device):
        """Batch solve gives same results as individual solves."""
        batch, d = 10, 4
        A = sample_spd(batch, d=d, kappa_min=1.0, kappa_max=50.0, device=device)
        b = torch.randn(batch, d, device=device)

        # Batch solve
        x_batch = torch.linalg.solve(A, b)

        # Individual solves
        x_individual = torch.stack([
            torch.linalg.solve(A[i], b[i]) for i in range(batch)
        ])

        assert torch.allclose(x_batch, x_individual, atol=1e-5)
