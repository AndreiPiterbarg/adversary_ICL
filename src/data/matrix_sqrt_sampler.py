"""
Matrix Square Root Sampler for Nonlinear Refinement Experiments

Given SPD matrix A, predict x = A^{-1/2} b.

Matrix token = A.  Context = (b_i, A^{-1/2} b_i) pairs.
Compute A^{-1/2} via eigendecomposition: A = Q diag(lam) Q^T
=> A^{-1/2} = Q diag(lam^{-1/2}) Q^T.

The model receives A (not A^{1/2}), so it cannot do one-step inversion.
Uses lower condition number range (1-20) to keep the problem well-conditioned.
"""

import torch
import numpy as np

from data.spd_sampler import sample_spd


def _compute_inv_sqrt(A):
    """Compute A^{-1/2} via eigendecomposition.

    Args:
        A: (B, d, d) SPD matrices

    Returns:
        A_inv_sqrt: (B, d, d)
    """
    eigenvalues, Q = torch.linalg.eigh(A)
    # Clamp for numerical safety
    eigenvalues = eigenvalues.clamp(min=1e-8)
    inv_sqrt_eigs = eigenvalues.pow(-0.5)
    return Q @ torch.diag_embed(inv_sqrt_eigs) @ Q.transpose(-2, -1)


def make_batch_matrix_sqrt(B, K, d, device,
                           kappa_min=1.0, kappa_max=20.0):
    """Generate a batch for matrix square root inversion.

    Returns:
        5-tuple (A, ctx_inputs, ctx_outputs, query_input, query_target)
        where A is (B, d, d) SPD matrix.
    """
    A = sample_spd(B, d, device, kappa_min, kappa_max)
    A_inv_sqrt = _compute_inv_sqrt(A)

    # Generate K+1 random vectors
    b_all = torch.randn(B, K + 1, d, device=device)

    # Compute x = A^{-1/2} b for all vectors
    # A_inv_sqrt: (B, d, d),  b_all: (B, K+1, d)
    x_all = torch.bmm(b_all, A_inv_sqrt.transpose(-2, -1))  # (B, K+1, d)

    ctx_in = b_all[:, :K]       # (B, K, d)
    ctx_out = x_all[:, :K]      # (B, K, d)
    q_in = b_all[:, K]          # (B, d)
    q_tgt = x_all[:, K]         # (B, d)

    return A, ctx_in, ctx_out, q_in, q_tgt


def gradient_correction(A, b, x_current):
    """Compute gradient-based correction for hypothesis testing.

    For the problem x = A^{-1/2} b, consider minimizing
        L(x) = ||A^{1/2} x - b||^2
    whose gradient is  A(x) - A^{1/2} b.

    Since A^{1/2} b = A^{1/2} (A^{1/2} x*) = A x* is computable from
    ground truth, we instead compute the Newton step for the linear system
        A^{1/2} x = b  =>  step = -(A^{1/2})^{-1} (A^{1/2} x - b)
                                  = A^{-1/2} b - x = x* - x

    For a more interesting comparison, we compute the preconditioned gradient:
        delta = -A^{-1} (A x_k - A^{1/2} b)
    which requires A^{1/2}, computed via eigendecomposition.

    Args:
        A:         (B, d, d) SPD matrices
        b:         (B, d) input vectors
        x_current: (B, d) current estimates

    Returns:
        grad_step: (B, d) negative preconditioned gradient direction
    """
    eigenvalues, Q = torch.linalg.eigh(A)
    eigenvalues = eigenvalues.clamp(min=1e-8)
    sqrt_eigs = eigenvalues.pow(0.5)
    A_sqrt = Q @ torch.diag_embed(sqrt_eigs) @ Q.transpose(-2, -1)

    # residual = A^{1/2} x_k - b
    residual = torch.bmm(A_sqrt, x_current.unsqueeze(-1)).squeeze(-1) - b

    # Preconditioned step: -A^{-1} A^{1/2} residual = -A^{-1/2} residual
    inv_sqrt_eigs = eigenvalues.pow(-0.5)
    A_inv_sqrt = Q @ torch.diag_embed(inv_sqrt_eigs) @ Q.transpose(-2, -1)
    grad_step = -torch.bmm(A_inv_sqrt, residual.unsqueeze(-1)).squeeze(-1)

    return grad_step
