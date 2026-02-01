"""
Polynomial Root Finding Sampler for Nonlinear Refinement Experiments

Generates systems of d=4 independent cubic polynomials and corresponding
(starting_point, converged_root) pairs for training ICL models on iterative
root finding via Newton's method.

Each polynomial: p_j(x) = c_{j,0} + c_{j,1}*x + c_{j,2}*x^2 + c_{j,3}*x^3
Newton step:     delta_j = -p_j(x_k[j]) / p_j'(x_k[j])
"""

import torch
import numpy as np


def _eval_cubic(C, x):
    """Evaluate cubic polynomials and their derivatives (vectorized).

    Args:
        C: (N, d, 4) coefficient matrix, row j = [c0, c1, c2, c3]
        x: (N, d) evaluation points

    Returns:
        p:  (N, d) polynomial values
        dp: (N, d) derivative values
    """
    x2 = x * x
    x3 = x2 * x
    p = C[:, :, 0] + C[:, :, 1] * x + C[:, :, 2] * x2 + C[:, :, 3] * x3
    dp = C[:, :, 1] + 2.0 * C[:, :, 2] * x + 3.0 * C[:, :, 3] * x2
    return p, dp


def _newton_converge(C, x0, max_iters=300, tol=1e-12):
    """Run Newton's method on independent cubics (vectorized).

    Args:
        C:  (N, d, 4) coefficients
        x0: (N, d) starting points

    Returns:
        x_final: (N, d) converged points
        converged: (N,) bool mask
    """
    x = x0.copy()
    converged = np.ones(x.shape[0], dtype=bool)

    for _ in range(max_iters):
        p, dp = _eval_cubic(C, x)
        bad = np.abs(dp) < 1e-15
        if np.any(bad):
            converged &= ~np.any(bad, axis=-1)
        dp_safe = np.where(bad, 1.0, dp)
        delta = -p / dp_safe
        x = x + delta
        if np.all(np.max(np.abs(delta), axis=-1) < tol):
            break

    # Final residual check
    p, _ = _eval_cubic(C, x)
    converged &= np.max(np.abs(p), axis=-1) < 1e-8
    return x, converged


def make_batch_polynomial(B, K, d, device,
                          root_range=3.0, min_separation=0.5,
                          start_noise=0.5):
    """Generate a batch for polynomial root-finding.

    Constructs d independent monic cubics from sampled roots, picks one root
    per polynomial as the target, generates starting points in the target's
    basin of attraction, and rejects samples that don't converge.

    Returns:
        5-tuple (C, ctx_inputs, ctx_outputs, query_input, query_target)
        where C is the coefficient matrix (B, d, d).
    """
    assert d == 4, "Polynomial sampler designed for d=4 (cubic has 4 coefficients)"
    oversample = int(B * 1.8) + 32

    while True:
        # --- 1. Sample 3 roots per polynomial --------------------------------
        roots = np.random.uniform(-root_range, root_range,
                                  (oversample, d, 3))  # (OS, d, 3)

        # Enforce minimum separation between roots
        r_sorted = np.sort(roots, axis=-1)
        diffs = np.diff(r_sorted, axis=-1)            # (OS, d, 2)
        well_sep = diffs.min(axis=-1).min(axis=-1) > min_separation  # (OS,)

        # --- 2. Build coefficients from roots ---------------------------------
        # p_j(x) = (x - r1)(x - r2)(x - r3) = x^3 + c2*x^2 + c1*x + c0
        r1, r2, r3 = roots[:, :, 0], roots[:, :, 1], roots[:, :, 2]
        C = np.zeros((oversample, d, d), dtype=np.float64)
        C[:, :, 3] = 1.0
        C[:, :, 2] = -(r1 + r2 + r3)
        C[:, :, 1] = r1 * r2 + r1 * r3 + r2 * r3
        C[:, :, 0] = -r1 * r2 * r3

        # --- 3. Target = first root of each polynomial -----------------------
        target = r1.copy()  # (OS, d)

        # --- 4. Generate K+1 starting points, verify convergence to target ---
        starts = (target[:, None, :] +
                  np.random.randn(oversample, K + 1, d) * start_noise)

        all_valid = well_sep.copy()
        for k in range(K + 1):
            x_conv, conv_mask = _newton_converge(C, starts[:, k, :])
            # Check converged to the TARGET root (not a different one)
            near_target = np.max(np.abs(x_conv - target), axis=-1) < 0.1
            all_valid &= conv_mask & near_target

        valid_idx = np.where(all_valid)[0]
        if len(valid_idx) >= B:
            idx = valid_idx[:B]

            C_out = torch.tensor(C[idx], dtype=torch.float32, device=device)
            ctx_in = torch.tensor(starts[idx, :K],
                                  dtype=torch.float32, device=device)
            # All context outputs are the same target root
            tgt_expanded = np.broadcast_to(target[idx, None, :],
                                           (B, K, d)).copy()
            ctx_out = torch.tensor(tgt_expanded,
                                   dtype=torch.float32, device=device)
            q_in = torch.tensor(starts[idx, K],
                                dtype=torch.float32, device=device)
            q_tgt = torch.tensor(target[idx],
                                 dtype=torch.float32, device=device)
            return C_out, ctx_in, ctx_out, q_in, q_tgt

        # Increase oversample and retry
        oversample = int(oversample * 2)


def newton_correction(C, x):
    """Compute one Newton correction step (for hypothesis testing).

    Args:
        C: (B, d, d) coefficient matrix (float tensor)
        x: (B, d) current estimate (float tensor)

    Returns:
        delta: (B, d) Newton correction  delta_j = -p_j(x_j) / p_j'(x_j)
    """
    x2 = x * x
    x3 = x2 * x
    p = C[:, :, 0] + C[:, :, 1] * x + C[:, :, 2] * x2 + C[:, :, 3] * x3
    dp = C[:, :, 1] + 2.0 * C[:, :, 2] * x + 3.0 * C[:, :, 3] * x2
    dp_safe = torch.where(dp.abs() < 1e-15, torch.ones_like(dp), dp)
    return -p / dp_safe
