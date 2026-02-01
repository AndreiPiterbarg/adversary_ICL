"""
Non-Linear Least Squares Sampler for Nonlinear Refinement Experiments

Fits the model:  y = theta[2] * sigmoid(theta[0]*t + theta[1]) + theta[3]
with 4 parameters (d=4) and N=4 data points.

Matrix token encodes the data: each row = (t_i, t_i^2, y_i, 1).
Context pairs = (theta_init, theta_star) from Gauss-Newton convergence.

Gauss-Newton step:
    r_i  = y_i - model(t_i; theta)
    J_ij = dr_i / dtheta_j
    delta = -(J^T J)^{-1} J^T r
"""

import torch
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _predict(theta, t):
    """Model prediction y = theta[2]*sigmoid(theta[0]*t + theta[1]) + theta[3].

    Args:
        theta: (N, 4)
        t:     (N, P) data points

    Returns:
        y: (N, P)
    """
    z = theta[:, 0:1] * t + theta[:, 1:2]
    return theta[:, 2:3] * _sigmoid(z) + theta[:, 3:4]


def _jacobian(theta, t):
    """Compute Jacobian of residuals r = y_data - model(t; theta).

    dr_i/dtheta_j has a negative sign because r = y - f(theta).

    Args:
        theta: (N, 4)
        t:     (N, P)

    Returns:
        J: (N, P, 4)  where J[n, i, j] = -df/dtheta_j at point i
    """
    z = theta[:, 0:1] * t + theta[:, 1:2]       # (N, P)
    sig = _sigmoid(z)
    sig_prime = sig * (1.0 - sig)

    J = np.zeros((*theta.shape[:1], t.shape[1], 4))
    J[:, :, 0] = -theta[:, 2:3] * sig_prime * t   # d/d(theta_0)
    J[:, :, 1] = -theta[:, 2:3] * sig_prime        # d/d(theta_1)
    J[:, :, 2] = -sig                               # d/d(theta_2)
    J[:, :, 3] = -1.0                               # d/d(theta_3)
    return J


def _gauss_newton_solve(theta_star, t, y_data, theta_init,
                        max_iters=500, tol=1e-10, damping=1e-6):
    """Run Gauss-Newton from theta_init to find theta_star (vectorized).

    Returns:
        theta_final: (N, 4)
        converged:   (N,) bool
    """
    N = theta_init.shape[0]
    theta = theta_init.copy()

    for _ in range(max_iters):
        pred = _predict(theta, t)
        r = y_data - pred                                  # (N, P)
        J = _jacobian(theta, t)                            # (N, P, 4)

        JtJ = np.einsum('npi,npj->nij', J, J)             # (N, 4, 4)
        Jtr = np.einsum('npi,np->ni', J, r)               # (N, 4)

        # Levenberg-Marquardt damping for stability
        JtJ += damping * np.eye(4)[None, :, :]

        try:
            delta = np.linalg.solve(JtJ, -Jtr[..., None]).squeeze(-1)  # (N, 4)
        except np.linalg.LinAlgError:
            return theta, np.zeros(N, dtype=bool)

        theta = theta + delta

        if np.all(np.max(np.abs(delta), axis=-1) < tol):
            break

    # Check convergence: did we reach theta_star?
    dist = np.max(np.abs(theta - theta_star), axis=-1)
    converged = dist < 0.05
    return theta, converged


def make_batch_nlls(B, K, d, device,
                    theta_scale=None, start_noise=0.3):
    """Generate a batch for non-linear least squares fitting.

    Returns:
        5-tuple (data_matrix, ctx_inputs, ctx_outputs, query_input, query_target)
        where data_matrix is (B, d, d) encoding (t, t^2, y, 1) per data point.
    """
    assert d == 4, "NLLS sampler designed for d=4"
    if theta_scale is None:
        theta_scale = np.array([1.0, 1.0, 2.0, 1.0])

    P = 4  # number of data points = d
    oversample = int(B * 2.0) + 32

    while True:
        # --- 1. Sample true parameters (moderate range for convergence) -------
        theta_star = np.random.randn(oversample, d) * theta_scale[None, :]
        # Clamp to avoid sigmoid saturation
        theta_star[:, 0] = np.clip(theta_star[:, 0], -2.0, 2.0)
        theta_star[:, 1] = np.clip(theta_star[:, 1], -2.0, 2.0)
        # Ensure scale parameter is nonzero
        theta_star[:, 2] = np.where(np.abs(theta_star[:, 2]) < 0.3,
                                    np.sign(theta_star[:, 2]) * 0.3 +
                                    (np.abs(theta_star[:, 2]) < 0.01) * 0.3,
                                    theta_star[:, 2])

        # --- 2. Sample data points and compute y_data ------------------------
        t_data = np.random.uniform(-2.0, 2.0, (oversample, P))
        y_data = _predict(theta_star, t_data)  # noiseless

        # --- 3. Build matrix token: rows = (t, t^2, y, 1) --------------------
        M = np.stack([t_data, t_data ** 2, y_data,
                      np.ones_like(t_data)], axis=-1)  # (OS, P, 4) = (OS, d, d)

        # --- 4. Generate K+1 starting points, verify GN convergence ----------
        starts = (theta_star[:, None, :] +
                  np.random.randn(oversample, K + 1, d) * start_noise)

        all_valid = np.ones(oversample, dtype=bool)
        for k in range(K + 1):
            _, conv = _gauss_newton_solve(theta_star, t_data, y_data,
                                          starts[:, k, :])
            all_valid &= conv

        valid_idx = np.where(all_valid)[0]
        if len(valid_idx) >= B:
            idx = valid_idx[:B]

            M_out = torch.tensor(M[idx], dtype=torch.float32, device=device)
            ctx_in = torch.tensor(starts[idx, :K],
                                  dtype=torch.float32, device=device)
            tgt_expanded = np.broadcast_to(theta_star[idx, None, :],
                                           (B, K, d)).copy()
            ctx_out = torch.tensor(tgt_expanded,
                                   dtype=torch.float32, device=device)
            q_in = torch.tensor(starts[idx, K],
                                dtype=torch.float32, device=device)
            q_tgt = torch.tensor(theta_star[idx],
                                 dtype=torch.float32, device=device)
            return M_out, ctx_in, ctx_out, q_in, q_tgt

        oversample = int(oversample * 2)


def gauss_newton_correction(data_matrix, theta_current, device):
    """Compute one Gauss-Newton correction step (for hypothesis testing).

    Args:
        data_matrix:    (B, d, d) data encoding — rows are (t, t^2, y, 1)
        theta_current:  (B, d) current parameter estimate

    Returns:
        delta: (B, d) Gauss-Newton correction
    """
    t = data_matrix[:, :, 0]        # (B, 4)
    y = data_matrix[:, :, 2]        # (B, 4)

    z = theta_current[:, 0:1] * t + theta_current[:, 1:2]
    sig = torch.sigmoid(z)
    sig_prime = sig * (1.0 - sig)

    pred = theta_current[:, 2:3] * sig + theta_current[:, 3:4]
    r = y - pred                                            # (B, 4)

    # Jacobian (B, 4, 4)
    J = torch.zeros(theta_current.shape[0], 4, 4, device=device)
    J[:, :, 0] = -theta_current[:, 2:3] * sig_prime * t
    J[:, :, 1] = -theta_current[:, 2:3] * sig_prime
    J[:, :, 2] = -sig
    J[:, :, 3] = -1.0

    JtJ = torch.bmm(J.transpose(1, 2), J)
    JtJ += 1e-6 * torch.eye(4, device=device).unsqueeze(0)
    Jtr = torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)

    delta = torch.linalg.solve(JtJ, -Jtr)
    return delta
