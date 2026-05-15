"""Linear probe (closed-form ridge regression) and supporting analysis.

Methodology:
  - Probe target: LSTM cell state c_t (Liu et al. 2023's "optimal" 1-bit memory
    realized in 128-d). See arXiv:2306.00946.
  - Probe input: Transformer residual stream x_t^(l) per layer l.
  - Fit per (model, layer): closed-form ridge regression with centering.
  - Score: variance-weighted R^2 (= 1 - sum SS_res / sum SS_tot across outputs).
  - Selectivity (Hewitt & Liang 2019, arXiv:1909.03368): identical procedure
    on c_t shuffled within each sequence; report `selectivity = R^2 - R^2_ctrl`.
  - Rank analysis: SVD of W to find smallest k s.t. rank-k truncation recovers
    >= threshold * R^2_full. Used downstream for causal subspace ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FitResult:
    W: torch.Tensor          # (d_in, d_out) regression matrix
    mean_x: torch.Tensor     # (1, d_in) train-set input mean
    mean_y: torch.Tensor     # (1, d_out) train-set target mean
    r2: float                # eval-set variance-weighted R^2


def _center(X_train, X_eval):
    mean = X_train.mean(dim=0, keepdim=True)
    return X_train - mean, X_eval - mean, mean


def fit_ridge(X_train: torch.Tensor, Y_train: torch.Tensor, ridge: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ridge: W = (X^T X + lambda I)^-1 X^T Y, with mean-centering."""
    mean_x = X_train.mean(dim=0, keepdim=True)
    mean_y = Y_train.mean(dim=0, keepdim=True)
    Xc = X_train - mean_x
    Yc = Y_train - mean_y
    d = Xc.size(1)
    A = Xc.T @ Xc + ridge * torch.eye(d, device=Xc.device, dtype=Xc.dtype)
    rhs = Xc.T @ Yc
    W = torch.linalg.solve(A, rhs)
    return W, mean_x, mean_y


def predict(X: torch.Tensor, W: torch.Tensor, mean_x: torch.Tensor, mean_y: torch.Tensor) -> torch.Tensor:
    return (X - mean_x) @ W + mean_y


def r2_variance_weighted(Y_pred: torch.Tensor, Y_true: torch.Tensor) -> float:
    """Variance-weighted R^2 = 1 - sum_j SS_res_j / sum_j SS_tot_j."""
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1.0 - (ss_res / ss_tot.clamp_min(1e-10)).item())


def fit_and_score(X_train, Y_train, X_eval, Y_eval, ridge: float = 1e-2) -> FitResult:
    W, mean_x, mean_y = fit_ridge(X_train, Y_train, ridge)
    Y_pred = predict(X_eval, W, mean_x, mean_y)
    return FitResult(W=W, mean_x=mean_x, mean_y=mean_y, r2=r2_variance_weighted(Y_pred, Y_eval))


def shuffle_within_sequence(Y: torch.Tensor, n_seq: int, seq_len: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Shuffle Y of shape (n_seq * seq_len, k) along time axis within each seq.

    Used as the Hewitt-Liang control task: preserves the marginal distribution
    of c_t but breaks the (input, target) correspondence.
    """
    Y_seq = Y.view(n_seq, seq_len, -1)
    if generator is None:
        perm = torch.stack([torch.randperm(seq_len) for _ in range(n_seq)])
    else:
        perm = torch.stack([torch.randperm(seq_len, generator=generator) for _ in range(n_seq)])
    perm = perm.to(Y.device)
    out = torch.gather(Y_seq, 1, perm.unsqueeze(-1).expand(-1, -1, Y_seq.size(-1)))
    return out.reshape(-1, Y.size(-1))


def rank_curve(W: torch.Tensor, X_eval: torch.Tensor, Y_eval: torch.Tensor, mean_x: torch.Tensor, mean_y: torch.Tensor, max_rank: Optional[int] = None) -> tuple[list[float], torch.Tensor]:
    """R^2 of rank-k SVD truncation for k = 1..min(d_in, d_out).

    Returns (r2_per_rank, U) where U is (d_in, r) with orthonormal columns —
    the input-side singular directions of W, in order of decreasing singular
    value. U[:, :k] spans the rank-k input subspace the probe reads.
    """
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    full_r = S.numel()
    if max_rank is None:
        max_rank = full_r
    max_rank = min(max_rank, full_r)
    r2s = []
    Xc = X_eval - mean_x
    for k in range(1, max_rank + 1):
        Wk = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        Y_pred = Xc @ Wk + mean_y
        r2s.append(r2_variance_weighted(Y_pred, Y_eval))
    return r2s, U


def smallest_rank_meeting(r2s: list[float], r2_full: float, fraction: float = 0.9) -> int:
    """Smallest 1-indexed k such that r2s[k-1] >= fraction * r2_full."""
    target = fraction * r2_full
    for k, r in enumerate(r2s, start=1):
        if r >= target:
            return k
    return len(r2s)
