"""Causal subspace ablation of the Transformer residual stream.

Implements the standard activation-patching variants from the mech-interp
literature (Vig et al. 2020 arXiv:2004.12265; Meng et al. 2022 arXiv:2202.05262;
Wang et al. 2023 arXiv:2211.00593) restricted to a fixed linear subspace U,
following the LEACE / INLP family (Belrose et al. 2023 arXiv:2306.03819;
Ravfogel et al. 2020 arXiv:2004.07667).

Given an orthonormal U (d, k) and a target layer l, we replace the projection
P x of the residual stream onto span(U) at layer l with one of:
  - mean    : the batch-averaged P x (do_mean ablation)
  - zero    : 0 (kills the subspace entirely)
  - resample: P x' for x' a different sequence in the same batch

x' = x - P(x - replacement) = x - P x + P replacement.

Hooks the inner GPT2 model: transformer.h[l] for l < n_layer (input to block l),
or transformer.ln_f for l == n_layer (residual immediately before the LM head).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch


AblationMode = Literal["mean", "zero", "resample"]


def _apply_subspace(x: torch.Tensor, P: torch.Tensor, mean_proj: torch.Tensor, mode: AblationMode) -> torch.Tensor:
    # x: (B, L, d). P: (d, d). mean_proj: (1, 1, d) precomputed = mean of (x @ P).
    proj_x = x @ P
    if mode == "mean":
        return x - proj_x + mean_proj
    if mode == "zero":
        return x - proj_x
    if mode == "resample":
        B = x.size(0)
        perm = torch.randperm(B, device=x.device)
        return x - proj_x + proj_x[perm]
    raise ValueError(f"unknown ablation mode: {mode}")


def _resolve_target_module(transformer_model, layer: int):
    """Return the module whose forward_pre_hook intercepts hidden_states[layer]."""
    inner = transformer_model.model.transformer
    n_layer = len(inner.h)
    if 0 <= layer < n_layer:
        return inner.h[layer]
    if layer == n_layer:
        return inner.ln_f
    raise ValueError(f"layer {layer} out of range (0..{n_layer})")


@contextmanager
def subspace_ablation(transformer_model, layer: int, U: torch.Tensor, mean_x_for_proj: torch.Tensor, mode: AblationMode):
    """Context manager that installs a forward_pre_hook ablating span(U) at `layer`.

    Args:
        layer: 0..n_layer (inclusive).
        U: (d, k) orthonormal columns.
        mean_x_for_proj: (1, d) reference mean used for `mean` ablation
            (typically the train-set mean from the probe fit).
        mode: 'mean' | 'zero' | 'resample'.
    """
    P = (U @ U.T).contiguous()
    mean_proj = (mean_x_for_proj @ P).reshape(1, 1, -1).contiguous()
    module = _resolve_target_module(transformer_model, layer)

    def pre_hook(_module, args, kwargs):
        if not args:
            return None
        x = args[0]
        x_new = _apply_subspace(x, P.to(x.device, x.dtype), mean_proj.to(x.device, x.dtype), mode)
        new_args = (x_new,) + tuple(args[1:])
        return new_args, kwargs

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def random_orthonormal(d: int, k: int, device, dtype, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample a (d, k) matrix with orthonormal columns (random rank-k subspace)."""
    A = torch.randn(d, k, device=device, dtype=dtype, generator=generator)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q
