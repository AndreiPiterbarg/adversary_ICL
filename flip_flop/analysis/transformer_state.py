"""Extract per-layer residual stream from FFLMTransformer.

GPT2LMHeadModel supports output_hidden_states natively; FFLMTransformer.forward
strips that, so we call the inner HF model directly.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def extract_transformer_hidden_states(transformer_model, tokens, device, batch_size=32, to_cpu=True):
    """Return tuple of (n_layer + 1) hidden-state tensors, each shape (B, L, n_embd).

    hidden_states[0] is the embedding output (= input to block 0).
    hidden_states[k] for k in 1..n_layer is the output of block k-1
    (= input to block k, or input to ln_f when k == n_layer).
    """
    tokens = tokens.to(device)
    B = tokens.size(0)
    accum = None
    for i in range(0, B, batch_size):
        batch = tokens[i : i + batch_size]
        out = transformer_model.model(
            input_ids=batch,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        if accum is None:
            accum = [[] for _ in hs]
        for j, h in enumerate(hs):
            accum[j].append(h.cpu() if to_cpu else h)
    return tuple(torch.cat(layer, dim=0) for layer in accum)
