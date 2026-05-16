"""FFLM "clean"-mode loss and glitch-rate evaluation.

Paper's primary metric: fraction of read-answer tokens mispredicted
(argmax decoding). In causal LM terms, logits at position t predict token t+1;
we score only positions where tokens[t] == R.
"""
import torch
import torch.nn.functional as F

from .data import R


def clean_loss(logits, tokens):
    """Cross-entropy on deterministic (post-read) positions only."""
    shift_logits = logits[:, :-1, :]
    shift_targets = tokens[:, 1:]
    mask = tokens[:, :-1] == R

    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    ).view_as(shift_targets)

    mask_f = mask.float()
    return (ce * mask_f).sum() / mask_f.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_dataset_per_seq(model, tokens, batch_size=64, device="cpu"):
    """Per-sequence read-error / read-count arrays for a paired bootstrap.

    Returns (errors_per_seq, reads_per_seq), both numpy int64 arrays of shape
    (N,). Same masking/logits as evaluate_dataset; this is the only addition
    needed for statistically-rigorous CIs at ~1e-5 error scales. The aggregate
    evaluate_dataset below is left untouched (existing callers unaffected).
    """
    import numpy as np

    was_training = model.training
    model.eval()
    err_chunks = []
    read_chunks = []
    for i in range(0, tokens.size(0), batch_size):
        batch = tokens[i : i + batch_size].to(device)
        logits = model(batch)
        shift_logits = logits[:, :-1, :]
        shift_targets = batch[:, 1:]
        mask = batch[:, :-1] == R
        preds = shift_logits.argmax(dim=-1)
        err = ((preds != shift_targets) & mask).sum(dim=1)   # (b,)
        rds = mask.sum(dim=1)                                 # (b,)
        err_chunks.append(err.cpu().numpy().astype("int64"))
        read_chunks.append(rds.cpu().numpy().astype("int64"))
    if was_training:
        model.train()
    return np.concatenate(err_chunks), np.concatenate(read_chunks)


@torch.no_grad()
def evaluate_dataset(model, tokens, batch_size=64, device="cpu"):
    """Compute read-error rate (glitch rate) on a pre-sampled dataset.

    Returns a dict with error_rate, num_errors, num_predictions, loss.
    """
    was_training = model.training
    model.eval()
    total_errors = 0
    total_preds = 0
    total_loss_sum = 0.0

    for i in range(0, tokens.size(0), batch_size):
        batch = tokens[i : i + batch_size].to(device)
        logits = model(batch)

        shift_logits = logits[:, :-1, :]
        shift_targets = batch[:, 1:]
        mask = batch[:, :-1] == R

        preds = shift_logits.argmax(dim=-1)
        total_errors += ((preds != shift_targets) & mask).sum().item()
        total_preds += mask.sum().item()

        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_targets.reshape(-1),
            reduction="none",
        ).view_as(shift_targets)
        total_loss_sum += (ce * mask.float()).sum().item()

    if was_training:
        model.train()

    denom = max(total_preds, 1)
    return {
        "error_rate": total_errors / denom,
        "num_errors": total_errors,
        "num_predictions": total_preds,
        "loss": total_loss_sum / denom,
    }
