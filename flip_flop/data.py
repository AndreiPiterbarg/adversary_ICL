"""FFL(T, p) data sampler.

Canonical family from Liu et al. 2023, Section 3.1:
  (i)   x_1 = w, x_{T-1} = r.
  (ii)  Other instructions drawn i.i.d. from (p_w, p_r, p_i).
  (iii) Data symbols following w or i are drawn i.i.d. uniformly from {0, 1};
        the symbol following r is deterministic (= most-recent-write value).

Uses the FFL(p_i) shorthand with p_w = p_r = (1 - p_i) / 2.

Vocabulary (size 5):  w=0, r=1, i=2, "0"=3, "1"=4.
Instructions occupy positions 0, 2, ..., T-2; data occupy positions 1, 3, ..., T-1.
"""
import numpy as np
import torch

W, R, I, ZERO, ONE = 0, 1, 2, 3, 4
VOCAB_SIZE = 5
ID_TO_SYMBOL = {W: "w", R: "r", I: "i", ZERO: "0", ONE: "1"}


def enforce_read_determinism(inst, data):
    """Propagate most-recent-write across reads, in-place on `data`.

    inst, data: (B, n_inst) int arrays. Positions where inst == R get data
    overwritten with the current stored bit; positions where inst == W update
    the stored bit. Also asserts the canonical boundary conditions
    inst[:, 0] == W and inst[:, -1] == R.

    Returns `data`.
    """
    assert (inst[:, 0] == W).all(), "first instruction must be W"
    assert (inst[:, -1] == R).all(), "last instruction must be R"
    n_inst = inst.shape[1]
    last_write = data[:, 0].copy()
    for k in range(1, n_inst):
        is_w = inst[:, k] == W
        is_r = inst[:, k] == R
        last_write = np.where(is_w, data[:, k], last_write)
        data[:, k] = np.where(is_r, last_write, data[:, k])
    return data


def interleave(inst, data):
    """Interleave instruction and data arrays into a (B, 2*n_inst) token tensor."""
    B, n_inst = inst.shape
    tokens = np.empty((B, 2 * n_inst), dtype=np.int64)
    tokens[:, 0::2] = inst
    tokens[:, 1::2] = data + ZERO
    return torch.from_numpy(tokens)


def sample_ffl(T, p_i, batch_size, rng):
    """Sample a batch of valid flip-flop strings from FFL(T, p_i).

    Returns a LongTensor of shape (batch_size, T).
    """
    assert T % 2 == 0 and T >= 4
    p_w = p_r = (1.0 - p_i) / 2.0
    n_inst = T // 2

    inst = rng.choice(
        np.array([W, R, I], dtype=np.int64),
        size=(batch_size, n_inst),
        p=[p_w, p_r, p_i],
    )
    inst[:, 0] = W    # first instruction always w
    inst[:, -1] = R   # last instruction always r

    data = rng.integers(0, 2, size=(batch_size, n_inst), dtype=np.int64)
    enforce_read_determinism(inst, data)
    return interleave(inst, data)


def decode(tokens):
    """Decode a batch of token sequences to space-separated strings."""
    return [" ".join(ID_TO_SYMBOL[t] for t in row) for row in tokens.tolist()]


def make_eval_dataset(p_i, num_sequences, T, seed):
    """Pre-sample a fixed evaluation dataset."""
    rng = np.random.default_rng(seed)
    return sample_ffl(T, p_i, num_sequences, rng)
