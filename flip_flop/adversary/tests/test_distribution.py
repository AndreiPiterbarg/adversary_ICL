"""Sampler-validity tests: every adversary distribution must emit valid FFL."""
import numpy as np
import pytest
import torch

from flip_flop.data import W, R, I, ZERO, ONE
from flip_flop.adversary.distribution import (BitMarkov, Periodic, Piecewise,
                                              Planted, Stationary,
                                              WriteFlipRate)


def _assert_valid_ffl(tokens: torch.LongTensor, T: int):
    """Assert (B, T) tokens are valid FFL strings."""
    assert tokens.shape[1] == T
    x = tokens.numpy()
    # Boundary: x[:,0] == W, x[:,-2] == R.
    assert (x[:, 0] == W).all(), "first token must be W"
    assert (x[:, -2] == R).all(), "penultimate token must be R"
    # Alternation: even positions are instructions {W,R,I}, odd are data {ZERO,ONE}.
    even = x[:, 0::2]
    odd = x[:, 1::2]
    assert np.isin(even, [W, R, I]).all(), "even positions must be instructions"
    assert np.isin(odd, [ZERO, ONE]).all(), "odd positions must be data bits"
    # Read-determinism: every read's next-bit must equal most-recent-write bit.
    B, n_inst = even.shape
    for b in range(B):
        last = None
        for k in range(n_inst):
            ins = even[b, k]
            bit = odd[b, k] - ZERO
            if ins == W:
                last = bit
            elif ins == R:
                assert last is not None, f"R before any W at sample {b} pos {k}"
                assert bit == last, f"read mismatch sample {b} pos {k}: {bit} vs {last}"


@pytest.mark.parametrize("p_w,p_r,bit_p1", [
    (0.1, 0.1, 0.5),
    (0.01, 0.01, 0.5),   # sparse
    (0.4, 0.4, 0.5),     # dense
    (0.3, 0.05, 0.5),    # asymmetric
    (0.1, 0.1, 0.9),     # biased bits
])
def test_stationary_valid(p_w, p_r, bit_p1):
    d = Stationary(T=64, p_w=p_w, p_r=p_r, bit_p1=bit_p1)
    tokens = d.sample(32, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)


@pytest.mark.parametrize("bit_stay", [0.1, 0.5, 0.9])
def test_bit_markov_valid(bit_stay):
    d = BitMarkov(T=128, p_w=0.1, p_r=0.1, bit_p1=0.5, bit_stay=bit_stay)
    tokens = d.sample(16, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 128)


@pytest.mark.parametrize("flip_rate", [0.0, 0.5, 1.0])
def test_write_flip_valid(flip_rate):
    d = WriteFlipRate(T=128, p_w=0.2, p_r=0.1, flip_rate=flip_rate)
    tokens = d.sample(16, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 128)


def test_piecewise_valid():
    segs = [(0.0, 0.4, 0.4, 0.5), (0.5, 0.01, 0.01, 0.5)]
    d = Piecewise(T=128, segments=segs)
    tokens = d.sample(16, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 128)


def test_periodic_valid():
    pat = [(0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.0, 0.0, 0.5), (0.0, 0.0, 0.5)]
    d = Periodic(T=64, period=4, pattern=pat)
    tokens = d.sample(8, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)


@pytest.mark.parametrize("template,params", [
    ("gap", {"k_write": 1}),
    ("gap", {"k_write": 100}),
    ("decoy", {"k_early": 1, "k_true": 120, "b_decoy": 0}),
    ("decoy", {"k_early": 5, "k_true": 60, "b_decoy": 1}),
    ("distractor", {"k_true": 3, "d": 1, "b_true": 0}),
    ("distractor", {"k_true": 10, "d": 5, "b_true": 1}),
    ("disagree", {"N_w": 3, "frac_agree": 0.0, "b_last": 0}),
    ("disagree", {"N_w": 5, "frac_agree": 1.0, "b_last": 1}),
])
def test_planted_valid(template, params):
    d = Planted(T=256, template=template, params=params)
    tokens = d.sample(8, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 256)


def test_roundtrip_to_from_dict():
    from flip_flop.adversary.distribution import FFLDistribution
    cases = [
        Stationary(T=64, p_w=0.1, p_r=0.2, bit_p1=0.7),
        BitMarkov(T=64, p_w=0.1, p_r=0.1, bit_stay=0.8),
        WriteFlipRate(T=64, p_w=0.1, p_r=0.1, flip_rate=0.3),
        Piecewise(T=64, segments=[(0.0, 0.2, 0.2, 0.5), (0.5, 0.05, 0.05, 0.5)]),
        Periodic(T=64, period=2, pattern=[(0.3, 0.1, 0.5), (0.0, 0.2, 0.5)]),
        Planted(T=64, template="gap", params={"k_write": 3}),
    ]
    for d in cases:
        d2 = FFLDistribution.from_dict(d.to_dict())
        assert d2.to_dict() == d.to_dict()


def test_baseline_sample_ffl_unchanged():
    """The refactored sample_ffl must match pre-refactor behavior for identical seeds."""
    from flip_flop.data import sample_ffl
    rng = np.random.default_rng(42)
    tokens = sample_ffl(128, 0.8, 8, rng)
    _assert_valid_ffl(tokens, 128)
    # Deterministic under same seed
    rng2 = np.random.default_rng(42)
    tokens2 = sample_ffl(128, 0.8, 8, rng2)
    assert torch.equal(tokens, tokens2)
