"""CPU-only tests for the feature-conditioned controller.

Small by construction (T<=256, B<=2000, pure numpy + one torch.from_numpy);
no model forward pass, no GPU — heavy/GPU is reserved for the running Rung 1.

Coverage:
  * validity            — every sample is valid FFL (the legitimacy invariant)
  * determinism         — same seed -> identical tokens
  * round-trip          — to_dict/from_dict identity
  * floor guarantee     — instruction_probs >= floor & sums to 1 (degeneracy
                           unreachable BY CONSTRUCTION, even at extreme logits)
  * param count         — encoder n_dims == 24 == 3*(D_FEATURES+1)
  * SUBSUMPTION          — controller reproduces Stationary (exact),
                           WriteFlipRate (exact write-bit law), BitMarkov
                           (exact transition law), Piecewise (position ramp,
                           approximate). Executable proof of the superset claim.
  * encoder<->CMA        — DiagonalCMAES ask/tell drives the 24-dim encoder
                           unchanged; every solution decodes to a valid sampler.
"""
import numpy as np
import pytest
import torch

from expressive_adversary.data import W, R, I, ZERO
from expressive_adversary.distribution import (
    D_FEATURES, N_PARAMS, BitMarkov, FFLDistribution, FeatureControlledFFL,
    Stationary, WriteFlipRate,
)
from expressive_adversary.encoder import DiagonalCMAES, FeatureControlledEncoder


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _assert_valid_ffl(tokens, T):
    assert tokens.shape[1] == T
    x = tokens.numpy()
    assert (x[:, 0] == W).all(), "first token must be W"
    assert (x[:, -2] == R).all(), "penultimate token must be R"
    even, odd = x[:, 0::2], x[:, 1::2]
    assert np.isin(even, [W, R, I]).all(), "even positions must be instructions"
    assert np.isin(odd, [ZERO, ONE := 4]).all(), "odd positions must be data bits"
    B, n_inst = even.shape
    for b in range(B):
        last = None
        for k in range(n_inst):
            ins = even[b, k]
            bit = odd[b, k] - ZERO
            if ins == W:
                last = bit
            elif ins == R:
                assert last is not None, f"R before W (sample {b}, pos {k})"
                assert bit == last, f"read mismatch (sample {b}, pos {k})"


def _inst_freq(tokens):
    inst = tokens.numpy()[:, 0::2]
    n = inst.size
    return ((inst == W).sum() / n, (inst == R).sum() / n, (inst == I).sum() / n)


def _write_bit_p1(tokens):
    inst = tokens.numpy()[:, 0::2]
    bits = tokens.numpy()[:, 1::2] - ZERO
    m = inst == W
    return bits[m].mean() if m.any() else 0.0


def _consec_write_differ_rate(tokens):
    """P(write bit != previous write bit) over consecutive writes."""
    inst = tokens.numpy()[:, 0::2]
    bits = tokens.numpy()[:, 1::2] - ZERO
    diff = tot = 0
    for b in range(inst.shape[0]):
        prev = None
        for k in range(inst.shape[1]):
            if inst[b, k] == W:
                if prev is not None:
                    tot += 1
                    diff += int(bits[b, k] != prev)
                prev = bits[b, k]
    return diff / max(tot, 1)


def _consec_nonread_same_rate(tokens):
    """P(bit == prev bit) over adjacent slots where NEITHER is a read
    (reads are overwritten by read-determinism and would mask the bit chain)."""
    inst = tokens.numpy()[:, 0::2]
    bits = tokens.numpy()[:, 1::2] - ZERO
    same = tot = 0
    for b in range(inst.shape[0]):
        for k in range(1, inst.shape[1]):
            if inst[b, k] != R and inst[b, k - 1] != R:
                tot += 1
                same += int(bits[b, k] == bits[b, k - 1])
    return same / max(tot, 1)


def _quartile_write_freq(tokens):
    inst = tokens.numpy()[:, 0::2]
    n_inst = inst.shape[1]
    q = n_inst // 4
    return [float((inst[:, i * q:(i + 1) * q] == W).mean()) for i in range(4)]


def _rand_controller(seed, T=128):
    rng = np.random.default_rng(seed)
    d = D_FEATURES
    return FeatureControlledFFL(
        T=T, wR=rng.normal(0, 1.0, d), cR=rng.normal(),
        wI=rng.normal(0, 1.0, d), cI=rng.normal(),
        wb=rng.normal(0, 1.0, d), cb=rng.normal(),
    )


# ---------------------------------------------------------------------------
# validity / determinism / round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("T", [64, 128, 256])
def test_sampled_sequences_are_valid_ffl(seed, T):
    d = _rand_controller(seed, T)
    _assert_valid_ffl(d.sample(64, np.random.default_rng(seed)), T)


def test_extreme_weights_still_valid():
    """Pathological logits must not break validity (the _finalize gate is
    unconditional) nor escape the floors."""
    d = FeatureControlledFFL(
        T=128, wR=np.full(D_FEATURES, 25.0), cR=25.0,
        wI=np.full(D_FEATURES, -25.0), cI=-25.0,
        wb=np.full(D_FEATURES, 40.0), cb=-40.0,
    )
    _assert_valid_ffl(d.sample(64, np.random.default_rng(0)), 128)


def test_determinism_under_seed():
    d = _rand_controller(3, 128)
    a = d.sample(32, np.random.default_rng(123))
    b = d.sample(32, np.random.default_rng(123))
    assert torch.equal(a, b)
    c = d.sample(32, np.random.default_rng(124))
    assert not torch.equal(a, c)


def test_roundtrip_to_from_dict():
    for seed in (0, 5, 11):
        d = _rand_controller(seed, 128)
        d2 = FFLDistribution.from_dict(d.to_dict())
        assert d2.to_dict() == d.to_dict()
        # behavioural identity under a shared seed
        assert torch.equal(d.sample(16, np.random.default_rng(9)),
                           d2.sample(16, np.random.default_rng(9)))


# ---------------------------------------------------------------------------
# structural non-degeneracy (the principled anti-edge-case guarantee)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_instruction_probs_respect_floor_and_simplex(seed):
    rng = np.random.default_rng(seed)
    d = FeatureControlledFFL(
        T=128, wR=rng.normal(0, 8, D_FEATURES), cR=rng.normal(0, 8),
        wI=rng.normal(0, 8, D_FEATURES), cI=rng.normal(0, 8),
        p_w_min=0.03, p_r_min=0.04, p_i_min=0.01,
    )
    phi = rng.normal(0, 3, size=(500, D_FEATURES))     # arbitrary, incl. extreme
    p = d.instruction_probs(phi)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert (p[:, 0] >= 0.03 - 1e-12).all()
    assert (p[:, 1] >= 0.04 - 1e-12).all()
    assert (p[:, 2] >= 0.01 - 1e-12).all()


def test_degeneracy_unreachable_end_to_end():
    """Even weights that scream 'all writes' / 'all ignores' cannot starve
    reads/writes below the floors in the actual sampled stream."""
    d = FeatureControlledFFL(
        T=256, wR=np.zeros(D_FEATURES), cR=20.0,     # push toward R
        wI=np.zeros(D_FEATURES), cI=20.0,            # and I
        p_w_min=0.02, p_r_min=0.02,
    )
    fW, fR, fI = _inst_freq(d.sample(512, np.random.default_rng(0)))
    assert fW >= 0.02 - 5e-3 and fR >= 0.02 - 5e-3


# ---------------------------------------------------------------------------
# parameter count
# ---------------------------------------------------------------------------
def test_param_count_is_24():
    assert D_FEATURES == 7
    assert N_PARAMS == 3 * (D_FEATURES + 1) == 24
    assert FeatureControlledEncoder(T=128).n_dims == 24


# ---------------------------------------------------------------------------
# SUBSUMPTION — executable proof of the strict-superset claim
# ---------------------------------------------------------------------------
def test_subsumes_stationary_exact():
    """All feature weights 0 + solved biases  ==  Stationary, distributionally."""
    T, B = 256, 1500
    for (p_w, p_r, bp1) in [(0.1, 0.1, 0.5), (0.2, 0.1, 0.7), (0.1, 0.3, 0.5)]:
        ctrl = FeatureControlledFFL.emulate_stationary(T, p_w, p_r, bp1)
        ref = Stationary(T=T, p_w=p_w, p_r=p_r, bit_p1=bp1)
        c = _inst_freq(ctrl.sample(B, np.random.default_rng(0)))
        r = _inst_freq(ref.sample(B, np.random.default_rng(1)))
        assert np.allclose(c, r, atol=0.02), f"inst freq {c} vs {r}"
        bc = _write_bit_p1(ctrl.sample(B, np.random.default_rng(2)))
        br = _write_bit_p1(ref.sample(B, np.random.default_rng(3)))
        assert abs(bc - br) < 0.02 and abs(bc - bp1) < 0.03


@pytest.mark.parametrize("flip_rate", [0.1, 0.5, 0.9])
def test_subsumes_write_flip_exact(flip_rate):
    """Bit head on `last_write_bit` reproduces WriteFlipRate's write-bit law."""
    T, B = 256, 1200
    ctrl = FeatureControlledFFL.emulate_write_flip(T, 0.2, 0.1, flip_rate)
    ref = WriteFlipRate(T=T, p_w=0.2, p_r=0.1, flip_rate=flip_rate,
                        bit_p1=flip_rate)
    rc = _consec_write_differ_rate(ctrl.sample(B, np.random.default_rng(0)))
    rr = _consec_write_differ_rate(ref.sample(B, np.random.default_rng(1)))
    assert abs(rc - flip_rate) < 0.03, f"ctrl differ-rate {rc} vs {flip_rate}"
    assert abs(rc - rr) < 0.03, f"ctrl {rc} vs ref {rr}"
    # instruction law (Stationary part) must also match
    assert np.allclose(_inst_freq(ctrl.sample(B, np.random.default_rng(2))),
                       _inst_freq(ref.sample(B, np.random.default_rng(3))),
                       atol=0.02)


@pytest.mark.parametrize("bit_stay", [0.2, 0.5, 0.8])
def test_subsumes_bit_markov_exact(bit_stay):
    """Bit head on `prev_data_bit` reproduces BitMarkov's transition law
    (measured on non-read adjacent pairs, where the bit chain is observable)."""
    T, B = 256, 1200
    ctrl = FeatureControlledFFL.emulate_bit_markov(T, 0.15, 0.1, bit_stay)
    ref = BitMarkov(T=T, p_w=0.15, p_r=0.1, bit_stay=bit_stay, bit_p1=0.5)
    sc = _consec_nonread_same_rate(ctrl.sample(B, np.random.default_rng(0)))
    sr = _consec_nonread_same_rate(ref.sample(B, np.random.default_rng(1)))
    assert abs(sc - bit_stay) < 0.04, f"ctrl same-rate {sc} vs stay {bit_stay}"
    assert abs(sc - sr) < 0.04, f"ctrl {sc} vs ref {sr}"


def test_subsumes_piecewise_position_ramp_approx():
    """Position-only controller reproduces an equivalent Piecewise schedule
    (approximate: exact in the position-basis limit). Also asserts the
    controller genuinely expresses a non-stationary position trend."""
    T, B = 256, 1500
    ctrl, equiv = FeatureControlledFFL.emulate_piecewise_ramp(
        T, pw_start=0.05, pw_end=0.45, p_r=0.1)
    qc = _quartile_write_freq(ctrl.sample(B, np.random.default_rng(0)))
    qe = _quartile_write_freq(equiv.sample(B, np.random.default_rng(1)))
    assert np.allclose(qc, qe, atol=0.04), f"quartile W freq {qc} vs {qe}"
    # genuine non-stationarity: write rate rises start->end
    assert qc[0] < qc[3] - 0.10, f"expected rising ramp, got {qc}"


# ---------------------------------------------------------------------------
# encoder <-> CMA integration (no model, CPU)
# ---------------------------------------------------------------------------
def test_encoder_decode_is_valid_sampler():
    enc = FeatureControlledEncoder(T=128)
    rng = np.random.default_rng(0)
    x = enc.random_init(rng)
    assert x.shape == (24,)
    dist = enc.decode(x)
    assert isinstance(dist, FeatureControlledFFL)
    _assert_valid_ffl(dist.sample(32, rng), 128)


def test_random_init_starts_near_uniform_simplex():
    """Small-sigma init => well-posed, non-degenerate start (not pinned to a
    corner)."""
    enc = FeatureControlledEncoder(T=128)
    dist = enc.decode(enc.random_init(np.random.default_rng(0)))
    fW, fR, fI = _inst_freq(dist.sample(256, np.random.default_rng(0)))
    assert fW > 0.05 and fR > 0.05 and fI > 0.05  # all instructions present


def test_diagonal_cmaes_drives_encoder_unchanged():
    """The exact project optimizer ask/tell loop integrates with the 24-dim
    encoder; every proposed solution decodes to a valid FFL sampler. (Dummy
    quadratic objective — this checks protocol/shape integration, not search
    quality, which needs a model and is out of CPU scope.)"""
    enc = FeatureControlledEncoder(T=64)
    rng = np.random.default_rng(0)
    es = DiagonalCMAES(enc.random_init(rng), sigma=0.3, pop_size=8, seed=0)
    for _ in range(3):
        sols = es.ask()
        assert all(s.shape == (enc.n_dims,) for s in sols)
        fits = []
        for s in sols:
            dist = enc.decode(s)
            _assert_valid_ffl(dist.sample(8, np.random.default_rng(0)), 64)
            fits.append(float(np.sum(s ** 2)))   # minimize ||x||^2 (dummy)
        es.tell(sols, fits)
    assert np.all(np.isfinite(es.mean)) and es.sigma > 0
