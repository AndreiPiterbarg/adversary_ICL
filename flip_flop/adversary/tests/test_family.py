"""Tests for family extraction: interpolation, α pull-back, mixture, planted twins."""
import json

import numpy as np
import pytest

from flip_flop.adversary.distribution import (Piecewise, Planted, Stationary)
from flip_flop.adversary.family import (ClusterFamily, MixtureFamily,
                                         PassthroughFamily,
                                         _bisect_alpha,
                                         extract_families_from_adversary_log,
                                         interpolate_params, pull_back_alpha,
                                         pull_back_alpha_mixture)
from flip_flop.data import W, R, I, ZERO, ONE
from flip_flop.model import FFLMLSTM, FFLMTransformer


def _assert_valid_ffl(tokens, T):
    x = tokens.numpy()
    assert (x[:, 0] == W).all()
    assert (x[:, -2] == R).all()
    assert np.isin(x[:, 0::2], [W, R, I]).all()
    assert np.isin(x[:, 1::2], [ZERO, ONE]).all()


# ---------------------------------------------------------------------------
# Parameter interpolation
# ---------------------------------------------------------------------------
def test_interpolate_stationary():
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.95}
    d0 = interpolate_params(base, adv, 0.0)
    d1 = interpolate_params(base, adv, 1.0)
    d_half = interpolate_params(base, adv, 0.5)
    assert d0.p_w == pytest.approx(base["p_w"])
    assert d1.p_w == pytest.approx(adv["p_w"])
    assert d_half.p_w == pytest.approx((base["p_w"] + adv["p_w"]) / 2)
    assert d_half.bit_p1 == pytest.approx(0.725)
    _assert_valid_ffl(d_half.sample(32, np.random.default_rng(0)), 64)


def test_interpolate_piecewise_lifts_stationary_base():
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "piecewise", "T": 64, "segments": [
        [0.0, 1.0, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]
    ]}
    d = interpolate_params(base, adv, 0.5)
    assert isinstance(d, Piecewise)
    # First segment: (0.1+1.0)/2 = 0.55
    assert d.segments[0][1] == pytest.approx(0.55)
    _assert_valid_ffl(d.sample(32, np.random.default_rng(0)), 64)


# ---------------------------------------------------------------------------
# Pull-back α
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models_cpu():
    t = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    l = FFLMLSTM(hidden_size=32)
    t.eval(); l.eval()
    return t, l


def test_pull_back_returns_valid_alpha(models_cpu):
    t, l = models_cpu
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.95}
    rng = np.random.default_rng(0)
    alpha, t_err, l_err = pull_back_alpha(
        base_cfg=base, adv_cfg=adv, transformer=t, lstm=l,
        device="cpu", T=64, rng=rng, n_probe=64, ref_glitch=0.5,
    )
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= t_err <= 1.0
    assert 0.0 <= l_err <= 1.0


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------
def test_extract_families_full_path(models_cpu, tmp_path):
    """Full pipeline: log -> dedup -> top-K -> α-bisect -> ClusterFamily."""
    t, l = models_cpu
    recs = []
    for _ in range(15):
        recs.append({
            "config": {"name": "stationary", "T": 64,
                       "p_w": 0.005 + 0.001 * np.random.rand(),
                       "p_r": 0.005 + 0.001 * np.random.rand(),
                       "bit_p1": 0.9 + 0.05 * np.random.rand()},
            "fitness": 0.8, "T_glitch": 0.8, "lstm_glitch": 0.0, "is_valid": True,
        })
    for _ in range(15):
        recs.append({
            "config": {"name": "stationary", "T": 64,
                       "p_w": 0.4 + 0.02 * np.random.rand(),
                       "p_r": 0.4 + 0.02 * np.random.rand(),
                       "bit_p1": 0.5},
            "fitness": 0.6, "T_glitch": 0.6, "lstm_glitch": 0.0, "is_valid": True,
        })
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    fams = extract_families_from_adversary_log(
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu", top_k=5,
    )
    assert 1 <= len(fams) <= 5
    for fam in fams:
        assert isinstance(fam, ClusterFamily)
        assert 0.0 <= fam.alpha <= 1.0
        tokens = fam.sample(16, np.random.default_rng(0))
        _assert_valid_ffl(tokens, 64)


def test_extract_families_fallback_to_passthrough(tmp_path):
    """When models unavailable, falls back to PassthroughFamily."""
    recs = [{
        "config": {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.9},
        "fitness": 0.8, "T_glitch": 0.8, "lstm_glitch": 0.0, "is_valid": True,
    }]
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    fams = extract_families_from_adversary_log(str(path), top_k=5)
    assert len(fams) == 1
    assert isinstance(fams[0], PassthroughFamily)


# ---------------------------------------------------------------------------
# MixtureFamily
# ---------------------------------------------------------------------------
def test_mixture_family_alpha_zero_is_pure_base():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.0)
    tokens = fam.sample(64, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    inst = tokens[:, 0::2].numpy()
    # Under base FFL with p_w=0.1, frac of W at pos 1 is ~10%; planted decoy
    # would force it to 100%, so α=0 should keep frac low.
    frac_w_at_1 = (inst[:, 1] == W).mean()
    assert frac_w_at_1 < 0.30, f"alpha=0 should sample base; saw frac W at pos 1 = {frac_w_at_1}"


def test_mixture_family_alpha_one_is_pure_adv():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=1.0)
    tokens = fam.sample(64, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    inst = tokens[:, 0::2].numpy()
    assert (inst[:, 1] == W).all()


def test_mixture_family_intermediate_alpha_is_valid():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.5)
    tokens = fam.sample(256, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    inst = tokens[:, 0::2].numpy()
    frac_w_at_1 = (inst[:, 1] == W).mean()
    assert 0.40 < frac_w_at_1 < 0.65, f"alpha=0.5 expected ~0.5 mix; got {frac_w_at_1}"


def test_mixture_family_to_dict_roundtrip_kind():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.5,
                        cluster_mean_glitch=0.97, name="planted_decoy_a0.50")
    d = fam.to_dict()
    assert d["kind"] == "MixtureFamily"
    assert d["alpha"] == 0.5
    assert d["base_dist"]["name"] == "stationary"
    assert d["adv_dists"][0]["name"] == "planted"
    assert d["n_adv_variants"] == 1


def test_mixture_family_t_mismatch_raises():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=128, template="decoy", params={"k_early": 1, "k_true": 60, "b_decoy": 0})
    with pytest.raises(AssertionError):
        MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.5)


def test_mixture_family_alpha_out_of_range_raises():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    with pytest.raises(AssertionError):
        MixtureFamily(base_dist=base, adv_dists=[adv], alpha=1.5)
    with pytest.raises(AssertionError):
        MixtureFamily(base_dist=base, adv_dists=[adv], alpha=-0.1)


def test_pull_back_alpha_mixture_returns_valid_alpha(models_cpu):
    t, l = models_cpu
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    rng = np.random.default_rng(0)
    alpha, t_err, l_err = pull_back_alpha_mixture(
        base_dist=base, adv_dists=[adv], transformer=t, lstm=l,
        device="cpu", rng=rng, n_probe=64,
    )
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= t_err <= 1.0
    assert 0.0 <= l_err <= 1.0


def test_pull_back_alpha_mixture_accepts_single_dist_for_compat(models_cpu):
    t, l = models_cpu
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    rng = np.random.default_rng(0)
    alpha, _, _ = pull_back_alpha_mixture(
        base_dist=base, adv_dists=adv, transformer=t, lstm=l,
        device="cpu", rng=rng, n_probe=64,
    )
    assert 0.0 <= alpha <= 1.0


def test_mixture_family_with_two_variants_picks_uniformly():
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv0 = Planted(T=64, template="decoy",
                    params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    adv1 = Planted(T=64, template="decoy",
                    params={"k_early": 1, "k_true": 28, "b_decoy": 1})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv0, adv1], alpha=1.0)
    tokens = fam.sample(512, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    data_at_early = tokens[:, 3].numpy() - 3
    frac_one = data_at_early.mean()
    assert 0.40 < frac_one < 0.60, f"expected ~0.5, got {frac_one}"


def test_planted_bit_flip_twins_decoy():
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "decoy", "filler_p_i": 1.0,
            "params": {"k_early": 5, "k_true": 28, "b_decoy": 0}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 2
    assert twins[0]["params"]["b_decoy"] == 0
    assert twins[1]["params"]["b_decoy"] == 1


def test_planted_bit_flip_twins_distractor():
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "distractor", "filler_p_i": 1.0,
            "params": {"k_true": 5, "d": 1, "b_true": 1}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 2
    assert twins[1]["params"]["b_true"] == 0


def test_planted_bit_flip_twins_gap_no_bit():
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "gap", "filler_p_i": 1.0,
            "params": {"k_write": 5}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 1


# ---------------------------------------------------------------------------
# Generic bisection helper
# ---------------------------------------------------------------------------
def test_bisect_alpha_finds_target_on_monotone_curve():
    def eval_alpha(alpha):
        return alpha, 0.0
    alpha, t_err, _ = _bisect_alpha(eval_alpha, target_t_glitch=0.5,
                                     max_lstm_glitch=0.01, tol=0.02)
    assert abs(alpha - 0.5) < 0.05, f"bisection on linear curve expected ~0.5, got {alpha}"
    assert abs(t_err - 0.5) < 0.05


def test_bisect_alpha_picks_alpha_one_when_adv_too_easy():
    def eval_alpha(alpha):
        return 0.2 * alpha, 0.0
    alpha, _, _ = _bisect_alpha(eval_alpha, target_t_glitch=0.5)
    assert alpha == 1.0


def test_bisect_alpha_skips_when_lstm_fails_at_one():
    def eval_alpha(alpha):
        return alpha, 0.5
    alpha, _, l_err = _bisect_alpha(eval_alpha, max_lstm_glitch=0.01)
    assert alpha == 1.0
    assert l_err == 0.5


def test_extract_planted_emits_mixture_family(models_cpu, tmp_path):
    """Planted candidates produce MixtureFamily with bit-flipped twin."""
    t, l = models_cpu
    recs = []
    for _ in range(8):
        recs.append({
            "config": {"name": "planted", "T": 64, "template": "decoy",
                       "filler_p_i": 1.0,
                       "params": {"k_early": 1 + np.random.randint(3),
                                  "k_true": 25 + np.random.randint(3),
                                  "b_decoy": 0}},
            "fitness": 0.95, "T_glitch": 0.95, "lstm_glitch": 0.0, "is_valid": True,
        })
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    fams = extract_families_from_adversary_log(
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu",
        top_k=5, min_t_glitch=0.01,
    )
    assert len(fams) >= 1
    planted_fams = [f for f in fams if isinstance(f, MixtureFamily)]
    assert len(planted_fams) >= 1
    for fam in planted_fams:
        assert "_a" in fam.name
        assert 0.0 <= fam.alpha <= 1.0
        # decoy template -> 2 adv_dists (orig + bit-flipped twin)
        assert len(fam.adv_dists) == 2, \
            f"decoy should yield 2 adv variants, got {len(fam.adv_dists)}"
        b_decoys = sorted(d.params["b_decoy"] for d in fam.adv_dists)
        assert b_decoys == [0, 1], f"expected b_decoys=[0,1], got {b_decoys}"
        tokens = fam.sample(16, np.random.default_rng(0))
        _assert_valid_ffl(tokens, 64)
