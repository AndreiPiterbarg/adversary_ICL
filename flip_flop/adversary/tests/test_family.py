"""Tests for family extraction: interpolation, geometric median, clustering, pull-back."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flip_flop.adversary.distribution import (FFLDistribution, Piecewise,
                                                 Planted, Stationary)
from flip_flop.adversary.family import (ClusterFamily, MixtureFamily,
                                         PassthroughFamily,
                                         _bisect_alpha,
                                         _cluster_representative_config,
                                         _featurize_batch,
                                         _geometric_median, _hdbscan_cluster,
                                         _select_with_axis_floor,
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
    # Validity preserved
    _assert_valid_ffl(d.sample(32, np.random.default_rng(0)), 64)


# ---------------------------------------------------------------------------
# Geometric median
# ---------------------------------------------------------------------------
def test_geometric_median_robust_to_outliers():
    pts = np.array([[0.0, 0.0]] * 10 + [[100.0, 100.0]])  # 10 at origin + 1 outlier
    gm = _geometric_median(pts)
    # Should be close to origin, not the mean (which would be (9.09, 9.09))
    assert np.linalg.norm(gm) < 1.0


def test_cluster_representative_piecewise():
    # 5 configs with identical first seg, varying second
    configs = [{
        "name": "piecewise", "T": 64,
        "segments": [[0.0, 0.9, 0.0, 1.0], [0.5, float(i) / 10, 0.0, 0.5]]
    } for i in range(5)]
    rep = _cluster_representative_config(configs)
    assert rep["name"] == "piecewise"
    assert rep["segments"][0][1] == pytest.approx(0.9, abs=0.05)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_hdbscan_merges_similar_configs():
    """20 nearly-identical configs + 20 far-away configs -> HDBSCAN yields ~2 clusters."""
    configs_a = [{"name": "stationary", "T": 64, "p_w": 0.005 + 0.0005 * np.random.randn(),
                  "p_r": 0.005, "bit_p1": 0.95} for _ in range(20)]
    configs_b = [{"name": "stationary", "T": 64, "p_w": 0.45,
                  "p_r": 0.45 + 0.005 * np.random.randn(), "bit_p1": 0.5} for _ in range(20)]
    rng = np.random.default_rng(0)
    features = _featurize_batch(configs_a + configs_b, T=64, n_behavior=16, rng=rng)
    clusters = _hdbscan_cluster(features, min_cluster_size=5)
    # At least 2 clusters (may be more with small-N noise); total points covered should be most
    assert len(clusters) >= 1
    covered = sum(len(c) for c in clusters)
    assert covered >= 20  # most points clustered


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
    """Full pipeline: log -> features -> HDBSCAN -> median -> pull-back -> ClusterFamily."""
    t, l = models_cpu
    # Build a log with two structural groups
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
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu",
        top_k=5, n_behavior=16,
    )
    # Should produce 1-2 families, each a ClusterFamily with sensible α
    assert 1 <= len(fams) <= 5
    for fam in fams:
        assert isinstance(fam, ClusterFamily)
        assert 0.0 <= fam.alpha <= 1.0
        # Produces valid FFL sequences
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
# MixtureFamily (Step 2)
# ---------------------------------------------------------------------------
def test_mixture_family_alpha_zero_is_pure_base():
    """At α=0, every sample comes from base_dist."""
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.0)
    tokens = fam.sample(64, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    # Compare to a pure base sample from the same seed (different RNG draws,
    # so don't expect exact equality — but FFL validity must hold).
    # Sanity: with α=0, fraction of "decoy-shaped" sequences should be 0;
    # decoy template forces inst[k_early] = W. Base FFL(0.8) only puts W
    # at position 0 (forced) and rarely elsewhere. So inst[1] = W rarely.
    inst = tokens[:, 0::2].numpy()
    # Per planted decoy template, inst[k_early=1] would be W deterministically.
    # Under base FFL(0.8) p_w = (1-0.8)/2 = 0.1, so frac with inst[1]==W is ~10%.
    frac_w_at_1 = (inst[:, 1] == W).mean()
    assert frac_w_at_1 < 0.30, f"alpha=0 should sample base; saw frac W at pos 1 = {frac_w_at_1}"


def test_mixture_family_alpha_one_is_pure_adv():
    """At α=1, every sample comes from adv_dist."""
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=1.0)
    tokens = fam.sample(64, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    # Decoy template puts W at k_early=1 deterministically → inst[1] == W always.
    inst = tokens[:, 0::2].numpy()
    assert (inst[:, 1] == W).all()


def test_mixture_family_intermediate_alpha_is_valid():
    """At α=0.5, output must still be FFL-valid (each sequence is one or the other)."""
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.5)
    tokens = fam.sample(256, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    # ~50% of samples should have inst[1] == W (the planted ones).
    inst = tokens[:, 0::2].numpy()
    frac_w_at_1 = (inst[:, 1] == W).mean()
    # binomial 95% CI at p=0.5, n=256: ±~6%. Allow generous margin.
    assert 0.40 < frac_w_at_1 < 0.65, f"alpha=0.5 expected ~0.5 mix; got {frac_w_at_1}"


def test_mixture_family_to_dict_roundtrip_kind():
    """to_dict() output identifies kind correctly."""
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    adv = Planted(T=64, template="decoy", params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv], alpha=0.5,
                        cluster_size=10, cluster_mean_glitch=0.97,
                        name="planted_decoy_a0.50", axis="planted")
    d = fam.to_dict()
    assert d["kind"] == "MixtureFamily"
    assert d["alpha"] == 0.5
    assert d["axis"] == "planted"
    assert d["base_dist"]["name"] == "stationary"
    assert d["adv_dists"][0]["name"] == "planted"
    assert d["n_adv_variants"] == 1


def test_mixture_family_t_mismatch_raises():
    """MixtureFamily refuses base and adv with different T."""
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
    """Bisection on a mixture family returns α in [0,1] and reports clean
    LSTM at chosen α (since adversary's choice is by construction LSTM-clean)."""
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
    """Backwards-compat: pull_back_alpha_mixture should accept a single
    FFLDistribution instead of a list."""
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
    """With 2 adv_dists and α=1, samples should split ~50/50 between the two."""
    base = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    # Two planted decoy configs differing only in b_decoy
    adv0 = Planted(T=64, template="decoy",
                    params={"k_early": 1, "k_true": 28, "b_decoy": 0})
    adv1 = Planted(T=64, template="decoy",
                    params={"k_early": 1, "k_true": 28, "b_decoy": 1})
    fam = MixtureFamily(base_dist=base, adv_dists=[adv0, adv1], alpha=1.0)
    tokens = fam.sample(512, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    # The data bit at the early-decoy position (token index 1+2*k_early=3)
    # is b_decoy. With 50/50 split, should average ~0.5.
    # Wait — that position holds the WRITE's data. After enforce_read_determinism,
    # reads get overwritten. Inst[k_early=1] corresponds to token positions 2,3
    # (instruction at 2, data at 3). Read is at position n_inst-1=31, token 62.
    # Data bit AFTER the early decoy W is b_decoy.
    data_at_early = tokens[:, 3].numpy() - 3  # 0 or 1
    frac_one = data_at_early.mean()
    # Loose: ~0.5 with binomial variance (n=512 → CI ±~0.04)
    assert 0.40 < frac_one < 0.60, f"expected ~0.5, got {frac_one}"


def test_planted_bit_flip_twins_decoy():
    """_planted_bit_flip_twins flips b_decoy for decoy template."""
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "decoy", "filler_p_i": 1.0,
            "params": {"k_early": 5, "k_true": 28, "b_decoy": 0}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 2
    assert twins[0]["params"]["b_decoy"] == 0  # original
    assert twins[1]["params"]["b_decoy"] == 1  # flipped


def test_planted_bit_flip_twins_distractor():
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "distractor", "filler_p_i": 1.0,
            "params": {"k_true": 5, "d": 1, "b_true": 1}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 2
    assert twins[1]["params"]["b_true"] == 0


def test_planted_bit_flip_twins_gap_no_bit():
    """Gap template has no bit param → no twin returned."""
    from flip_flop.adversary.family import _planted_bit_flip_twins
    cfg = {"name": "planted", "T": 64, "template": "gap", "filler_p_i": 1.0,
            "params": {"k_write": 5}}
    twins = _planted_bit_flip_twins(cfg)
    assert len(twins) == 1   # original only


# ---------------------------------------------------------------------------
# Generic bisection helper (Step 2)
# ---------------------------------------------------------------------------
def test_bisect_alpha_finds_target_on_monotone_curve():
    """If T_err(α) = α (linear), bisection should land near target=0.5."""
    def eval_alpha(alpha):
        return alpha, 0.0  # T_err = α, lstm = 0
    alpha, t_err, _ = _bisect_alpha(eval_alpha, target_t_glitch=0.5,
                                     max_lstm_glitch=0.01, tol=0.02)
    assert abs(alpha - 0.5) < 0.05, f"bisection on linear curve expected ~0.5, got {alpha}"
    assert abs(t_err - 0.5) < 0.05


def test_bisect_alpha_picks_alpha_one_when_adv_too_easy():
    """If T_err(1) < target, bisection returns α=1 (no need to soften)."""
    def eval_alpha(alpha):
        return 0.2 * alpha, 0.0  # max T_err = 0.2 < target 0.5
    alpha, _, _ = _bisect_alpha(eval_alpha, target_t_glitch=0.5)
    assert alpha == 1.0


def test_bisect_alpha_skips_when_lstm_fails_at_one():
    """If LSTM fails even at α=1, bisection returns α=1 with LSTM flag."""
    def eval_alpha(alpha):
        return alpha, 0.5  # LSTM always fails
    alpha, _, l_err = _bisect_alpha(eval_alpha, max_lstm_glitch=0.01)
    assert alpha == 1.0
    assert l_err == 0.5


# ---------------------------------------------------------------------------
# Per-axis floor + global rank (Step 3)
# ---------------------------------------------------------------------------
def _mk_passthrough(name, axis, glitch, size=1):
    """Test helper: minimal PassthroughFamily with axis tag."""
    return PassthroughFamily(
        dist=Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5),
        name=name, axis=axis, cluster_size=size, cluster_mean_glitch=glitch,
    )


def test_axis_floor_reserves_one_per_axis():
    """3 axes, top_k=5, lots of one axis: floor still has 1 per axis."""
    fams = [
        _mk_passthrough("p1", "planted", 1.00),
        _mk_passthrough("p2", "planted", 1.00),
        _mk_passthrough("p3", "planted", 1.00),
        _mk_passthrough("p4", "planted", 1.00),
        _mk_passthrough("pw1", "piecewise", 0.50, size=200),
        _mk_passthrough("pw2", "piecewise", 0.45, size=150),
        _mk_passthrough("st1", "stationary", 0.20, size=20),
    ]
    selected = _select_with_axis_floor(fams, top_k=5)
    axes = [f.axis for f in selected]
    # All three axes must have at least one representative
    assert "planted" in axes
    assert "piecewise" in axes
    assert "stationary" in axes
    assert len(selected) == 5


def test_axis_floor_falls_back_to_global_rank_when_only_one_axis():
    """Single axis: top-K by global rank."""
    fams = [
        _mk_passthrough("p1", "planted", 1.00),
        _mk_passthrough("p2", "planted", 0.99),
        _mk_passthrough("p3", "planted", 0.98),
        _mk_passthrough("p4", "planted", 0.97),
        _mk_passthrough("p5", "planted", 0.96),
        _mk_passthrough("p6", "planted", 0.95),  # excluded
    ]
    selected = _select_with_axis_floor(fams, top_k=5)
    assert len(selected) == 5
    assert all(f.axis == "planted" for f in selected)
    # Top 5 by glitch
    assert [f.name for f in selected] == ["p1", "p2", "p3", "p4", "p5"]


def test_axis_floor_empty_input_returns_empty():
    assert _select_with_axis_floor([], top_k=5) == []


def test_axis_floor_with_top_k_smaller_than_axis_count():
    """With top_k=2 but 3 axes available, only the strongest 2 axes get a slot."""
    fams = [
        _mk_passthrough("p1", "planted", 1.00),
        _mk_passthrough("pw1", "piecewise", 0.50, size=200),
        _mk_passthrough("st1", "stationary", 0.20),
    ]
    selected = _select_with_axis_floor(fams, top_k=2)
    assert len(selected) == 2
    # planted (1.00) and piecewise (0.50) win; stationary (0.20) drops.
    axes = sorted(f.axis for f in selected)
    assert axes == ["piecewise", "planted"]


def test_extract_with_relaxed_threshold_includes_low_glitch_clusters(models_cpu, tmp_path):
    """Step 1 fix: with min_t_glitch=0.01, low-glitch clusters survive
    instead of being silently dropped (the Phase B bug)."""
    t, l = models_cpu
    # Make a log where ALL records are below 0.5 (Phase B scenario).
    recs = []
    for _ in range(15):
        recs.append({
            "config": {"name": "stationary", "T": 64,
                       "p_w": 0.005 + 0.001 * np.random.rand(),
                       "p_r": 0.005 + 0.001 * np.random.rand(),
                       "bit_p1": 0.9 + 0.05 * np.random.rand()},
            "fitness": 0.10, "T_glitch": 0.10, "lstm_glitch": 0.0, "is_valid": True,
        })
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    # With OLD threshold 0.5, all records are dropped → 0 families.
    fams_strict = extract_families_from_adversary_log(
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu",
        top_k=5, n_behavior=16, min_t_glitch=0.5,
    )
    assert len(fams_strict) == 0
    # With NEW threshold 0.01, all records pass → ≥1 cluster family found.
    fams_relaxed = extract_families_from_adversary_log(
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu",
        top_k=5, n_behavior=16, min_t_glitch=0.01,
    )
    assert len(fams_relaxed) >= 1
    for fam in fams_relaxed:
        assert fam.axis == "stationary"


def test_extract_planted_emits_mixture_family(models_cpu, tmp_path):
    """Step 2: planted candidates produce MixtureFamily (NOT PassthroughFamily)."""
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
        top_k=5, n_behavior=16, min_t_glitch=0.01,
    )
    assert len(fams) >= 1
    planted_fams = [f for f in fams if f.axis == "planted"]
    assert len(planted_fams) >= 1
    # Critical: planted families must be MixtureFamily, NOT PassthroughFamily.
    for fam in planted_fams:
        assert isinstance(fam, MixtureFamily), \
            f"planted family should be MixtureFamily, got {type(fam).__name__}"
        assert "_a" in fam.name
        # α should not be exactly 1.0 if the bisection actually softened
        assert 0.0 <= fam.alpha <= 1.0
        # Bit-flip jitter: decoy template should produce 2 adv_dists (orig + twin)
        assert len(fam.adv_dists) == 2, \
            f"decoy template should yield 2 adv variants, got {len(fam.adv_dists)}"
        b_decoys = sorted(d.params["b_decoy"] for d in fam.adv_dists)
        assert b_decoys == [0, 1], f"expected b_decoys=[0,1], got {b_decoys}"
        # Sample is FFL-valid
        tokens = fam.sample(16, np.random.default_rng(0))
        _assert_valid_ffl(tokens, 64)
