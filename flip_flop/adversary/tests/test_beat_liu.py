"""PLAN_beat_liu_r4 mechanism tests: §1 gap helper + objective gate/hinge/
two-checkpoint, §2.A/B encoders, §2.C descriptor-grid pooling."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from flip_flop.data import W, R, I, ZERO, write_read_gaps, interleave
from flip_flop.adversary.distribution import (AntiCorrelatedBits,
                                              MixtureTwoStationary, Stationary)
from flip_flop.adversary.objective import fitness, seed_averaged_fitness
from flip_flop.adversary.search import (AntiCorrelatedBitsEncoder,
                                        StationaryMixtureEncoder)
from flip_flop.model import FFLMLSTM, FFLMTransformer


# ---------------------------------------------------------------------------
# §1  write_read_gaps
# ---------------------------------------------------------------------------
def test_write_read_gaps_known_value():
    # inst = [W, R, I, R]  data arbitrary. Token positions: W@0, R@2, I@4, R@6.
    # gap(first R) = (1-0)*2 = 2 tokens; gap(second R) = (3-0)*2 = 6 tokens.
    inst = np.array([[W, R, I, R]], dtype=np.int64)
    data = np.array([[1, 1, 0, 1]], dtype=np.int64)
    tok = interleave(inst, data)
    mean, p90 = write_read_gaps(tok)
    assert mean == pytest.approx((2 + 6) / 2)
    assert p90 == pytest.approx(np.percentile([2, 6], 90))


def test_write_read_gaps_intermediate_write_resets():
    # inst = [W, W, R]  -> read's most-recent write is slot 1, gap = (2-1)*2 = 2.
    inst = np.array([[W, W, R]], dtype=np.int64)
    data = np.array([[0, 1, 1]], dtype=np.int64)
    mean, p90 = write_read_gaps(interleave(inst, data))
    assert mean == pytest.approx(2.0)


def test_write_read_gaps_no_reads_returns_zero():
    inst = np.array([[W, W, W]], dtype=np.int64)
    data = np.array([[0, 1, 0]], dtype=np.int64)
    assert write_read_gaps(interleave(inst, data)) == (0.0, 0.0)


def test_write_read_gaps_orders_ffl_tails():
    """Sparse FFL(0.98) must have a far larger gap than dense FFL(0.1)."""
    sparse = Stationary(T=512, p_w=0.01, p_r=0.01).sample(128, np.random.default_rng(0))
    dense = Stationary(T=512, p_w=0.45, p_r=0.45).sample(128, np.random.default_rng(0))
    assert write_read_gaps(sparse)[0] > 10 * write_read_gaps(dense)[0]


# ---------------------------------------------------------------------------
# §1  objective: gap fields, gap gate, regret hinge, two-checkpoint
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models():
    t = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    t2 = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    l = FFLMLSTM(hidden_size=32)
    for m in (t, t2, l):
        m.eval()
    return t, t2, l


def test_fitness_populates_gap_fields(models):
    t, _, l = models
    fr = fitness(Stationary(T=64, p_w=0.1, p_r=0.1), t, l,
                 n=64, batch_size=32, device="cpu", rng=np.random.default_rng(0))
    assert fr.is_valid
    assert fr.gap > 0.0 and fr.gap_p90 >= fr.gap - 1e-9
    assert fr.T_glitch2 == 0.0   # no transformer2 supplied


def test_min_gap_gate_rejects_when_threshold_huge(models):
    t, _, l = models
    common = dict(n=64, batch_size=32, device="cpu")
    d = Stationary(T=64, p_w=0.1, p_r=0.1)
    ok = fitness(d, t, l, rng=np.random.default_rng(0), min_gap=0.0, **common)
    bad = fitness(d, t, l, rng=np.random.default_rng(0), min_gap=1e9, **common)
    assert ok.is_valid and np.isfinite(ok.fitness)
    assert (not bad.is_valid) and bad.fitness == float("-inf")
    assert bad.gap > 0.0  # descriptor still recorded on the reject path


def test_min_reads_gate_still_works(models):
    t, _, l = models
    bad = fitness(Stationary(T=64, p_w=0.1, p_r=0.1), t, l, n=64, batch_size=32,
                  device="cpu", rng=np.random.default_rng(0),
                  min_reads_per_seq=1e9)
    assert not bad.is_valid and bad.fitness == float("-inf")


def test_regret_hinge_formula(models):
    t, _, l = models
    fr = fitness(Stationary(T=64, p_w=0.1, p_r=0.1), t, l,
                 n=128, batch_size=32, device="cpu",
                 rng=np.random.default_rng(0), objective_mode="regret",
                 regret_hinge_coef=5.0, regret_hinge_tol=1e-3)
    expected = (fr.T_glitch - fr.lstm_glitch) \
        - 5.0 * max(0.0, fr.lstm_glitch - 1e-3)
    assert fr.fitness == pytest.approx(expected, abs=1e-6)


def test_two_checkpoint_takes_min_regret(models):
    t, t2, l = models
    fr = fitness(Stationary(T=64, p_w=0.1, p_r=0.1), t, l,
                 n=128, batch_size=32, device="cpu",
                 rng=np.random.default_rng(0), objective_mode="regret",
                 transformer2=t2, regret_hinge_coef=5.0, regret_hinge_tol=1e-3)
    assert fr.T_glitch2 > 0.0
    regret = min(fr.T_glitch - fr.lstm_glitch, fr.T_glitch2 - fr.lstm_glitch)
    expected = regret - 5.0 * max(0.0, fr.lstm_glitch - 1e-3)
    assert fr.fitness == pytest.approx(expected, abs=1e-6)


def test_seed_averaged_fitness_carries_new_fields(models):
    t, t2, l = models
    fr = seed_averaged_fitness(Stationary(T=64, p_w=0.1, p_r=0.1), t, l,
                               n=64, batch_size=32, device="cpu", n_seeds=2,
                               objective_mode="regret", transformer2=t2)
    assert fr.is_valid and fr.gap > 0.0 and fr.T_glitch2 > 0.0


# ---------------------------------------------------------------------------
# §2.A / §2.B  encoders
# ---------------------------------------------------------------------------
def test_stationary_mixture_encoder_decodes_valid():
    enc = StationaryMixtureEncoder(T=128)
    assert enc.n_dims == 7
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = enc.decode(rng.standard_normal(7) * 2.0)
        assert isinstance(d, MixtureTwoStationary)
        assert 0.0 <= d.lam <= 1.0
        assert d.p_w_a + d.p_r_a <= 1.0 + 1e-9
        assert d.p_w_b + d.p_r_b <= 1.0 + 1e-9
        d.sample(8, rng)  # must not raise (valid params)


def test_anti_correlated_bits_encoder_decodes_valid():
    enc = AntiCorrelatedBitsEncoder(T=128)
    assert enc.n_dims == 4
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = enc.decode(rng.standard_normal(4) * 2.0)
        assert isinstance(d, AntiCorrelatedBits)
        assert 0.0 <= d.rho <= 1.0
        assert d.p_w + d.p_r <= 1.0 + 1e-9
        d.sample(8, rng)


def test_encoders_satisfy_protocol():
    from flip_flop.adversary.search import Encoder
    for enc in (StationaryMixtureEncoder(T=64), AntiCorrelatedBitsEncoder(T=64)):
        assert isinstance(enc, Encoder)  # runtime-checkable structural check


# ---------------------------------------------------------------------------
# §2.C  descriptor-grid pooling
# ---------------------------------------------------------------------------
from flip_flop.scripts.pool_breaking_points import (_density_bin,
                                                    _descriptor_grid_pool,
                                                    _gap_bin)


def test_gap_and_density_bins():
    assert _gap_bin(4.0, 512) == 0 and _gap_bin(15.9, 512) == 0
    assert _gap_bin(16.0, 512) == 1 and _gap_bin(63.9, 512) == 1
    assert _gap_bin(64.0, 512) == 2 and _gap_bin(255.9, 512) == 2
    assert _gap_bin(256.0, 512) == 3 and _gap_bin(99999, 512) == 3
    assert _density_bin(3.0) == 0 and _density_bin(7.99) == 0
    assert _density_bin(8.0) == 1 and _density_bin(500) == 1


def _rec(gp, rd, fit):
    return {"config": {"name": "stationary", "T": 512, "p_w": 0.1,
                       "p_r": 0.1, "bit_p1": 0.5},
            "fitness": fit, "T_glitch": fit + 0.01, "lstm_glitch": 0.0,
            "read_density": rd, "gap_p90": gp, "is_valid": True}


def test_descriptor_grid_keeps_one_per_cell_and_spreads():
    recs = [
        _rec(10, 4, 0.3), _rec(12, 5, 0.5),   # same cell (0,0): keep 0.5
        _rec(100, 4, 0.9),                    # interior high-regret cell (2,0)
        _rec(400, 4, 0.7),                    # gap-extreme cell (3,0)
        _rec(300, 20, 0.8),                   # gap-extreme cell (3,1)
    ]
    pooled = _descriptor_grid_pool(recs, T=512, max_families=8)
    cells = {tuple(p["cell"]) for p in pooled}
    assert len(pooled) == 4, "the two (0,0) records must collapse to one"
    assert (2, 0) in cells and (3, 0) in cells and (3, 1) in cells, \
        "gap-axis extremes must survive even at lower regret than interior"
    by_cell = {tuple(p["cell"]): p for p in pooled}
    assert by_cell[(0, 0)]["fitness"] == 0.5  # higher-regret kept in shared cell


def test_descriptor_grid_caps_max_families():
    recs = [_rec(g, d, f) for (g, d, f) in
            [(10, 4, 0.1), (40, 4, 0.2), (100, 4, 0.3), (400, 4, 0.4),
             (10, 20, 0.5), (40, 20, 0.6), (100, 20, 0.7), (400, 20, 0.8)]]
    pooled = _descriptor_grid_pool(recs, T=512, max_families=3)
    assert len(pooled) == 3
    # capped to the 3 highest-regret cells
    assert sorted(p["fitness"] for p in pooled) == [0.6, 0.7, 0.8]


def test_descriptor_grid_pool_end_to_end(tmp_path):
    """Full CLI: synthetic log -> grid pool -> sampler.json + breaking_points."""
    import subprocess, sys
    recs = [_rec(10, 4, 0.5), _rec(400, 4, 0.7), _rec(300, 20, 0.8)]
    log = tmp_path / "adversary_log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    out = tmp_path / "pooled"
    r = subprocess.run(
        [sys.executable, "-m", "flip_flop.scripts.pool_breaking_points",
         "--logs", str(log), "--out_dir", str(out),
         "--min_t_glitch", "0.05", "--max_families", "8"],
        capture_output=True, text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
    )
    assert r.returncode == 0, r.stderr
    sampler = json.loads((out / "sampler.json").read_text())
    # 3 grid families + 2 Liu tails
    assert len(sampler["families"]) == 5
    names = [f["name"] for f in sampler["families"]]
    assert any("liu_ffl_098" in n for n in names)
    assert any("liu_ffl_010" in n for n in names)
