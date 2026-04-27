"""End-to-end smoke tests for the adversary pipeline.

Uses fresh (untrained) models so the tests don't depend on existing
checkpoints. Glitch rates will be near 0.5 (uninformed) but that's enough to
validate the plumbing: sampling -> objective -> search -> IO.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from flip_flop.adversary.distribution import Stationary, Piecewise, Planted
from flip_flop.adversary.objective import fitness
from flip_flop.adversary.search import (PiecewiseEncoder, cma_search,
                                        grid_search)
from flip_flop.adversary.io import save_top_k, dump_final_eval
from flip_flop.model import FFLMTransformer, FFLMLSTM


@pytest.fixture(scope="module")
def models():
    """Tiny fresh Transformer + LSTM, CPU."""
    transformer = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    lstm = FFLMLSTM(hidden_size=32)
    transformer.eval()
    lstm.eval()
    return transformer, lstm


def test_fitness_runs(models):
    transformer, lstm = models
    dist = Stationary(T=64, p_w=0.1, p_r=0.1)
    fr = fitness(dist, transformer, lstm,
                 n=16, batch_size=8, device="cpu",
                 rng=np.random.default_rng(0))
    assert fr.is_valid
    assert 0.0 <= fr.T_glitch <= 1.0
    assert 0.0 <= fr.lstm_glitch <= 1.0


def test_grid_search_end_to_end(models):
    transformer, lstm = models
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        def obj(dist):
            return fitness(dist, transformer, lstm, n=16, batch_size=8,
                           device="cpu", rng=rng)
        def factory(params):
            return Stationary(T=64, **params)
        grid = {"p_w": [0.1, 0.2], "p_r": [0.1, 0.2], "bit_p1": [0.5]}
        results = grid_search(factory, grid, obj, tmp, log_every=100)
        assert len(results) == 4
        assert all(r.is_valid for r in results)
        log_path = os.path.join(tmp, "adversary_log.jsonl")
        lines = open(log_path).read().strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            rec = json.loads(line)
            assert "fitness" in rec and "T_glitch" in rec
        top = save_top_k(results, k=2, out_dir=tmp)
        assert len(top) == 2
        dump_final_eval(top, transformer, lstm,
                        n=16, batch_size=8, device="cpu", n_seeds=1,
                        out_dir=tmp)
        final_path = os.path.join(tmp, "final_eval.jsonl")
        assert os.path.exists(final_path)
        final_lines = open(final_path).read().strip().split("\n")
        assert len(final_lines) == 2


def test_cma_search_end_to_end(models):
    transformer, lstm = models
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        def obj(dist):
            return fitness(dist, transformer, lstm, n=16, batch_size=8,
                           device="cpu", rng=rng)
        encoder = PiecewiseEncoder(T=64, K=2)
        results = cma_search(encoder, obj, tmp,
                             budget=8, pop_size=4, sigma_init=0.3,
                             num_restarts=1, seed=0)
        assert len(results) >= 8
        assert all(r.is_valid for r in results)


def test_planted_grid_end_to_end(models):
    transformer, lstm = models
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        def obj(dist):
            return fitness(dist, transformer, lstm, n=16, batch_size=8,
                           device="cpu", rng=rng)
        def factory(params):
            return Planted(T=64, template="decoy", params=params)
        grid = {"k_early": [1, 3], "k_true": [20, 28], "b_decoy": [0]}
        results = grid_search(factory, grid, obj, tmp, log_every=100)
        assert len(results) == 4
        assert all(r.is_valid for r in results)
