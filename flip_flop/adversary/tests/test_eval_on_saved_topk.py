"""Tests for the standalone eval_on_saved_topk module."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from flip_flop.adversary.distribution import Stationary
from flip_flop.model import FFLMTransformer, FFLMLSTM
from flip_flop.scripts.eval_on_saved_topk import _eval_topk_file


@pytest.fixture(scope="module")
def models_cpu():
    t = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    l = FFLMLSTM(hidden_size=32)
    t.eval(); l.eval()
    return t, l


def _write_topk(path, configs, fitness_values=None):
    with open(path, "w") as f:
        for i, c in enumerate(configs):
            r = {
                "config": c,
                "T_glitch": fitness_values[i] if fitness_values else 0.5,
                "fitness": fitness_values[i] if fitness_values else 0.5,
                "is_valid": True,
            }
            f.write(json.dumps(r) + "\n")


def test_eval_topk_file_returns_per_config_metrics(models_cpu):
    t, l = models_cpu
    cfgs = [
        {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.9},
        {"name": "stationary", "T": 64, "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.9},
        {"name": "stationary", "T": 64, "p_w": 0.05, "p_r": 0.05, "bit_p1": 0.5},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "top_k.jsonl")
        _write_topk(path, cfgs)
        rng = np.random.default_rng(0)
        out = _eval_topk_file(t, l, path, n=128, batch_size=64,
                              device="cpu", rng=rng)
        assert out["n_configs"] == 3
        assert out["n_per_config"] == 128
        assert 0.0 <= out["T_glitch_mean"] <= 1.0
        assert 0.0 <= out["T_glitch_max"] <= 1.0
        assert out["T_glitch_min"] <= out["T_glitch_max"]
        assert len(out["per_config"]) == 3
        for c in out["per_config"]:
            assert "T_glitch" in c
            assert "lstm_glitch" in c
            assert "rank" in c


def test_eval_topk_file_handles_no_lstm(models_cpu):
    t, _ = models_cpu
    cfgs = [{"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "top_k.jsonl")
        _write_topk(path, cfgs)
        rng = np.random.default_rng(0)
        out = _eval_topk_file(t, None, path, n=64, batch_size=64,
                              device="cpu", rng=rng)
        assert out["per_config"][0]["lstm_glitch"] is None
        assert out["lstm_glitch_max"] == 0.0


def test_eval_topk_file_top5_mean(models_cpu):
    """top5_mean averages the 5 highest T_glitch values across configs."""
    t, l = models_cpu
    # 8 configs of varying difficulty
    cfgs = [
        {"name": "stationary", "T": 64,
         "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.9},  # likely hard
        {"name": "stationary", "T": 64,
         "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.9},
        {"name": "stationary", "T": 64,
         "p_w": 0.05, "p_r": 0.05, "bit_p1": 0.7},
        {"name": "stationary", "T": 64,
         "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5},
        {"name": "stationary", "T": 64,
         "p_w": 0.2, "p_r": 0.2, "bit_p1": 0.5},
        {"name": "stationary", "T": 64,
         "p_w": 0.3, "p_r": 0.3, "bit_p1": 0.5},
        {"name": "stationary", "T": 64,
         "p_w": 0.4, "p_r": 0.4, "bit_p1": 0.5},
        {"name": "stationary", "T": 64,
         "p_w": 0.45, "p_r": 0.45, "bit_p1": 0.5},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "top_k.jsonl")
        _write_topk(path, cfgs)
        rng = np.random.default_rng(0)
        out = _eval_topk_file(t, l, path, n=64, batch_size=64,
                              device="cpu", rng=rng)
        # top5_mean must be >= overall mean (average of largest values)
        assert out["T_glitch_top5_mean"] >= out["T_glitch_mean"] - 1e-9


def test_eval_topk_file_missing_config_raises(models_cpu):
    t, l = models_cpu
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"foo": "bar"}) + "\n")  # no "config" key
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="no 'config' field"):
            _eval_topk_file(t, l, path, n=8, batch_size=8,
                            device="cpu", rng=rng)
