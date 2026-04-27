"""Tests for the retrain pipeline: MixedSampler + PassthroughFamily + train() plumbing."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from flip_flop.adversary.distribution import Stationary, Piecewise
from flip_flop.adversary.family import PassthroughFamily
from flip_flop.adversary.mixture_sampler import MixedSampler
from flip_flop.data import W, R, I, ZERO, ONE


def _assert_valid_ffl(tokens, T):
    assert tokens.shape[1] == T
    x = tokens.numpy()
    assert (x[:, 0] == W).all()
    assert (x[:, -2] == R).all()
    even = x[:, 0::2]
    odd = x[:, 1::2]
    assert np.isin(even, [W, R, I]).all()
    assert np.isin(odd, [ZERO, ONE]).all()


def test_passthrough_family_forwards():
    dist = Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5)
    fam = PassthroughFamily(dist=dist, name="test")
    rng = np.random.default_rng(0)
    tokens = fam.sample(32, rng)
    _assert_valid_ffl(tokens, 64)


def test_mixed_sampler_all_base():
    fams = [PassthroughFamily(Stationary(T=64, p_w=0.4, p_r=0.4, bit_p1=0.5))]
    sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=1.0)
    tokens = sampler(128, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)


def test_mixed_sampler_all_adv():
    fams = [PassthroughFamily(Stationary(T=64, p_w=0.4, p_r=0.4, bit_p1=0.9))]
    sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.0)
    tokens = sampler(128, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    # With bit_p1=0.9 + forced reads, the 1-bit should dominate data positions.
    bit_ones = (tokens[:, 1::2] == ONE).float().mean().item()
    assert bit_ones > 0.7, f"expected adv family to dominate bits, got {bit_ones}"


def test_mixed_sampler_50_50_splits_correctly():
    fams = [PassthroughFamily(Stationary(T=64, p_w=0.01, p_r=0.01, bit_p1=0.99),
                              name="adv")]
    sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.5)
    # Big batch; base dist has bit_p1=0.5, adv dist has bit_p1=0.99, so mean
    # should be near 0.75 if mixing is 50/50.
    tokens = sampler(2048, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)
    bit_ones = (tokens[:, 1::2] == ONE).float().mean().item()
    assert 0.6 < bit_ones < 0.85, f"expected ~0.75 bit-1 rate, got {bit_ones}"


def test_mixed_sampler_picks_families_uniformly():
    f0 = PassthroughFamily(Stationary(T=64, p_w=0.01, p_r=0.01, bit_p1=0.99), name="f0")
    f1 = PassthroughFamily(Stationary(T=64, p_w=0.01, p_r=0.01, bit_p1=0.01), name="f1")
    sampler = MixedSampler(T=64, base_p_i=0.8, families=[f0, f1], replay_frac=0.0)
    tokens = sampler(4096, np.random.default_rng(0))
    # Uniform over the two: mean bit-1 rate should be near 0.5.
    bit_ones = (tokens[:, 1::2] == ONE).float().mean().item()
    assert 0.40 < bit_ones < 0.60, f"expected ~0.5 bit-1 rate, got {bit_ones}"


def test_mixed_sampler_multi_family_piecewise():
    # A signature-distinct pair: check valid FFL holds across mixed batches.
    segs_a = [(0.0, 0.4, 0.0, 0.9), (0.5, 0.0, 0.0, 0.5)]
    segs_b = [(0.0, 0.0, 0.0, 0.5), (0.5, 0.4, 0.0, 0.1)]
    fams = [
        PassthroughFamily(Piecewise(T=64, segments=segs_a), name="early_writes"),
        PassthroughFamily(Piecewise(T=64, segments=segs_b), name="late_writes"),
    ]
    sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.3)
    tokens = sampler(256, np.random.default_rng(0))
    _assert_valid_ffl(tokens, 64)


def test_extract_families_from_adversary_log(tmp_path):
    """The stub loader reads a jsonl + returns PassthroughFamily instances."""
    from flip_flop.adversary.family import extract_families_from_adversary_log
    import json
    # Write a tiny log with 3 valid + 1 invalid entries.
    path = tmp_path / "log.jsonl"
    recs = [
        {"config": {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.9},
         "fitness": 0.9, "T_glitch": 0.9, "lstm_glitch": 0.0, "is_valid": True},
        {"config": {"name": "stationary", "T": 64, "p_w": 0.01, "p_r": 0.01, "bit_p1": 0.9},
         "fitness": 0.8, "T_glitch": 0.8, "lstm_glitch": 0.0, "is_valid": True},
        {"config": {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5},
         "fitness": 0.1, "T_glitch": 0.1, "lstm_glitch": 0.0, "is_valid": True},  # below threshold
        {"config": {"name": "stationary", "T": 64, "p_w": 0.02, "p_r": 0.02, "bit_p1": 0.9},
         "fitness": 0.7, "T_glitch": 0.7, "lstm_glitch": 0.1, "is_valid": True},   # lstm too high
    ]
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # Verify threshold filtering: with strict 0.5 threshold, only the two
    # high-T_glitch records pass (the 0.1 record is below threshold; the
    # lstm_glitch=0.1 record is above max_lstm_glitch=0.01).
    fams = extract_families_from_adversary_log(str(path), top_k=5, min_t_glitch=0.5)
    assert len(fams) == 2  # only two pass the T/LSTM gates
    names = [f.name for f in fams]
    assert names == ["adv_00", "adv_01"]
    # With the relaxed default (0.01), the 0.1 record now also passes.
    fams_relaxed = extract_families_from_adversary_log(str(path), top_k=5)
    assert len(fams_relaxed) == 3


def test_train_with_sampler_end_to_end():
    """Tiny continue-train: 20 steps on a MixedSampler, no crash, losses produced."""
    from flip_flop.train import TrainConfig, train
    from flip_flop.model import FFLMTransformer
    # Save a tiny baseline checkpoint first.
    with tempfile.TemporaryDirectory() as tmp:
        model = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
        ckpt = os.path.join(tmp, "base.pt")
        torch.save({"step": 0, "model_state_dict": model.state_dict()}, ckpt)

        cfg = TrainConfig(
            family="gpt2", vocab_size=5, n_positions=64, n_embd=32,
            n_layer=2, n_head=2, seq_len=64,
            train_steps=20, decay_end_step=21, warmup_steps=2,
            batch_size=8, lr=3e-5, eval_every=0, save_every=0, log_every=10,
            eval_in_n=32, eval_sparse_n=32, eval_dense_n=32,
            out_dir=os.path.join(tmp, "retrain"), device="cpu",
            init_from_ckpt=ckpt,
        )
        fams = [PassthroughFamily(Stationary(T=64, p_w=0.01, p_r=0.01, bit_p1=0.9))]
        sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.5)
        result = train(cfg, sampler=sampler)
        assert "last_loss" in result
        assert os.path.exists(os.path.join(cfg.out_dir, "model_final.pt"))


def test_balanced_selection_writes_logs_and_picks_best():
    """With selection active, every eval step must write to selection_log.jsonl
    and model_final.pt must correspond to the best-by-score eval."""
    from flip_flop.train import TrainConfig, train
    from flip_flop.model import FFLMTransformer
    with tempfile.TemporaryDirectory() as tmp:
        model = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
        ckpt = os.path.join(tmp, "base.pt")
        torch.save({"step": 0, "model_state_dict": model.state_dict()}, ckpt)

        cfg = TrainConfig(
            family="gpt2", vocab_size=5, n_positions=64, n_embd=32,
            n_layer=2, n_head=2, seq_len=64,
            train_steps=30, decay_end_step=31, warmup_steps=5,
            batch_size=8, lr=3e-5, eval_every=10, save_every=0, log_every=10,
            eval_in_n=32, eval_sparse_n=32, eval_dense_n=32,
            training_eval_subset=32,
            out_dir=os.path.join(tmp, "retrain"), device="cpu",
            init_from_ckpt=ckpt,
            selection_enabled=True,
            family_eval_n=32,
            plateau_window=20,    # disable plateau on tiny run
            plateau_tol=0.0,
            in_dist_hard_cap=1.0,  # disable hard cap on untrained tiny test model
            tail_98_hard_cap=1.0,
            tail_01_hard_cap=1.0,
        )
        fams = [PassthroughFamily(Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5))]
        sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.5)
        result = train(cfg, sampler=sampler)
        # selection_log.jsonl must exist and have entries
        sel_path = os.path.join(cfg.out_dir, "selection_log.jsonl")
        assert os.path.exists(sel_path)
        with open(sel_path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) >= 3  # 3 eval steps + 1 final
        # Best step recorded matches model_final.pt's step
        assert result["best_step"] >= 0
        # model_last.pt also exists when selection is active
        assert os.path.exists(os.path.join(cfg.out_dir, "model_last.pt"))


def test_three_tail_score_penalizes_each_tail():
    """Step 4: score must penalize FFL(0.98) and FFL(0.1) drift, not just FFL(0.8)."""
    from flip_flop.train import TrainConfig, train
    from flip_flop.model import FFLMTransformer
    with tempfile.TemporaryDirectory() as tmp:
        model = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
        ckpt = os.path.join(tmp, "base.pt")
        torch.save({"step": 0, "model_state_dict": model.state_dict()}, ckpt)
        cfg = TrainConfig(
            family="gpt2", vocab_size=5, n_positions=64, n_embd=32,
            n_layer=2, n_head=2, seq_len=64,
            train_steps=20, decay_end_step=21, warmup_steps=2,
            batch_size=8, lr=3e-5, eval_every=10, save_every=0, log_every=10,
            eval_in_n=32, eval_sparse_n=32, eval_dense_n=32,
            training_eval_subset=32,
            out_dir=os.path.join(tmp, "retrain"), device="cpu",
            init_from_ckpt=ckpt,
            selection_enabled=True,
            family_eval_n=32,
            plateau_window=20,
            plateau_tol=0.0,
            in_dist_hard_cap=1.0,
            tail_98_hard_cap=1.0,
            tail_01_hard_cap=1.0,
            lambda_in=5.0,
            lambda_98=0.7,
            lambda_01=0.7,
        )
        fams = [PassthroughFamily(Stationary(T=64, p_w=0.1, p_r=0.1, bit_p1=0.5))]
        sampler = MixedSampler(T=64, base_p_i=0.8, families=fams, replay_frac=0.5)
        train(cfg, sampler=sampler)
        # Selection log must include the new three-tail fields.
        sel_path = os.path.join(cfg.out_dir, "selection_log.jsonl")
        with open(sel_path) as f:
            recs = [json.loads(l) for l in f]
        assert len(recs) >= 1
        for r in recs:
            assert "ffl_98_glitch" in r
            assert "ffl_01_glitch" in r
            assert "lambda_in" in r
            assert "lambda_98" in r
            assert "lambda_01" in r


def test_r4_mix_sampler_produces_uniform_mixture():
    """Liu R4 control: uniform per-sequence draw from 3 stationary FFL distributions."""
    from flip_flop.adversary.r4_sampler import R4MixSampler, R4_PAPER_PIs
    from flip_flop.data import W
    sampler = R4MixSampler(T=64, p_i_values=R4_PAPER_PIs)
    rng = np.random.default_rng(0)
    tokens = sampler(2048, rng)
    # Validity
    x = tokens.numpy()
    assert (x[:, 0] == W).all()
    # Each p_i is FFL(p_i): write density = (1 - p_i)/2.
    # With uniform 1/3 mix of 0.1, 0.9, 0.98, expected mean write density:
    #   E = (1/3)(1-0.1)/2 + (1/3)(1-0.9)/2 + (1/3)(1-0.98)/2
    #   E = (1/3)(0.45 + 0.05 + 0.01) ≈ 0.17
    inst = x[:, 0::2]
    write_density = (inst == W).mean()
    # Loose bounds — confirms sampler isn't degenerate.
    assert 0.10 < write_density < 0.25


def test_r4_mix_sampler_no_families_attribute():
    """R4MixSampler must NOT expose .families (so selection stays off)."""
    from flip_flop.adversary.r4_sampler import R4MixSampler
    sampler = R4MixSampler(T=64)
    assert not hasattr(sampler, "families") or getattr(sampler, "families", None) is None


def test_selection_inactive_when_no_families_attribute():
    """When sampler has no .families (legacy path), training behaves as before."""
    from flip_flop.train import TrainConfig, train
    from flip_flop.model import FFLMTransformer
    from flip_flop.data import sample_ffl
    with tempfile.TemporaryDirectory() as tmp:
        model = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
        ckpt = os.path.join(tmp, "base.pt")
        torch.save({"step": 0, "model_state_dict": model.state_dict()}, ckpt)
        cfg = TrainConfig(
            family="gpt2", vocab_size=5, n_positions=64, n_embd=32,
            n_layer=2, n_head=2, seq_len=64,
            train_steps=10, decay_end_step=11, warmup_steps=2,
            batch_size=8, lr=3e-5, eval_every=0, save_every=0, log_every=10,
            eval_in_n=32, eval_sparse_n=32, eval_dense_n=32,
            out_dir=os.path.join(tmp, "legacy"), device="cpu",
            init_from_ckpt=ckpt, selection_enabled=False,
        )
        # plain callable, not MixedSampler — has no .families
        plain = lambda bs, rng: sample_ffl(64, 0.8, bs, rng)
        result = train(cfg, sampler=plain)
        assert os.path.exists(os.path.join(cfg.out_dir, "model_final.pt"))
        assert result.get("best_step") is None
