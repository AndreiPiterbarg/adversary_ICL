"""CPU-only proof that the controller is wired into the REAL flip_flop pipeline.

Exercises the entire integration path WITHOUT any model or GPU:

  search side   : config yaml -> AdversaryConfig -> run._make_encoder
                  -> FeatureControlledEncoder (n_dims 24) -> decode -> valid FFL
  downstream    : flip_flop's own FFLDistribution.from_dict reconstructs the
                  controller through flip_flop's REGISTRY  (this is exactly the
                  reconstruction pool_breaking_points / dump_dataset /
                  run_retrain_from_dataset rely on — so wiring it proves the
                  whole pool->dump->retrain->eval chain accepts the family).

Importing flip_flop.adversary.run pulls torch/model/io but loads NO checkpoint
and never touches CUDA — a cheap CPU import, safe alongside the running Rung 1.
"""
import numpy as np
import pytest

# flip_flop's OWN modules (not the expressive_adversary copies) — this is the
# real pipeline surface.
ff_dist = pytest.importorskip("flip_flop.adversary.distribution")
ff_run = pytest.importorskip("flip_flop.adversary.run")
from flip_flop.adversary.search import FeatureControlledEncoder as FFEnc  # noqa: E402

from expressive_adversary.distribution import FeatureControlledFFL  # noqa: E402

CONFIG = "flip_flop/configs/quick_adversary_feature_controlled.yaml"


def test_registered_in_flipflop_registry():
    assert "feature_controlled" in ff_dist.REGISTRY
    assert ff_dist.REGISTRY["feature_controlled"] is FeatureControlledFFL
    assert FFEnc is not None, "FeatureControlledEncoder must import in pipeline"


def test_config_parses_with_regret_and_gates():
    cfg = ff_run.AdversaryConfig.from_yaml(CONFIG)
    assert cfg.strategy == "cma"
    assert cfg.dist_name == "feature_controlled"
    assert cfg.T == 512
    assert cfg.objective_mode == "regret"
    assert cfg.min_reads_per_seq == 3.0 and cfg.min_gap == 4.0


def test_make_encoder_returns_feature_controlled_encoder():
    cfg = ff_run.AdversaryConfig.from_yaml(CONFIG)
    enc = ff_run._make_encoder(cfg)
    assert isinstance(enc, FFEnc)
    assert enc.n_dims == 24
    dist = enc.decode(enc.random_init(np.random.default_rng(0)))
    assert isinstance(dist, FeatureControlledFFL)


def test_flipflop_from_dict_roundtrip_through_registry():
    """The exact reconstruction pool/dump/retrain perform on a logged config."""
    cfg = ff_run.AdversaryConfig.from_yaml(CONFIG)
    enc = ff_run._make_encoder(cfg)
    dist = enc.decode(enc.random_init(np.random.default_rng(1)))
    cfg_dict = dist.to_dict()                       # what gets logged
    rebuilt = ff_dist.FFLDistribution.from_dict(cfg_dict)   # flip_flop's classmethod
    assert rebuilt.to_dict() == cfg_dict
    # reconstructed sampler still produces valid FFL (small CPU batch)
    tokens = rebuilt.sample(4, np.random.default_rng(2))
    x = tokens.numpy()
    assert x.shape == (4, cfg.T)
    assert (x[:, 0] == 0).all()       # W
    assert (x[:, -2] == 1).all()      # R (penultimate token)
    even, odd = x[:, 0::2], x[:, 1::2]
    assert np.isin(even, [0, 1, 2]).all() and np.isin(odd, [3, 4]).all()
    # read-determinism holds on the reconstructed stream
    for b in range(4):
        last = None
        for k in range(even.shape[1]):
            if even[b, k] == 0:
                last = odd[b, k] - 3
            elif even[b, k] == 1:
                assert last is not None and (odd[b, k] - 3) == last
