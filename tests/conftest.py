"""Shared pytest fixtures for Self-Refine ICL tests."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
_src_dir = Path(__file__).parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import ComponentTransformerModel, ComponentModelConfig
from curriculum_model.embedders import ComponentEmbedders
from curriculum_model.roles import RoleEmbedding


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small model config for fast tests."""
    return ComponentModelConfig(
        d=4,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=64,
        max_examples=32,
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_config, device):
    """Small model for unit tests."""
    return ComponentTransformerModel(small_config).to(device)


@pytest.fixture
def embedders(small_config, device):
    """Component embedders."""
    return ComponentEmbedders(small_config.d, small_config.n_embd).to(device)


@pytest.fixture
def role_embedding(small_config, device):
    """Role embedding layer."""
    return RoleEmbedding(small_config.n_embd).to(device)


@pytest.fixture
def seeded_rng():
    """Fixture for reproducible random state."""
    def _seed(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _seed


def sample_spd(batch_size, d, kappa_min, kappa_max, device):
    """Sample SPD matrices with controlled condition numbers."""
    log_min, log_max = np.log(kappa_min), np.log(kappa_max)

    u = torch.rand(batch_size, device=device)
    kappas = torch.exp(torch.tensor(log_min, device=device) + u * (log_max - log_min))

    u_eigs = torch.rand(batch_size, d, device=device)
    eigs = torch.exp(u_eigs * kappas.unsqueeze(-1).log())

    G = torch.randn(batch_size, d, d, device=device)
    Q, _ = torch.linalg.qr(G)
    A = Q @ torch.diag_embed(eigs) @ Q.transpose(-2, -1)
    return 0.5 * (A + A.transpose(-2, -1))
