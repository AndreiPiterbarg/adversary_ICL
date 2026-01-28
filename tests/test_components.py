"""Unit tests for core model components."""

import torch
import pytest
from curriculum_model.roles import Role, NUM_ROLES, RoleEmbedding, compose_token
from curriculum_model.embedders import VectorEmbedder, MatrixEmbedder, ScalarEmbedder


class TestRoles:
    def test_role_count(self):
        """Should have exactly 6 roles."""
        assert NUM_ROLES == 6

    def test_role_values_unique(self):
        """All role values should be unique."""
        values = [r.value for r in Role]
        assert len(values) == len(set(values))

    def test_role_names(self):
        """Expected roles exist."""
        expected = {"MATRIX", "VEC_PRIMARY", "VEC_SECONDARY", "VEC_BIAS", "SCALAR", "OUTPUT"}
        actual = {r.name for r in Role}
        assert actual == expected


class TestRoleEmbedding:
    def test_output_shape(self, role_embedding, device):
        """Role embedding outputs correct shape."""
        idx = torch.tensor(Role.MATRIX.value, device=device)
        emb = role_embedding(idx)
        assert emb.shape == (role_embedding.n_embd,)

    def test_batch_output_shape(self, role_embedding, device):
        """Batch role embedding outputs correct shape."""
        indices = torch.tensor([r.value for r in Role], device=device)
        emb = role_embedding(indices)
        assert emb.shape == (NUM_ROLES, role_embedding.n_embd)

    def test_convenience_methods(self, role_embedding):
        """All convenience methods work."""
        for method in ["get_matrix_role", "get_primary_role", "get_secondary_role",
                       "get_bias_role", "get_scalar_role", "get_output_role"]:
            emb = getattr(role_embedding, method)()
            assert emb.shape == (role_embedding.n_embd,)


class TestVectorEmbedder:
    def test_output_shape(self, device):
        """VectorEmbedder: (B, d) -> (B, n_embd)."""
        d, n_embd, batch = 4, 64, 8
        embedder = VectorEmbedder(d, n_embd).to(device)
        x = torch.randn(batch, d, device=device)
        out = embedder(x)
        assert out.shape == (batch, n_embd)

    def test_single_vector(self, device):
        """Works with single vector."""
        d, n_embd = 4, 64
        embedder = VectorEmbedder(d, n_embd).to(device)
        x = torch.randn(d, device=device)
        out = embedder(x)
        assert out.shape == (n_embd,)


class TestMatrixEmbedder:
    def test_output_shape(self, device):
        """MatrixEmbedder: (B, d, d) -> (B, n_embd)."""
        d, n_embd, batch = 4, 64, 8
        embedder = MatrixEmbedder(d, n_embd).to(device)
        x = torch.randn(batch, d, d, device=device)
        out = embedder(x)
        assert out.shape == (batch, n_embd)


class TestScalarEmbedder:
    def test_output_shape(self, device):
        """ScalarEmbedder: (B,) -> (B, n_embd)."""
        n_embd, batch = 64, 8
        embedder = ScalarEmbedder(n_embd).to(device)
        x = torch.randn(batch, device=device)
        out = embedder(x)
        assert out.shape == (batch, n_embd)


class TestComposeToken:
    def test_additive_composition(self, embedders, role_embedding, device):
        """compose_token = component + role (additive)."""
        d = embedders.d
        vec = torch.randn(d, device=device)
        comp_emb = embedders.vector(vec)
        role_emb = role_embedding.get_output_role()

        composed = compose_token(comp_emb, role_emb)
        expected = comp_emb + role_emb

        assert torch.allclose(composed, expected)

    def test_shape_preservation(self, embedders, role_embedding, device):
        """Composed token has same shape as component embedding."""
        batch, d = 8, embedders.d
        vecs = torch.randn(batch, d, device=device)
        comp_emb = embedders.vector(vecs)
        role_emb = role_embedding.get_output_role()

        composed = compose_token(comp_emb, role_emb)
        assert composed.shape == comp_emb.shape


class TestModelForwardPass:
    def test_forward_output_shape(self, small_model, small_config, device):
        """Model forward pass produces correct output shape."""
        batch, seq_len = 4, 20
        n_embd = small_config.n_embd

        tokens = torch.randn(batch, seq_len, n_embd, device=device)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        output = small_model(tokens, ex_pos, mask_pos)

        assert output.vector_output.shape == (batch, small_config.d)
        assert output.scalar_output.shape == (batch, 1)

    def test_gradient_flow(self, small_model, small_config, device):
        """Gradients flow through the model."""
        batch, seq_len = 4, 20
        n_embd = small_config.n_embd

        tokens = torch.randn(batch, seq_len, n_embd, device=device, requires_grad=True)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        output = small_model(tokens, ex_pos, mask_pos)
        loss = output.vector_output.sum()
        loss.backward()

        assert tokens.grad is not None
        assert tokens.grad.abs().sum() > 0
