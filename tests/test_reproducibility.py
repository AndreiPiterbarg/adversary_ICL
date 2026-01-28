"""Reproducibility tests to ensure deterministic results with seeding."""

import torch
import pytest
import numpy as np
from pathlib import Path
import sys

_src_dir = Path(__file__).parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from curriculum_model.component_model import ComponentTransformerModel, ComponentModelConfig
from conftest import sample_spd


class TestDeterminism:
    def test_seeded_spd_sampling(self, device):
        """SPD matrix sampling is deterministic with same seed."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            np.random.seed(42)
            A = sample_spd(10, d=4, kappa_min=1.0, kappa_max=100.0, device=device)
            results.append(A.cpu())

        assert torch.allclose(results[0], results[1]), "SPD sampling not deterministic"

    def test_seeded_model_init(self, device):
        """Model initialization is deterministic with same seed."""
        config = ComponentModelConfig(d=4, n_embd=64, n_layer=2, n_head=4)

        weights = []
        for _ in range(2):
            torch.manual_seed(42)
            model = ComponentTransformerModel(config).to(device)
            weights.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        for key in weights[0]:
            assert torch.allclose(weights[0][key], weights[1][key]), f"Weight {key} not deterministic"

    def test_seeded_forward_pass(self, device):
        """Forward pass is deterministic with same seed."""
        config = ComponentModelConfig(d=4, n_embd=64, n_layer=2, n_head=4, dropout=0.0)

        torch.manual_seed(42)
        model = ComponentTransformerModel(config).to(device)
        model.eval()

        batch, seq_len = 4, 20
        tokens = torch.randn(batch, seq_len, config.n_embd, device=device)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        outputs = []
        for _ in range(2):
            with torch.no_grad():
                out = model(tokens, ex_pos, mask_pos)
                outputs.append(out.vector_output.cpu())

        assert torch.allclose(outputs[0], outputs[1]), "Forward pass not deterministic"

    def test_different_seeds_differ(self, device):
        """Different seeds produce different results."""
        results = []
        for seed in [42, 123]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            A = sample_spd(10, d=4, kappa_min=1.0, kappa_max=100.0, device=device)
            results.append(A.cpu())

        assert not torch.allclose(results[0], results[1]), "Different seeds should give different results"


class TestCheckpointConsistency:
    def test_save_load_weights_identical(self, small_model, device, tmp_path):
        """Model weights are identical after save/load cycle."""
        checkpoint_path = tmp_path / "model.pt"

        # Save
        original_state = {k: v.cpu().clone() for k, v in small_model.state_dict().items()}
        torch.save(small_model.state_dict(), checkpoint_path)

        # Load into new model
        loaded_model = ComponentTransformerModel(small_model.config).to(device)
        loaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        loaded_state = {k: v.cpu().clone() for k, v in loaded_model.state_dict().items()}

        # Compare
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key]), f"Weight {key} differs after load"

    def test_inference_after_load(self, small_model, small_config, device, tmp_path):
        """Inference produces same results after loading."""
        checkpoint_path = tmp_path / "model.pt"
        small_model.eval()

        # Setup input
        batch, seq_len = 4, 20
        tokens = torch.randn(batch, seq_len, small_config.n_embd, device=device)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        # Get original output
        with torch.no_grad():
            original_output = small_model(tokens, ex_pos, mask_pos).vector_output.cpu()

        # Save and load
        torch.save(small_model.state_dict(), checkpoint_path)
        loaded_model = ComponentTransformerModel(small_config).to(device)
        loaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        loaded_model.eval()

        # Get loaded output
        with torch.no_grad():
            loaded_output = loaded_model(tokens, ex_pos, mask_pos).vector_output.cpu()

        assert torch.allclose(original_output, loaded_output, atol=1e-6), "Output differs after load"


class TestNumericalStability:
    def test_no_nan_in_forward(self, small_model, small_config, device):
        """Forward pass should not produce NaN values."""
        small_model.eval()
        batch, seq_len = 8, 30

        tokens = torch.randn(batch, seq_len, small_config.n_embd, device=device)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        with torch.no_grad():
            output = small_model(tokens, ex_pos, mask_pos)

        assert not torch.isnan(output.vector_output).any(), "NaN in vector output"
        assert not torch.isnan(output.scalar_output).any(), "NaN in scalar output"

    def test_gradient_not_exploding(self, small_model, small_config, device):
        """Gradients should not explode during training step."""
        batch, seq_len = 4, 20

        tokens = torch.randn(batch, seq_len, small_config.n_embd, device=device, requires_grad=True)
        ex_pos = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        mask_pos = torch.full((batch,), seq_len - 1, dtype=torch.long, device=device)

        output = small_model(tokens, ex_pos, mask_pos)
        loss = output.vector_output.sum()
        loss.backward()

        # Check gradient norms are reasonable
        for name, param in small_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e6, f"Gradient explosion in {name}: norm = {grad_norm}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
