# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing iterative self-refinement for transformers trained on in-context learning (ICL) with linear systems. The model learns to predict residuals/corrections rather than direct solutions, enabling iterative refinement at inference.

## Commands

```bash
# Run main experiment (Approach C - Residual Prediction)
python scripts/approach_c_residual_prediction.py --device cuda

# Run full comparison suite (Baseline, Approach B, Approach C)
python scripts/run_approaches_b_c.py --device cuda

# Run tests
python -m pytest tests/ -v

# Install dependencies
pip install torch numpy scipy
```

## Architecture

### Module Structure

**`src/curriculum_model/`** - Model components:
- `component_model.py`: Main `ComponentTransformerModel` class
- `roles.py`: 6 semantic roles (MATRIX, VEC_PRIMARY, VEC_SECONDARY, VEC_BIAS, SCALAR, OUTPUT)
- `embedders.py`: Vector/Matrix/Scalar embedders
- `special_tokens.py`: SEP and MASK tokens
- `sequence_builder.py`: PositionalEncoder (example-level positional encoding)
- `output_heads.py`: Dual output head (vector + scalar)

**`src/custom_transformer/`** - GPT-style transformer:
- `transformer.py`: CustomGPTBackbone
- `block.py`: TransformerBlock (pre-norm architecture)
- `attention.py`: Multi-head causal attention
- `ffn.py`: Feed-forward network

### Token Composition
```
token = embed_component(component) + embed_role(role)
```

### Key Design: Role-Based Disambiguation
Current estimate uses VEC_SECONDARY role; ground-truth solutions use OUTPUT role:
```
[SEP, A, SEP, b_1, x_1*, ..., SEP, b_query, x_tilde, MASK]
                                           ^^^^^^^^
                                    VEC_SECONDARY role
```

### Refinement Algorithm
```
x_0 = f(context, query)                 # Initial prediction
x_{k+1} = x_k + f(context, query, x_k)  # Refinement iterations
```

## Key Configuration (scripts/approach_c_residual_prediction.py)

```python
d = 4                    # Vector/matrix dimension
n_embd = 128            # Transformer hidden dimension
n_layer = 6, n_head = 4 # Transformer architecture
training_steps = 50000
residual_weight = 0.5   # Mix of direct/residual loss
num_context = 5         # Context examples per sample
```

## Data Generation

SPD matrices sampled with controlled condition numbers (eigenvalues on log scale). Training uses dual loss:
- L_direct = ||f(C, b_query) - x*||²
- L_residual = ||f(C, b_query, x_tilde) - (x* - x_tilde)||²
