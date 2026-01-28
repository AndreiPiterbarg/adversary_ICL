# Self-Refine ICL: Iterative Self-Refinement in Transformer In-Context Learning

This repository contains the code for experiments on iterative self-refinement in transformers trained for in-context learning (ICL) on linear systems.

## Problem Setting

We study whether transformers trained for ICL on linear systems can iteratively refine predictions at inference time. Given a symmetric positive-definite matrix A and K context examples {(b_i, x_i*)} where x_i* = A^{-1}b_i, the model predicts x* for a query b_query.

## Key Finding

**Baseline Failure**: Naively feeding predictions back as additional context causes catastrophic degradation (~1600x MSE increase).

**Solution - Residual Prediction**: Training with a dual objective enables iterative refinement:
- Direct prediction loss: L_direct = ||f(C, b_query) - x*||^2
- Residual prediction loss: L_residual = ||f(C, b_query, x_tilde) - (x* - x_tilde)||^2

Where x_tilde is a noisy estimate and the model learns to predict corrections.


## Project Structure

```
Self_Refine_ICL/
├── scripts/
│   ├── approach_c_residual_prediction.py  # Main residual prediction experiment
│   └── run_approaches_b_c.py              # Full comparison suite
├── src/
│   ├── curriculum_model/                  # Model components
│   │   ├── component_model.py             # Main transformer model
│   │   ├── embedders.py                   # Vector/matrix/scalar embedders
│   │   ├── roles.py                       # Role embeddings (key for refinement)
│   │   ├── special_tokens.py              # SEP and MASK tokens
│   │   ├── sequence_builder.py            # Token sequence construction
│   │   ├── output_heads.py                # Prediction heads
│   │   └── tasks.py                       # Task specifications
│   └── custom_transformer/                # Transformer backbone
│       ├── transformer.py                 # GPT-style backbone
│       ├── block.py                       # Transformer block
│       ├── attention.py                   # Multi-head attention
│       └── ...
└── experiment_results/
    └── results.json                       # Pre-computed results
```

## Method Details

### Architecture Modification
The current estimate x_tilde is encoded with a distinct role embedding (VEC_SECONDARY) separate from ground-truth solutions (OUTPUT). The query sequence becomes:

```
[SEP, A, SEP, b_1, x_1*, ..., SEP, b_query, x_tilde, MASK]
```

The model outputs the residual at MASK position. Inference proceeds as:
```
x_0 = f(C, b_query)              # Initial prediction
x_{k+1} = x_k + f(C, b_query, x_k)  # Refinement iterations
```

### Key Insights
1. **Explicit residual prediction** rather than direct solution prediction
2. **Role-based disambiguation** of estimates from ground truth
3. **Fixed context structure** across iterations
