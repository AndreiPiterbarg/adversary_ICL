"""Expressive adversary — feature-conditioned FFL controller.

Self-contained package implementing `docs/PLAN_expressive_adversary.md`. It does
NOT import from `flip_flop/` so it cannot interfere with experiments running
against that tree (e.g. Rung 1). The relevant data/distribution/encoder/CMA
code is copied verbatim (faithful to the originals) so this folder is a
standalone, testable artifact.

Public surface:
  data                — FFL token primitives (validity is centralized here)
  distribution        — FFLDistribution ABC, reference families, FeatureControlledFFL
  encoder             — Encoder protocol, DiagonalCMAES, FeatureControlledEncoder
"""
from __future__ import annotations

__all__ = ["data", "distribution", "encoder"]
