"""Metric implementations.

Pairwise metrics compare two feature spaces. Encoder diagnostics describe
a single feature space (the runner applies them on each side of a pair).
"""
from torch_align_med.metrics.diagnostic.cds import CorrelationDecaySlope
from torch_align_med.metrics.diagnostic.lds import LocalDistantSimilarity
from torch_align_med.metrics.diagnostic.rmsc import RMSSpatialContrast
from torch_align_med.metrics.pairwise.cka import LinearCKA, linear_cka
from torch_align_med.metrics.pairwise.cosine import PatchCosineAlignment
from torch_align_med.metrics.pairwise.mdms import MarginalDMS

__all__ = [
    "CorrelationDecaySlope",
    "LinearCKA",
    "LocalDistantSimilarity",
    "MarginalDMS",
    "PatchCosineAlignment",
    "RMSSpatialContrast",
    "linear_cka",
]
