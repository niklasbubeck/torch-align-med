"""torch-align-med — semantic-alignment metrics between latent spaces.

Public API:

>>> from torch_align_med import calculate_alignment_metrics
>>> from torch_align_med import make_grid_coords
>>> import torch
>>> a = torch.randn(2, 16, 64)
>>> b = torch.randn(2, 16, 64)
>>> coords = make_grid_coords((4, 4))
>>> calculate_alignment_metrics(a, b, coords=coords, linear_cka=True, lds=True)
{'linear_cka': ..., 'lds_input1': ..., 'lds_input2': ...}
"""
from torch_align_med._grids import make_grid_coords
from torch_align_med.api import calculate_alignment_metrics
from torch_align_med.base import EncoderDiagnostic, PairwiseMetric
from torch_align_med.metrics import (
    CorrelationDecaySlope,
    LinearCKA,
    LocalDistantSimilarity,
    MarginalDMS,
    PatchCosineAlignment,
    RMSSpatialContrast,
)

__version__ = "0.1.0"

__all__ = [
    "CorrelationDecaySlope",
    "EncoderDiagnostic",
    "LinearCKA",
    "LocalDistantSimilarity",
    "MarginalDMS",
    "PairwiseMetric",
    "PatchCosineAlignment",
    "RMSSpatialContrast",
    "calculate_alignment_metrics",
    "make_grid_coords",
]
