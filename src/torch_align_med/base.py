"""Base classes for torch-align-med metrics.

Two metric families:

- :class:`PairwiseMetric` compares two feature spaces and returns a scalar.
  Inputs are token tensors of shape ``[B, N, D]`` or pooled tensors ``[B, D]``.
- :class:`EncoderDiagnostic` characterises a single feature space. Some
  diagnostics depend on a spatial grid (passed via ``coords: [N, S]``);
  others do not.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class PairwiseMetric(ABC):
    name: str = ""
    higher_is_better: bool = True

    @abstractmethod
    def compute(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        **kwargs,
    ) -> float: ...

    def __call__(self, feats_a: torch.Tensor, feats_b: torch.Tensor, **kwargs) -> float:
        return self.compute(feats_a, feats_b, **kwargs)


class EncoderDiagnostic(ABC):
    name: str = ""
    requires_coords: bool = False
    higher_is_better: bool = True

    @abstractmethod
    def compute(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None = None,
        **kwargs,
    ) -> float: ...

    def __call__(
        self, feats: torch.Tensor, coords: torch.Tensor | None = None, **kwargs
    ) -> float:
        return self.compute(feats, coords=coords, **kwargs)
