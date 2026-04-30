"""Common interface for feature-extraction wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseEncoder(ABC):
    """Wraps a frozen vision encoder.

    Subclasses expose patch tokens via :meth:`extract_tokens` and pooled
    embeddings via :meth:`extract_pooled`. The grid layout is reported by
    :meth:`grid_shape` so callers can build a matching ``coords`` tensor.
    """

    repo_id: str = ""
    license: str = ""

    @abstractmethod
    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens of shape ``[B, N, D]``."""

    @abstractmethod
    def extract_pooled(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled embeddings of shape ``[B, D]``."""

    @abstractmethod
    def grid_shape(self) -> tuple[int, ...]:
        """Spatial layout of the patch tokens (e.g. ``(H, W)`` for 2D)."""
