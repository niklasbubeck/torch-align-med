"""Helpers for spatial grids and coordinate handling.

Spatial-structure metrics need to know where each token sits on a grid so
they can compute Manhattan distances. We accept an explicit ``coords``
tensor of shape ``[N, S]`` (S = number of spatial dimensions) so that 2D
images, 3D volumes, and irregular layouts are all supported by the same
metric code.
"""
from __future__ import annotations

import torch


def make_grid_coords(shape: tuple[int, ...]) -> torch.Tensor:
    """Build integer coordinates for a regular S-dim grid.

    >>> make_grid_coords((2, 3)).tolist()
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    grids = torch.meshgrid(
        *[torch.arange(s) for s in shape], indexing="ij"
    )
    coords = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    return coords.long()


def manhattan_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Pairwise L1 distance, shape ``[N, N]``. ``coords`` is ``[N, S]``."""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    return diff.abs().sum(dim=-1)


def cosine_self_similarity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise cosine similarity. ``x`` is ``[..., N, D]``; result ``[..., N, N]``."""
    x_norm = x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))
    return x_norm @ x_norm.transpose(-1, -2)
