"""Shared fixtures for synthetic feature tensors."""
from __future__ import annotations

import pytest
import torch

from torch_align_med import make_grid_coords


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def random_tokens_2d() -> torch.Tensor:
    return torch.randn(4, 16, 32)  # B=4, N=16 (4x4 grid), D=32


@pytest.fixture
def coords_2d() -> torch.Tensor:
    return make_grid_coords((4, 4))


@pytest.fixture
def coords_3d() -> torch.Tensor:
    return make_grid_coords((3, 3, 3))


@pytest.fixture
def random_tokens_3d() -> torch.Tensor:
    return torch.randn(2, 27, 24)  # B=2, N=27 (3^3 grid)


@pytest.fixture
def smooth_tokens_2d(coords_2d) -> torch.Tensor:
    """Tokens whose value is a smooth function of position — strong local structure."""
    pos = coords_2d.float()
    n, _ = pos.shape
    # 32-dim features that vary smoothly with position.
    proj = torch.randn(2, 32)
    feats = torch.stack([torch.sin(pos @ proj), torch.cos(pos @ proj)], dim=-1).reshape(n, -1)
    return feats.unsqueeze(0).repeat(4, 1, 1)
