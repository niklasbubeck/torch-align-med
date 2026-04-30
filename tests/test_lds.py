import pytest
import torch

from torch_align_med import LocalDistantSimilarity


def test_smooth_tokens_have_positive_lds(smooth_tokens_2d, coords_2d):
    score = LocalDistantSimilarity().compute(smooth_tokens_2d, coords=coords_2d)
    assert score > 0.05


def test_random_tokens_lds_is_near_zero(random_tokens_2d, coords_2d):
    score = LocalDistantSimilarity().compute(random_tokens_2d, coords=coords_2d)
    assert abs(score) < 0.2


def test_3d_grid_works(random_tokens_3d, coords_3d):
    score = LocalDistantSimilarity().compute(random_tokens_3d, coords=coords_3d)
    assert isinstance(score, float)


def test_missing_coords_errors(random_tokens_2d):
    with pytest.raises(ValueError, match="requires `coords"):
        LocalDistantSimilarity().compute(random_tokens_2d)


def test_coord_count_mismatch_errors(random_tokens_2d):
    with pytest.raises(ValueError, match="coords has N="):
        LocalDistantSimilarity().compute(random_tokens_2d, coords=torch.zeros(8, 2))
