import torch

from torch_align_med import CorrelationDecaySlope


def test_smooth_tokens_have_positive_cds(smooth_tokens_2d, coords_2d):
    score = CorrelationDecaySlope().compute(smooth_tokens_2d, coords=coords_2d)
    assert score > 0.0


def test_random_tokens_cds_near_zero(random_tokens_2d, coords_2d):
    score = CorrelationDecaySlope().compute(random_tokens_2d, coords=coords_2d)
    assert abs(score) < 0.2


def test_works_on_3d_grid(random_tokens_3d, coords_3d):
    score = CorrelationDecaySlope().compute(random_tokens_3d, coords=coords_3d)
    assert isinstance(score, float)
