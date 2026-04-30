import pytest
import torch

from torch_align_med import RMSSpatialContrast


def test_constant_features_are_zero():
    x = torch.ones(2, 16, 32)
    assert RMSSpatialContrast().compute(x) == pytest.approx(0.0, abs=1e-6)


def test_orthonormal_tokens_have_known_value():
    # 8 tokens that are 8 orthonormal basis vectors in R^8: each unit-norm,
    # mean is 1/8 * sum(e_i), squared distance to mean = 1 - 1/8 = 7/8.
    x = torch.eye(8).unsqueeze(0)  # [1, 8, 8]
    score = RMSSpatialContrast().compute(x)
    expected = (7.0 / 8.0) ** 0.5
    assert score == pytest.approx(expected, abs=1e-5)


def test_random_tokens_have_positive_rmsc():
    x = torch.randn(4, 32, 64)
    assert RMSSpatialContrast().compute(x) > 0.5
