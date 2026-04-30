import pytest
import torch

from torch_align_med import PatchCosineAlignment


def test_self_alignment_is_one():
    x = torch.randn(2, 8, 16)
    assert PatchCosineAlignment().compute(x, x) == pytest.approx(1.0, abs=1e-5)


def test_negation_is_minus_one():
    x = torch.randn(2, 8, 16)
    assert PatchCosineAlignment().compute(x, -x) == pytest.approx(-1.0, abs=1e-5)


def test_orthogonal_is_near_zero():
    x = torch.randn(8, 64, 128)
    y = torch.randn(8, 64, 128)
    score = PatchCosineAlignment().compute(x, y)
    assert abs(score) < 0.05


def test_shape_mismatch_errors():
    with pytest.raises(ValueError, match="must share shape"):
        PatchCosineAlignment().compute(torch.randn(2, 8, 16), torch.randn(2, 8, 32))
