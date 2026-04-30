import pytest
import torch

from torch_align_med import MarginalDMS


def test_self_alignment_is_zero():
    x = torch.randn(2, 16, 32)
    assert MarginalDMS().compute(x, x) == pytest.approx(0.0, abs=1e-6)


def test_scaling_does_not_change_score():
    x = torch.randn(2, 16, 32)
    y = 3.0 * x
    assert MarginalDMS().compute(x, y) == pytest.approx(0.0, abs=1e-5)


def test_unrelated_inputs_have_positive_score():
    x = torch.randn(2, 16, 32)
    y = torch.randn(2, 16, 32)
    assert MarginalDMS().compute(x, y) > 0.0


def test_margin_clips_small_differences():
    x = torch.randn(2, 16, 32)
    y = x + 0.01 * torch.randn_like(x)
    score = MarginalDMS(margin=1.0).compute(x, y)
    assert score == pytest.approx(0.0, abs=1e-6)
