import pytest
import torch

from torch_align_med import LinearCKA


def test_self_cka_is_one():
    x = torch.randn(64, 32)
    assert LinearCKA().compute(x, x) == pytest.approx(1.0, abs=1e-5)


def test_scale_invariance():
    x = torch.randn(64, 32)
    assert LinearCKA().compute(x, 5.7 * x) == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_transform_invariance():
    torch.manual_seed(0)
    x = torch.randn(128, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert LinearCKA().compute(x, x @ q) == pytest.approx(1.0, abs=1e-4)


def test_different_widths_supported():
    x = torch.randn(128, 16)
    y = torch.randn(128, 64)
    score = LinearCKA().compute(x, y)
    assert 0.0 <= score <= 1.0


def test_token_tensor_flattened():
    x = torch.randn(4, 16, 32)
    pooled = LinearCKA().compute(x, x)
    flat = LinearCKA().compute(x.reshape(-1, 32), x.reshape(-1, 32))
    assert pooled == pytest.approx(flat, abs=1e-5)
