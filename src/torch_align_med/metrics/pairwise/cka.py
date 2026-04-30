"""Linear Centered Kernel Alignment (CKA).

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019, arXiv:1905.00414.

For two feature matrices :math:`X \\in \\mathbb{R}^{n \\times d_x}` and
:math:`Y \\in \\mathbb{R}^{n \\times d_y}` (centered along the sample axis),

.. math:: \\text{CKA}(X, Y) = \\frac{\\|Y^\\top X\\|_F^2}{\\|X^\\top X\\|_F \\, \\|Y^\\top Y\\|_F}

Result lies in ``[0, 1]``. Symmetric and invariant to orthogonal transforms
of either argument. Token tensors ``[B, N, D]`` are flattened to
``[B*N, D]`` (each token contributes one sample).
"""
from __future__ import annotations

import torch

from torch_align_med.base import PairwiseMetric


def _flatten_samples(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"expected 2D or 3D input, got ndim={x.ndim}")


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = _center(_flatten_samples(x).float())
    y = _center(_flatten_samples(y).float())
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"sample counts disagree after flattening: {x.shape[0]} vs {y.shape[0]}"
        )
    yx = y.T @ x
    xx = x.T @ x
    yy = y.T @ y
    num = (yx * yx).sum()
    den = xx.norm() * yy.norm()
    if den <= 0:
        return 0.0
    return float((num / den).item())


class LinearCKA(PairwiseMetric):
    name = "linear_cka"
    higher_is_better = True

    def compute(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        **_: object,
    ) -> float:
        return linear_cka(feats_a, feats_b)
