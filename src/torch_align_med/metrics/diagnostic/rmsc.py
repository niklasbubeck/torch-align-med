"""RMS Spatial Contrast (RMSC).

Reference: Singh et al., "What matters for Representation Alignment"
(iREPA), arXiv:2512.10794.

Operates on unit-normalised tokens :math:`\\hat{x}_t = x_t / \\|x_t\\|`:

.. math:: \\text{RMSC}(X) = \\sqrt{\\frac{1}{T}\\sum_t \\|\\hat{x}_t - \\bar{x}\\|_2^2}

where :math:`\\bar{x} = \\frac{1}{T}\\sum_t \\hat{x}_t`. Higher means tokens
are more spread on the unit hypersphere — a single-feature property, no
spatial coordinates needed.
"""
from __future__ import annotations

import torch

from torch_align_med.base import EncoderDiagnostic


class RMSSpatialContrast(EncoderDiagnostic):
    name = "rmsc"
    requires_coords = False
    higher_is_better = True

    def compute(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None = None,  # noqa: ARG002 - signature parity
        **_: object,
    ) -> float:
        if feats.ndim != 3:
            raise ValueError("RMSC expects feats of shape [B, N, D]")
        x = feats.float()
        x_hat = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_bar = x_hat.mean(dim=1, keepdim=True)  # [B, 1, D]
        sq = (x_hat - x_bar).pow(2).sum(dim=-1)  # [B, N]
        rms = sq.mean(dim=-1).sqrt()  # [B]
        return float(rms.mean().item())
