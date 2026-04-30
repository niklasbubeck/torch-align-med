"""Correlation Decay Slope (CDS).

Reference: Singh et al., "What matters for Representation Alignment"
(iREPA), arXiv:2512.10794.

For each integer Manhattan distance :math:`\\delta \\in \\{1, \\dots, \\Delta\\}`,

.. math:: g(\\delta) = \\mathbb{E}\\!\\left[\\,K(t,t') \\,\\big|\\, d(t,t') = \\delta\\,\\right]

then a least-squares line :math:`g(\\delta) \\approx \\alpha + \\beta\\delta`
is fit, and ``CDS = -beta``. Higher means similarity decays *faster* with
distance — i.e. richer local clustering.
"""
from __future__ import annotations

import torch

from torch_align_med._grids import cosine_self_similarity, manhattan_distance_matrix
from torch_align_med.base import EncoderDiagnostic


class CorrelationDecaySlope(EncoderDiagnostic):
    name = "cds"
    requires_coords = True
    higher_is_better = True

    def __init__(self, max_distance: int | None = None):
        self.max_distance = max_distance

    def compute(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None = None,
        **_: object,
    ) -> float:
        if coords is None:
            raise ValueError("CDS requires `coords: [N, S]`")
        if feats.ndim != 3:
            raise ValueError("CDS expects feats of shape [B, N, D]")

        d = manhattan_distance_matrix(coords)
        max_d = int(d.max().item())
        delta_max = self.max_distance if self.max_distance is not None else max_d
        if delta_max < 2:
            raise ValueError("CDS needs at least two distance bins")

        sim = cosine_self_similarity(feats.float()).mean(dim=0)  # [N, N], avg over batch

        deltas: list[float] = []
        means: list[float] = []
        for delta in range(1, delta_max + 1):
            mask = d == delta
            if not mask.any():
                continue
            deltas.append(float(delta))
            means.append(float(sim[mask].mean().item()))
        if len(deltas) < 2:
            raise ValueError("Not enough non-empty distance bins to fit a slope")

        x = torch.tensor(deltas, dtype=torch.float64)
        y = torch.tensor(means, dtype=torch.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        beta = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        return float(-beta.item())
