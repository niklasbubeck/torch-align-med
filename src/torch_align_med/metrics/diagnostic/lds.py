"""Local vs. Distant Similarity (LDS).

Reference: Singh et al., "What matters for Representation Alignment"
(iREPA), arXiv:2512.10794.

.. math:: \\text{LDS}(X) =
   \\mathbb{E}\\!\\left[K(t, t') \\,\\big|\\, d(t, t') < r_{\\text{near}}\\right]
   \\, - \\,
   \\mathbb{E}\\!\\left[K(t, t') \\,\\big|\\, d(t, t') \\ge r_{\\text{far}}\\right]

with :math:`K` the cosine similarity kernel and :math:`d` the Manhattan
distance on the spatial grid. Default ``r_near = r_far = max_dim / 2``.

Higher means richer local structure relative to global structure.
Dimension-agnostic: 2D, 3D, or any S-dim grid via ``coords: [N, S]``.
"""
from __future__ import annotations

import torch

from torch_align_med._grids import cosine_self_similarity, manhattan_distance_matrix
from torch_align_med.base import EncoderDiagnostic


class LocalDistantSimilarity(EncoderDiagnostic):
    name = "lds"
    requires_coords = True
    higher_is_better = True

    def __init__(self, r_near: int | None = None, r_far: int | None = None):
        self.r_near = r_near
        self.r_far = r_far

    def compute(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor | None = None,
        **_: object,
    ) -> float:
        if coords is None:
            raise ValueError("LDS requires `coords: [N, S]` to define spatial neighbourhoods")
        if feats.ndim != 3:
            raise ValueError("LDS expects feats of shape [B, N, D]")
        n_coords = coords.shape[0]
        if feats.shape[1] != n_coords:
            raise ValueError(
                f"coords has N={n_coords} but feats has N={feats.shape[1]}"
            )

        d = manhattan_distance_matrix(coords)
        max_d = int(d.max().item())
        r_near = self.r_near if self.r_near is not None else max_d // 2
        r_far = self.r_far if self.r_far is not None else max_d // 2
        if r_near < 1 or r_far < 1:
            raise ValueError("r_near and r_far must be >= 1; pass non-trivial coords")

        sim = cosine_self_similarity(feats.float())  # [B, N, N]

        # Exclude self-pairs (d == 0).
        near_mask = (d > 0) & (d < r_near)
        far_mask = d >= r_far
        if not near_mask.any() or not far_mask.any():
            raise ValueError(
                f"empty near/far set with r_near={r_near}, r_far={r_far}, max_d={max_d}"
            )

        near_mean = sim[:, near_mask].mean()
        far_mean = sim[:, far_mask].mean()
        return float((near_mean - far_mean).item())
