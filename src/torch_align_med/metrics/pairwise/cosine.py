"""Patch-token cosine alignment (REPA-style).

Reference: Yu et al., "Representation Alignment for Generation", arXiv:2410.06940.

Mean cosine similarity over corresponding tokens. Both inputs must share
shape ``[B, N, D]`` (or ``[B, D]`` for pooled features). Token order matters:
position ``n`` in ``feats_a`` is paired with position ``n`` in ``feats_b``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_align_med.base import PairwiseMetric


class PatchCosineAlignment(PairwiseMetric):
    name = "patch_cosine"
    higher_is_better = True

    def compute(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        **_: object,
    ) -> float:
        if feats_a.shape != feats_b.shape:
            raise ValueError(
                f"feats_a and feats_b must share shape, got {feats_a.shape} vs {feats_b.shape}. "
                "Project one side to match the other before calling."
            )
        if feats_a.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D input, got ndim={feats_a.ndim}")
        sim = F.cosine_similarity(feats_a, feats_b, dim=-1)
        return float(sim.mean().item())
