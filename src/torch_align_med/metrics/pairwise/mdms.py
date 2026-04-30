"""Marginal Distance-Matrix Similarity (L_mdms).

Reference: Yao et al., "Reconstruction vs. Generation: Taming Optimization
Dilemma in Latent Diffusion Models" (VA-VAE), arXiv:2501.01423.

Compares the cosine self-similarity matrices of the two feature spaces:

.. math:: L_{\\text{mdms}} = \\frac{1}{N^2} \\sum_{i,j}
   \\operatorname{ReLU}\\!\\left(\\,
   \\left|\\, S^{a}_{ij} - S^{b}_{ij}\\,\\right| - m_2
   \\,\\right)

with :math:`S^{x}_{ij} = \\cos(x_i, x_j)`. Margin :math:`m_2` defaults to 0
for evaluation use (the original paper uses a non-zero margin for training).

Lower means more aligned relational structure. Dimensionality-agnostic on
:math:`D`: ``feats_a`` and ``feats_b`` may have different feature widths.
"""
from __future__ import annotations

import torch

from torch_align_med._grids import cosine_self_similarity
from torch_align_med.base import PairwiseMetric


class MarginalDMS(PairwiseMetric):
    name = "l_mdms"
    higher_is_better = False

    def __init__(self, margin: float = 0.0):
        self.margin = float(margin)

    def compute(
        self,
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        **_: object,
    ) -> float:
        if feats_a.ndim != 3 or feats_b.ndim != 3:
            raise ValueError(
                "L_mdms expects token tensors of shape [B, N, D]; pool first if needed"
            )
        if feats_a.shape[:2] != feats_b.shape[:2]:
            raise ValueError(
                f"batch and token counts must match: {feats_a.shape[:2]} vs {feats_b.shape[:2]}"
            )
        s_a = cosine_self_similarity(feats_a.float())
        s_b = cosine_self_similarity(feats_b.float())
        diff = (s_a - s_b).abs() - self.margin
        diff = diff.clamp_min(0.0)
        return float(diff.mean().item())
