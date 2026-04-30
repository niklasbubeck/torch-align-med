"""BiomedCLIP wrapper via ``open_clip``.

Repo: ``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224``
License: MIT, ungated.

BiomedCLIP is a CLIP-style contrastive model trained on PMC-15M. The image
encoder is a ViT-B/16 at 224×224. **Global features dominate** in
CLIP-style models, so BiomedCLIP is a strong target for pairwise metrics
that consume pooled embeddings (REPA-cosine, Linear CKA on pooled inputs)
but a *poor* target for spatial-structure diagnostics (LDS, CDS, RMSC).

Use :meth:`extract_pooled` for the canonical use case. :meth:`extract_tokens`
is provided but the patch tokens are less semantically structured than a
DINOv2-style encoder; do not expect iREPA's |r| > 0.85 finding to hold.
"""
from __future__ import annotations

import torch

from torch_align_med.encoders.base import BaseEncoder


class BiomedClipEncoder(BaseEncoder):
    repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    license = "MIT"

    def __init__(
        self,
        device: str | torch.device = "cpu",
    ):
        try:
            from open_clip import create_model_from_pretrained
        except ImportError as e:
            raise ImportError(
                "BiomedClipEncoder requires the `encoders-medical` extra. "
                "Install with: pip install 'torch-align-med[encoders-medical]'"
            ) from e

        self.device = torch.device(device)
        self.model, self.preprocess = create_model_from_pretrained(
            f"hf-hub:{self.repo_id}"
        )
        self.model = self.model.to(self.device).eval()
        self._grid = (14, 14)  # 224 / 16

    @torch.inference_mode()
    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # open_clip ViTs expose ``trunk`` for timm-vit; access patch tokens via the
        # transformer block stack and final layer norm.
        visual = self.model.visual
        x = visual.trunk.forward_features(images.to(self.device))
        # timm ViTs return [B, 1+N, D] including CLS.
        return x[:, 1:, :].contiguous()

    @torch.inference_mode()
    def extract_pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images.to(self.device))

    def grid_shape(self) -> tuple[int, int]:
        return self._grid
