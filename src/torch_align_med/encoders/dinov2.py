"""DINOv2 / RAD-DINO wrapper via Hugging Face ``transformers``.

Works with any DINOv2-architecture checkpoint that exposes patch tokens via
the standard ``last_hidden_state`` output, including:

- ``facebook/dinov2-base``, ``-large``, ``-giant`` (Apache-2.0, ungated)
- ``facebook/dinov2-with-registers-base``, ``-large``, ``-giant``
  (Apache-2.0, ungated, has 4 register tokens)
- ``microsoft/rad-dino`` (MIT, ungated, chest-X-ray DINOv2-base finetune)
- DINOv3 variants (gated; license differs)

The wrapper drops CLS and register tokens by default so the returned tensor
contains *only* patch tokens — that is the canonical input shape for the
spatial-structure metrics.
"""
from __future__ import annotations

import torch

from torch_align_med.encoders.base import BaseEncoder


class DinoV2Encoder(BaseEncoder):
    license = "varies — check the model card"

    def __init__(
        self,
        repo_id: str = "facebook/dinov2-with-registers-large",
        device: str | torch.device = "cpu",
        num_register_tokens: int | None = None,
    ):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise ImportError(
                "DinoV2Encoder requires the `encoders-2d` extra. "
                "Install with: pip install 'torch-align-med[encoders-2d]'"
            ) from e

        self.repo_id = repo_id
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(repo_id).to(self.device).eval()

        cfg = self.model.config
        if num_register_tokens is not None:
            self._n_register = num_register_tokens
        else:
            self._n_register = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self._n_special = 1 + self._n_register  # CLS + registers

        patch = int(getattr(cfg, "patch_size", 14))
        size = int(getattr(cfg, "image_size", 224))
        self._grid = (size // patch, size // patch)

    @torch.inference_mode()
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=images.to(self.device))
        return out.last_hidden_state  # [B, 1+n_reg+H*W, D]

    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        h = self._forward(images)
        return h[:, self._n_special :, :].contiguous()

    def extract_pooled(self, images: torch.Tensor) -> torch.Tensor:
        h = self._forward(images)
        return h[:, 0, :].contiguous()  # CLS

    def grid_shape(self) -> tuple[int, int]:
        return self._grid
