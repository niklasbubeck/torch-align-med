"""3DINO-ViT wrapper (AICONSlab).

Repo: https://github.com/AICONSlab/3DINO  (HF: ``AICONSlab/3DINO-ViT``)
License: **CC-BY-NC-ND 4.0** — research / non-commercial only.

3DINO is a ViT trained with a 3D-extended DINOv2 objective on ~100k 3D
medical scans (CT + MRI, 10+ organs). It produces patch tokens analogous
to 2D DINOv2 but on a 3D voxel grid, so the spatial-structure metrics
(LDS, CDS, RMSC) work natively with a 3D ``coords`` tensor.

Caveats handled here:

- The upstream repo pins Python 3.9. This wrapper only touches it lazily;
  the rest of ``torch-align-med`` works on Python ≥ 3.10. If the lazy
  import fails on your Python, fall back to running the upstream repo in
  its own 3.9 env, dumping features to ``.pt``, and using
  ``calculate_alignment_metrics`` on the saved tensors directly.
- The state-dict layout follows DINOv2 conventions extended to 3D
  (``conv3d`` patch embed, sinusoidal 3D positional embedding). A minimal
  loader is on the roadmap; the current wrapper relies on the upstream
  ``dinov2`` package being importable.
"""
from __future__ import annotations

import torch

from torch_align_med.encoders.base import BaseEncoder


class ThreeDinoEncoder(BaseEncoder):
    repo_id = "AICONSlab/3DINO-ViT"
    license = "CC-BY-NC-ND-4.0"

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        device: str | torch.device = "cpu",
        grid_shape: tuple[int, int, int] = (8, 8, 8),
    ):
        try:
            # Upstream package; install per the 3DINO README.
            from dinov2.eval.setup import setup_and_build_model  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ThreeDinoEncoder requires the upstream 3DINO package. "
                "See https://github.com/AICONSlab/3DINO for installation. "
                "If the Python version constraint blocks you, run the upstream "
                "repo to dump features to .pt and feed those into "
                "calculate_alignment_metrics directly."
            ) from e

        self.device = torch.device(device)
        self.model, _ = setup_and_build_model(config_path, weights_path)
        self.model = self.model.to(self.device).eval()
        self._grid = grid_shape

    @torch.inference_mode()
    def extract_tokens(self, volumes: torch.Tensor) -> torch.Tensor:
        # volumes: [B, C, D, H, W]
        out = self.model(volumes.to(self.device), is_training=True)
        # 3DINO follows DINOv2 conventions: patch tokens are in the
        # ``x_norm_patchtokens`` field of the dict output.
        return out["x_norm_patchtokens"]

    @torch.inference_mode()
    def extract_pooled(self, volumes: torch.Tensor) -> torch.Tensor:
        out = self.model(volumes.to(self.device), is_training=True)
        return out["x_norm_clstoken"]

    def grid_shape(self) -> tuple[int, int, int]:
        return self._grid
