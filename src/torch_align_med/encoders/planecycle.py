"""PlaneCycle wrapper — pseudo-3D feature extraction from a 2D ViT backbone.

Repo: https://github.com/HINTLab/PlaneCycle  (Apache-2.0)

PlaneCycle is a *training-free* converter that rewrites a frozen DINOv3 ViT
to process volumetric input by cycling its transformer blocks across the
three orthogonal planes (HW, DW, DH). The result is a 3D-aware feature
extractor with no learned 3D parameters.

This wrapper is deliberately thin: it accepts an already-loaded,
already-converted backbone and turns its output into the
``[B, N_3d, D] + coords: [N_3d, 3]`` shape that the rest of
``torch-align-med`` consumes. Loading and converting the DINOv3 ViT itself
is the user's responsibility — the upstream API is tightly coupled to
``block_type="PlaneCycle"`` and a custom ``forward_features`` path, so
inlining the loader here would mean vendoring the upstream model code.

Example (with the upstream PlaneCycle repo and DINOv3 weights available)::

    import torch
    from planecycle.converters.converter import PlaneCycleConverter

    model = torch.hub.load(
        repo_or_dir="path/to/PlaneCycle/models",
        model="dinov3_vitl16",
        source="local",
        pretrained=False,
        block_type="PlaneCycle",
    )
    model.load_state_dict(torch.load("dinov3_vitl16.pth"), strict=False)
    model = PlaneCycleConverter(cycle_order=("HW", "DW", "DH", "HW"),
                                pool_method="PCg")(model)

    from torch_align_med.encoders import PlaneCycleEncoder
    enc = PlaneCycleEncoder(model, patch_size=16, device="cuda")

    vol = torch.randn(2, 3, 32, 224, 224)        # [B, C, D, H, W]
    tokens = enc.extract_tokens(vol)              # [B, D*h*w, C_feat]
    coords = make_grid_coords(enc.grid_shape())   # [D*h*w, 3]
"""
from __future__ import annotations

import torch
from torch import nn

from torch_align_med.encoders.base import BaseEncoder


class PlaneCycleEncoder(BaseEncoder):
    repo_id = "HINTLab/PlaneCycle"
    license = "Apache-2.0"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_size: int = 16,
        device: str | torch.device = "cpu",
    ):
        if not hasattr(backbone, "get_intermediate_layers"):
            raise TypeError(
                "backbone must expose `get_intermediate_layers(x, n, reshape=True)`. "
                "Pass a DINOv3 ViT loaded with block_type='PlaneCycle' and converted "
                "via `PlaneCycleConverter`. See class docstring for the full loading "
                "recipe."
            )
        self.backbone = backbone.to(device).eval()
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self._grid: tuple[int, int, int] | None = None

    @torch.inference_mode()
    def extract_tokens(self, volumes: torch.Tensor) -> torch.Tensor:
        """Volumes ``[B, C, D, H, W]`` → tokens ``[B, D*h*w, C_feat]``."""
        if volumes.ndim != 5:
            raise ValueError(
                f"expected 5D volumes [B, C, D, H, W], got shape {tuple(volumes.shape)}"
            )
        out = self.backbone.get_intermediate_layers(
            volumes.to(self.device), n=1, reshape=True
        )
        feats = out[0] if isinstance(out, (list, tuple)) else out  # [B, C_feat, D, h, w]
        if feats.ndim != 5:
            raise RuntimeError(
                f"expected 5D feature map from get_intermediate_layers(reshape=True), "
                f"got shape {tuple(feats.shape)}"
            )
        _, _, d, h, w = feats.shape
        self._grid = (d, h, w)
        # [B, C, D, h, w] -> [B, D*h*w, C]
        return feats.flatten(2).transpose(1, 2).contiguous()

    @torch.inference_mode()
    def extract_pooled(self, volumes: torch.Tensor) -> torch.Tensor:
        if volumes.ndim != 5:
            raise ValueError(
                f"expected 5D volumes [B, C, D, H, W], got shape {tuple(volumes.shape)}"
            )
        # The upstream ViT returns a dict when called with is_training=True.
        out = self.backbone(volumes.to(self.device), is_training=True)
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
        # Fallback: mean-pool the patch tokens.
        return self.extract_tokens(volumes).mean(dim=1)

    def grid_shape(self) -> tuple[int, int, int]:
        if self._grid is None:
            raise RuntimeError(
                "grid_shape is determined on the first call to `extract_tokens`. "
                "Run a forward pass first."
            )
        return self._grid
