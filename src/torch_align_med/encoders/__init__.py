"""Optional convenience wrappers around standard alignment-target encoders.

Core metrics never depend on this module — it is pure convenience for users
who want a one-liner that turns images into feature tensors. Each wrapper
lazily imports its backend (``transformers``, ``open_clip``) so the base
install stays minimal.

Recommended starting points (per-modality):

============== ==========================================================
2D natural     :class:`DinoV2Encoder` with ``facebook/dinov2-with-registers-large``
2D medical     :class:`DinoV2Encoder` with ``microsoft/rad-dino`` (radiology),
               or :class:`BiomedClipEncoder` (cross-modality, pooled only)
3D medical     :class:`ThreeDinoEncoder` with ``AICONSlab/3DINO-ViT``
               (CC-BY-NC-ND, research only, see class docstring)
============== ==========================================================
"""
from torch_align_med.encoders.base import BaseEncoder
from torch_align_med.encoders.biomedclip import BiomedClipEncoder
from torch_align_med.encoders.dinov2 import DinoV2Encoder
from torch_align_med.encoders.planecycle import PlaneCycleEncoder
from torch_align_med.encoders.threedino import ThreeDinoEncoder

__all__ = [
    "BaseEncoder",
    "BiomedClipEncoder",
    "DinoV2Encoder",
    "PlaneCycleEncoder",
    "ThreeDinoEncoder",
]
