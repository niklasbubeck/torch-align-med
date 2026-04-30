"""Tests for the PlaneCycleEncoder shape contract.

We don't import the upstream ``planecycle`` package; instead a tiny fake
backbone replays the contract (``get_intermediate_layers(..., reshape=True)``
returns ``[B, C, D, h, w]``). This keeps the test offline and Python-version
independent while still pinning down the wrapper's reshape logic.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_align_med import calculate_alignment_metrics, make_grid_coords
from torch_align_med.encoders import PlaneCycleEncoder


class FakeDinoV3PlaneCycleBackbone(nn.Module):
    """Returns a deterministic 5D feature map matching the upstream contract."""

    def __init__(self, c_feat: int = 32, patch_size: int = 16):
        super().__init__()
        self.c_feat = c_feat
        self.patch_size = patch_size

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: int = 1,
        reshape: bool = False,
        **_: object,
    ):
        assert reshape, "wrapper must request reshape=True"
        b, _c, d, h, w = x.shape
        h_tok = h // self.patch_size
        w_tok = w // self.patch_size
        # Use a content-dependent value so wrong reshapes would scramble it.
        # Offset by 1 so no token is the zero vector — cosine similarity on a
        # zero vector is undefined and our eps-clamp returns 0, which would
        # otherwise dilute self-alignment scores.
        idx = torch.arange(1, 1 + d * h_tok * w_tok, dtype=torch.float32)
        feat = idx.view(1, 1, d, h_tok, w_tok).expand(b, self.c_feat, -1, -1, -1).contiguous()
        return [feat] * (n if isinstance(n, int) else len(n))

    def forward(self, x: torch.Tensor, *, is_training: bool = True):
        b = x.shape[0]
        return {"x_norm_clstoken": torch.zeros(b, self.c_feat)}


def test_extract_tokens_shape_and_order():
    backbone = FakeDinoV3PlaneCycleBackbone(c_feat=8, patch_size=16)
    enc = PlaneCycleEncoder(backbone, patch_size=16)

    vol = torch.zeros(2, 3, 4, 32, 32)            # B=2, D=4, H=W=32 → h=w=2
    tokens = enc.extract_tokens(vol)
    assert tokens.shape == (2, 4 * 2 * 2, 8)
    assert enc.grid_shape() == (4, 2, 2)

    # The fake backbone wrote token index n into every channel; verify the
    # spatial flattening matches make_grid_coords' (D, h, w) layout.
    expected = torch.arange(1, 1 + 4 * 2 * 2, dtype=torch.float32).unsqueeze(-1).expand(-1, 8)
    assert torch.allclose(tokens[0], expected)


def test_grid_shape_unset_until_first_forward():
    enc = PlaneCycleEncoder(FakeDinoV3PlaneCycleBackbone())
    with pytest.raises(RuntimeError, match="first call"):
        enc.grid_shape()


def test_rejects_non_volume_input():
    enc = PlaneCycleEncoder(FakeDinoV3PlaneCycleBackbone())
    with pytest.raises(ValueError, match="5D volumes"):
        enc.extract_tokens(torch.zeros(2, 3, 32, 32))


def test_rejects_backbone_without_get_intermediate_layers():
    with pytest.raises(TypeError, match="get_intermediate_layers"):
        PlaneCycleEncoder(nn.Linear(8, 8))


def test_extract_pooled_returns_cls():
    enc = PlaneCycleEncoder(FakeDinoV3PlaneCycleBackbone(c_feat=16))
    pooled = enc.extract_pooled(torch.zeros(3, 3, 4, 32, 32))
    assert pooled.shape == (3, 16)


def test_metrics_consume_planecycle_output_end_to_end():
    backbone = FakeDinoV3PlaneCycleBackbone(c_feat=8, patch_size=16)
    enc = PlaneCycleEncoder(backbone)

    vol = torch.randn(2, 3, 4, 32, 32)
    tokens = enc.extract_tokens(vol)
    coords = make_grid_coords(enc.grid_shape())
    assert coords.shape == (tokens.shape[1], 3)

    out = calculate_alignment_metrics(
        tokens, tokens, coords=coords,
        patch_cosine=True, linear_cka=True, lds=True, cds=True, rmsc=True,
    )
    # Self-alignment sanity: pairwise metrics agree, diagnostics are equal on both sides.
    assert out["patch_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert out["linear_cka"] == pytest.approx(1.0, abs=1e-5)
    assert out["lds_input1"] == pytest.approx(out["lds_input2"], abs=1e-6)
    assert out["rmsc_input1"] == pytest.approx(out["rmsc_input2"], abs=1e-6)
