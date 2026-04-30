"""Public dispatcher: ``calculate_alignment_metrics``.

Mirrors the ``torch-fidelity`` ergonomic — pass two feature tensors plus
boolean flags, get a dict of scores back.
"""
from __future__ import annotations

import torch

from torch_align_med.metrics import (
    CorrelationDecaySlope,
    LinearCKA,
    LocalDistantSimilarity,
    MarginalDMS,
    PatchCosineAlignment,
    RMSSpatialContrast,
)

PAIRWISE_REGISTRY = {
    "patch_cosine": PatchCosineAlignment,
    "linear_cka": LinearCKA,
    "l_mdms": MarginalDMS,
}

DIAGNOSTIC_REGISTRY = {
    "lds": LocalDistantSimilarity,
    "cds": CorrelationDecaySlope,
    "rmsc": RMSSpatialContrast,
}


def calculate_alignment_metrics(
    input1: torch.Tensor,
    input2: torch.Tensor,
    *,
    coords: torch.Tensor | None = None,
    patch_cosine: bool = False,
    linear_cka: bool = False,
    l_mdms: bool = False,
    lds: bool = False,
    cds: bool = False,
    rmsc: bool = False,
    metric_kwargs: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Compute selected alignment metrics between two feature tensors.

    Parameters
    ----------
    input1, input2:
        Token tensors of shape ``[B, N, D]`` or pooled tensors ``[B, D]``.
        Some metrics (``patch_cosine``, ``l_mdms``) require token-level
        inputs; others (``linear_cka``, ``rmsc``) work on either.
    coords:
        Optional ``[N, S]`` integer coordinate grid. Required by spatial
        diagnostics (``lds``, ``cds``).
    metric_kwargs:
        Optional per-metric kwargs, keyed by registry name. e.g.
        ``{"lds": {"r_near": 4, "r_far": 8}}``.

    Returns
    -------
    dict
        ``{metric_name: score}``. Diagnostic metrics emit one entry per
        side (``{name}_input1`` and ``{name}_input2``).
    """
    metric_kwargs = metric_kwargs or {}
    requested_pairwise = {
        "patch_cosine": patch_cosine,
        "linear_cka": linear_cka,
        "l_mdms": l_mdms,
    }
    requested_diagnostic = {"lds": lds, "cds": cds, "rmsc": rmsc}

    out: dict[str, float] = {}

    for name, on in requested_pairwise.items():
        if not on:
            continue
        cls = PAIRWISE_REGISTRY[name]
        kwargs = metric_kwargs.get(name, {})
        try:
            metric = cls(**kwargs.get("init", {}))
        except TypeError:
            metric = cls()
        out[name] = metric.compute(input1, input2, **kwargs.get("call", {}))

    for name, on in requested_diagnostic.items():
        if not on:
            continue
        cls = DIAGNOSTIC_REGISTRY[name]
        kwargs = metric_kwargs.get(name, {})
        try:
            metric = cls(**kwargs.get("init", {}))
        except TypeError:
            metric = cls()
        out[f"{name}_input1"] = metric.compute(
            input1, coords=coords, **kwargs.get("call", {})
        )
        out[f"{name}_input2"] = metric.compute(
            input2, coords=coords, **kwargs.get("call", {})
        )

    return out
