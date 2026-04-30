"""Command-line interface mirroring ``torch-fidelity``'s ``fidelity``.

Loads two feature tensors from disk (``.pt`` / ``.npy``), optionally a
coords tensor, computes the selected metrics, and prints a JSON dict.

Example::

    torch-align-med --input1 a.pt --input2 b.pt --coords coords.pt \\
        --linear-cka --lds --rmsc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from torch_align_med.api import calculate_alignment_metrics


def _load(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    if suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if hasattr(arr, "files"):
            arr = arr[arr.files[0]]
        return torch.as_tensor(arr)
    raise ValueError(f"unsupported file type: {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("torch-align-med")
    p.add_argument("--input1", type=Path, required=True)
    p.add_argument("--input2", type=Path, required=True)
    p.add_argument("--coords", type=Path, default=None)
    for flag in ("patch-cosine", "linear-cka", "l-mdms", "lds", "cds", "rmsc"):
        p.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args(argv)

    feats_a = _load(args.input1)
    feats_b = _load(args.input2)
    coords = _load(args.coords) if args.coords else None

    result = calculate_alignment_metrics(
        feats_a,
        feats_b,
        coords=coords,
        patch_cosine=args.patch_cosine,
        linear_cka=args.linear_cka,
        l_mdms=args.l_mdms,
        lds=args.lds,
        cds=args.cds,
        rmsc=args.rmsc,
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
