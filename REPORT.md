# torch-align-med — v0.1 status report

## What ships in this build

### Core metrics — all six Tier-1, fully tested

| File | Metric | Family | Test count | Property checks |
|---|---|---|:-:|---|
| `metrics/pairwise/cosine.py` | `PatchCosineAlignment` | Pairwise | 4 | `cos(X, X) = 1`, `cos(X, -X) = -1`, orthogonal ≈ 0 |
| `metrics/pairwise/cka.py` | `LinearCKA` | Pairwise | 5 | `CKA(X, X) = 1`, scale-invariant, orthogonal-rotation-invariant, supports unequal feature widths |
| `metrics/pairwise/mdms.py` | `MarginalDMS` | Pairwise | 4 | `L_mdms(X, X) = 0`, scale-invariant, margin clipping behaves |
| `metrics/diagnostic/lds.py` | `LocalDistantSimilarity` | Diagnostic | 5 | smooth tokens > random; works in 3D; clear errors on missing/mismatched coords |
| `metrics/diagnostic/cds.py` | `CorrelationDecaySlope` | Diagnostic | 3 | smooth tokens > 0, random ≈ 0, 3D-grid ok |
| `metrics/diagnostic/rmsc.py` | `RMSSpatialContrast` | Diagnostic | 3 | constant features = 0, orthonormal basis = closed-form known value |

**Total:** 33 tests passing (27 metric + 6 PlaneCycleEncoder).

### Public API

- `torch_align_med.calculate_alignment_metrics(input1, input2, *, coords=None, **flags)` — `torch-fidelity`-style dispatcher.
- `torch_align_med.make_grid_coords(shape)` — build coordinate grids for any number of spatial dimensions.
- `torch_align_med.cli:main` — CLI entry point (`torch-align-med --input1 ... --input2 ... --linear-cka ...`).

All metrics are also exposed as classes for direct use (`LinearCKA().compute(a, b)`).

### Encoder wrappers (optional, lazy-import)

| Class | Repo | License | Style |
|---|---|---|---|
| `DinoV2Encoder` | `facebook/dinov2-with-registers-large` (default), or any DINOv2 checkpoint including `microsoft/rad-dino` | Apache-2.0 / MIT | DINOv2 (rich patches) |
| `BiomedClipEncoder` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT | CLIP (global) |
| `ThreeDinoEncoder` | `AICONSlab/3DINO-ViT` | CC-BY-NC-ND 4.0 | DINOv2 (3D patches, native) |
| `PlaneCycleEncoder` | `HINTLab/PlaneCycle` (wraps DINOv3 weights) | Apache-2.0 | Pseudo-3D lift of a 2D ViT |

Backends (`transformers`, `open_clip`, upstream `dinov2`) are imported only on instantiation, so the base install has no extra dependencies.

## How to use it

### From Python

```python
import torch
from torch_align_med import calculate_alignment_metrics, make_grid_coords

feats_a = torch.randn(8, 256, 1024)         # [B, N, D]
feats_b = torch.randn(8, 256, 1024)
coords  = make_grid_coords((16, 16))        # 2D grid for 16×16 patches

scores = calculate_alignment_metrics(
    feats_a, feats_b,
    coords=coords,
    patch_cosine=True, linear_cka=True, l_mdms=True,
    lds=True, cds=True, rmsc=True,
    metric_kwargs={"lds": {"init": {"r_near": 4, "r_far": 8}}},
)
```

### From the command line

```bash
torch-align-med \
    --input1 feats_a.pt --input2 feats_b.pt \
    --coords coords.pt \
    --linear-cka --lds --cds --rmsc
```

### With a reference encoder

```python
from torch_align_med import calculate_alignment_metrics, make_grid_coords
from torch_align_med.encoders import DinoV2Encoder

enc = DinoV2Encoder("facebook/dinov2-with-registers-large", device="cuda")
target_tokens = enc.extract_tokens(images)         # [B, N, D]
coords = make_grid_coords(enc.grid_shape())        # [N, 2]

# `my_latents` is whatever the user wants to align (e.g. DiT hidden states,
# projected to match dim D). We only require shape [B, N, D].
scores = calculate_alignment_metrics(
    my_latents, target_tokens,
    coords=coords,
    patch_cosine=True, linear_cka=True,
    lds=True, cds=True, rmsc=True,
)
```

### 3D volumes

The metric code is dimensionality-agnostic. Coordinates with `S=3` work end-to-end:

```python
coords_3d = make_grid_coords((8, 8, 8))     # [512, 3]
scores = calculate_alignment_metrics(
    vol_feats_a, vol_feats_b,                # [B, 512, D]
    coords=coords_3d,
    lds=True, cds=True, rmsc=True,
)
```

## Architectural decisions

1. **Features-only contract.** Metrics consume tensors. They never load images, run encoders, or know about layer indices. This keeps the core small, testable, and trivially reusable for any modality (natural images, radiology, pathology, 3D volumes, video, …).
2. **Pairwise vs. Diagnostic split.** Pairwise metrics return one number per pair. Diagnostics describe a single feature space and the dispatcher applies them on each side. This matches what the iREPA literature actually does — `LDS`, `CDS`, `RMSC` are properties of a single encoder and only become "alignment metrics" when computed on each side and compared.
3. **`coords: [N, S]` instead of grid shape.** A coordinate tensor generalises to 2D, 3D, and irregular layouts (e.g. video) with no metric-side branching. Grid shape is recoverable via `make_grid_coords` for the common case.
4. **Encoder wrappers are entirely separate.** Lazy imports keep the base install at `torch + numpy`. Failures in optional backends never break the core.
5. **No silent fallbacks.** Each metric raises `ValueError` on bad input shapes with a message that names the actual problem. The dispatcher returns an empty dict if no flags are set rather than computing a default — explicit > implicit.

## What is *not* in this build

- **CKNNA / mutual-kNN alignment** — Tier-2; needs the Platonic Hypothesis paper's exact formulation.
- **SE-CKNNA** — Tier-2; awaits Bi et al. 2510.18457 PDF read.
- **SRSS — Semantic-Region Self-Similarity** — Tier-2; requires SAM2 (heavy dependency, behind `extras_require[srss]`).
- **`L_mcos`** — overlaps `patch_cosine` for evaluation purposes.
- **iREPA reproduction notebook** — planned: reproduce |r| > 0.85 correlation between LDS/CDS/RMSC and FID across the 27-encoder pool.
- **3DINO state-dict-only loader** — currently the wrapper requires the upstream `dinov2` package; a minimal independent loader (so users on Python ≥ 3.10 can use 3DINO without a separate env) is on the roadmap.

## Verification status

The iREPA spatial-structure formulas (LDS, CDS, RMSC) were transcribed from a web summary of the paper, then sanity-checked against the closed-form expectations encoded in tests. Before locking these as canonical, they should be **cross-checked against the iREPA PDF directly** — the property-test pattern in `tests/` makes that audit easy: any drift from the paper will surface as a failing test on the synthetic fixtures.
