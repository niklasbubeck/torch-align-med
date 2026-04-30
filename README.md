# torch-align-med

**Semantic-alignment metrics between latent spaces — the alignment-evaluation analog of [`torch-fidelity`](https://github.com/toshas/torch-fidelity).**

When generative models are trained with representation alignment objectives (REPA, VA-VAE, REPA-E, iREPA, …), the inner question is always: *how aligned are the two latent spaces?* This package answers that question with six precise, reproducible metrics — works on 2D natural images, 2D medical images, and 3D medical volumes from the same call site.

## Install

```bash
pip install torch-align-med                              # core (torch + numpy)
pip install "torch-align-med[encoders-2d]"               # + DINOv2 wrappers
pip install "torch-align-med[encoders-medical]"          # + BiomedCLIP, RAD-DINO
pip install "torch-align-med[all]"                       # everything
```

## Quick start

```python
import torch
from torch_align_med import calculate_alignment_metrics, make_grid_coords

# Two feature tensors of shape [B, N, D] — produced however you like.
feats_a = torch.randn(8, 256, 1024)   # e.g. DiT layer-8 hidden states
feats_b = torch.randn(8, 256, 1024)   # e.g. DINOv2 patch tokens

coords = make_grid_coords((16, 16))   # spatial grid for LDS/CDS

scores = calculate_alignment_metrics(
    feats_a, feats_b,
    coords=coords,
    patch_cosine=True,    # REPA-style cosine alignment
    linear_cka=True,      # Kornblith et al. CKA
    l_mdms=True,          # VA-VAE relational alignment
    lds=True, cds=True, rmsc=True,  # iREPA spatial-structure diagnostics
)
# {'patch_cosine': ..., 'linear_cka': ..., 'l_mdms': ..., 'lds_input1': ..., ...}
```

`coords` is `[N, S]` — `S=2` for images, `S=3` for volumes. The same metric code works for both.

## Metrics

| Metric | Family | Higher is better | Reference |
|---|:-:|:-:|---|
| Patch-token cosine alignment | Pairwise | ✓ | REPA — Yu et al. 2024 ([2410.06940](https://arxiv.org/abs/2410.06940)) |
| Linear CKA | Pairwise | ✓ | Kornblith et al. 2019 ([1905.00414](https://arxiv.org/abs/1905.00414)) |
| `L_mdms` (relational) | Pairwise | ✗ (lower better) | VA-VAE — Yao et al. 2025 ([2501.01423](https://arxiv.org/abs/2501.01423)) |
| LDS — Local vs. Distant Similarity | Diagnostic | ✓ | iREPA — Singh et al. 2025 ([2512.10794](https://arxiv.org/abs/2512.10794)) |
| CDS — Correlation Decay Slope | Diagnostic | ✓ | iREPA |
| RMSC — RMS Spatial Contrast | Diagnostic | ✓ | iREPA |

The iREPA paper reports |r| > 0.85 between the spatial-structure diagnostics and downstream FID across 27 encoders — a much stronger signal than ImageNet-1K linear probing (|r| = 0.26).

**Pairwise vs. Diagnostic:** Pairwise metrics consume both feature tensors and emit one scalar. Diagnostic metrics describe a single feature space; the runner applies them on each side and emits `{name}_input1` / `{name}_input2`.

## Reference encoders

Optional convenience wrappers. Core metrics never depend on these.

| Modality | Wrapper | HF repo | License |
|---|---|---|---|
| 2D natural | `DinoV2Encoder` | `facebook/dinov2-with-registers-large` | Apache-2.0 |
| 2D medical (radiology) | `DinoV2Encoder` | `microsoft/rad-dino` | MIT |
| 2D medical (cross-modality) | `BiomedClipEncoder` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT |
| 3D medical (native) | `ThreeDinoEncoder` | `AICONSlab/3DINO-ViT` | CC-BY-NC-ND 4.0 |
| 3D medical (pseudo-3D) | `PlaneCycleEncoder` | `HINTLab/PlaneCycle` (lifts DINOv3) | Apache-2.0 |

```python
from torch_align_med.encoders import DinoV2Encoder

enc = DinoV2Encoder("microsoft/rad-dino", device="cuda")
tokens = enc.extract_tokens(images)          # [B, N, D]
coords = make_grid_coords(enc.grid_shape())  # [N, 2]
```

**Notes.** BiomedCLIP is CLIP-style — global features dominate, so it is a strong target for `patch_cosine` / `linear_cka` on **pooled** embeddings, but a *weak* target for spatial-structure diagnostics (`lds`, `cds`, `rmsc`). 3DINO is research-only (CC-BY-NC-ND); the wrapper lazy-imports the upstream package so the rest of `torch-align-med` is unaffected by its Python 3.9 pin. **PlaneCycle** is a permissive (Apache-2.0) alternative for 3D — it lifts a frozen 2D DINOv3 ViT into a 3D feature extractor without any training. The wrapper takes a pre-loaded converted backbone (loading is documented in the class docstring); spatial-structure metrics work natively on its `[B, D·h·w, C]` token output via a 3D `coords` grid.

## CLI

```bash
torch-align-med \
    --input1 feats_a.pt --input2 feats_b.pt \
    --coords coords.pt \
    --patch-cosine --linear-cka --lds --cds --rmsc
```

Outputs JSON to stdout.

## Design

- **Inputs only.** Metrics never load images, run encoders, or know about layers. The user produces two feature tensors and hands them in.
- **Dimensionality-agnostic.** Spatial metrics consume `coords: [N, S]`. 2D images, 3D volumes, and irregular layouts use the same code path.
- **Optional everything else.** Encoder wrappers are a separate, opt-in module — nothing in `metrics/` imports `transformers`.

See [`PLAN.md`](PLAN.md) for the implementation roadmap and [`REPORT.md`](REPORT.md) for what's currently shipped.

## License

Apache-2.0. The library is permissively licensed; some encoder weights downloaded by the optional wrappers are not (e.g. 3DINO is CC-BY-NC-ND). Each wrapper documents its underlying license.
