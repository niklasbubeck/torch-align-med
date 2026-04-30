---
layout: default
title: torch-align-med
---

# torch-align-med

**Semantic-alignment metrics between latent spaces.**
The alignment-evaluation analog of [`torch-fidelity`](https://github.com/toshas/torch-fidelity), built for 2D natural images, 2D medical images, and 3D medical volumes from a single API.

[GitHub repository](https://github.com/AICONSlab/torch-align-med){: .btn} · [Implementation plan](plan.html){: .btn} · [Status report](report.html){: .btn}

---

## Why

Modern diffusion / VAE training increasingly leans on **representation alignment** — see REPA, VA-VAE, REPA-E, iREPA, RAE, and the wider [Representation-for-Generation literature](https://kdidi.netlify.app/blog/ml/2025-12-31-r4g/). Every paper measures alignment slightly differently, often re-implementing the same metric from scratch. `torch-align-med` collects six precise, reproducible alignment metrics behind one ergonomic API.

The package answers two questions:

1. **How aligned are these two latent spaces?** — pairwise metrics: `patch_cosine`, `linear_cka`, `l_mdms`.
2. **Does this encoder have the spatial structure that makes it a good alignment target?** — single-encoder diagnostics from iREPA: `lds`, `cds`, `rmsc`. The iREPA paper reports |r| > 0.85 between these and downstream FID across 27 encoders — far stronger than ImageNet-1K linear probing (|r| = 0.26).

## Quick start

```python
import torch
from torch_align_med import calculate_alignment_metrics, make_grid_coords

feats_a = torch.randn(8, 256, 1024)         # [B, N, D]
feats_b = torch.randn(8, 256, 1024)
coords  = make_grid_coords((16, 16))        # 2D grid; use (D, H, W) for 3D

scores = calculate_alignment_metrics(
    feats_a, feats_b,
    coords=coords,
    patch_cosine=True, linear_cka=True, l_mdms=True,
    lds=True, cds=True, rmsc=True,
)
```

`coords` is `[N, S]` — `S = 2` for images, `S = 3` for volumes. The same metric code path serves both.

## Reference encoders

Optional convenience wrappers around standard alignment-target encoders, lazy-imported so they never weigh on the base install.

| Modality | Wrapper | Hugging Face repo | License |
|---|---|---|---|
| 2D natural | `DinoV2Encoder` | `facebook/dinov2-with-registers-large` | Apache-2.0 |
| 2D medical (radiology) | `DinoV2Encoder` | `microsoft/rad-dino` | MIT |
| 2D medical (cross-modality) | `BiomedClipEncoder` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT |
| 3D medical (native) | `ThreeDinoEncoder` | `AICONSlab/3DINO-ViT` | CC-BY-NC-ND 4.0 |
| 3D medical (pseudo-3D) | `PlaneCycleEncoder` | `HINTLab/PlaneCycle` (lifts a frozen 2D DINOv3) | Apache-2.0 |

## Design principles

- **Features-only.** Core metrics never load images or run models — they consume two feature tensors. Convenience encoder wrappers live in a separate, optional module.
- **Dimensionality-agnostic.** A coordinate tensor `[N, S]` generalises to 2D, 3D, and irregular layouts with no metric-side branching.
- **Two metric families, one dispatcher.** Pairwise metrics emit one scalar; encoder diagnostics emit two (`{name}_input1`, `{name}_input2`).

## Install

```bash
pip install torch-align-med                              # core
pip install "torch-align-med[encoders-2d]"               # + DINOv2 wrappers
pip install "torch-align-med[encoders-medical]"          # + RAD-DINO, BiomedCLIP
pip install "torch-align-med[all]"                       # everything
```

## Documentation

- **[Status report](report.html)** — what ships in v0.1, with API examples and verification details.
- **[Implementation plan](plan.html)** — roadmap, metric backlog, and architectural decisions.
- **[GitHub README](https://github.com/AICONSlab/torch-align-med#readme)** — quick reference.

## Citing the underlying methods

If you use the metrics in published work, please cite the source papers:

- **REPA** — Yu et al., *Representation Alignment for Generation*, arXiv:2410.06940 (2024).
- **VA-VAE** — Yao et al., *Reconstruction vs. Generation*, arXiv:2501.01423 (2025).
- **iREPA** — Singh et al., *What matters for Representation Alignment*, arXiv:2512.10794 (2025).
- **CKA** — Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML 2019, arXiv:1905.00414.
