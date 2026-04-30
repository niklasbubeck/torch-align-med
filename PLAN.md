# torch-align-med — Implementation Plan

This is the working plan for v0. It is meant to be edited as choices land or change.

## Goal

Ship a PyTorch library that measures **semantic alignment between two latent spaces**, in the ergonomic style of `torch-fidelity`. Motivated by the R4G literature (REPA, VA-VAE, REPA-E, iREPA, …) where diffusion / VAE latents are aligned to frozen vision-foundation-model features. Reference: https://kdidi.netlify.app/blog/ml/2025-12-31-r4g/.

## API contract

**Core metrics are features-in only.** They never load images, never run encoders, never know about layers. The user produces two feature tensors and hands them in. Everything model-related is optional and separate.

```python
torch_align_med.calculate_alignment_metrics(
    input1: Tensor,    # [B, N, D] tokens, or [B, D] pooled
    input2: Tensor,
    coords: Tensor | None = None,   # [N, S] spatial coordinates; S=2 for 2D, 3 for 3D
                                    #   required only by spatial-structure metrics
    repa_cosine=True, cka=True, lds=True, cds=True, rmsc=True, l_mdms=True,
    ...,
) -> dict[str, float]
```

Two metric families:

- `PairwiseMetric.compute(feats_a, feats_b) -> float` — single scalar over the pair.
- `EncoderDiagnostic.compute(feats, coords=None) -> float` — single-encoder property; runner calls on each side and emits `{name}_input1`, `{name}_input2`.

**Dimensionality-agnostic by design.** Metrics that need a spatial grid (`LDS`, `CDS`, `RMSC`, `SRSS`) take `coords: [N, S]`. With `S=2` you get 2D images, `S=3` covers 3D volumes (and CT/MRI feature maps from VoCo). No 2D-only assumptions in metric code.

## Metric set

**Tier 1 — MVP:**

| Metric | Family | Formula source |
|---|---|---|
| Patch-token cosine alignment (REPA) | Pairwise | REPA paper, Yu et al. 2024 |
| `L_mdms` (pairwise distance-matrix) | Pairwise | VA-VAE, Yao et al. 2025 |
| Linear CKA | Pairwise | Kornblith et al. 2019 |
| LDS — Local vs. Distant Similarity | Diagnostic | iREPA, Singh et al. 2025 |
| CDS — Correlation Decay Slope | Diagnostic | iREPA |
| RMSC — RMS Spatial Contrast | Diagnostic | iREPA |

**Tier 2 — fast-follow:**

- CKNNA / mutual-kNN alignment (sparse CKA variant)
- SRSS — Semantic-Region Self-Similarity (gates SAM2 dependency behind `extras_require[srss]`)
- SE-CKNNA (after reading Bi et al. 2510.18457)
- `L_mcos` standalone (lower priority — overlaps with patch-token cosine for evaluation)

**Skip:** Dispersive Loss (training regularizer, not a metric); ImageNet-1K linear probing (the R4G literature shows it correlates poorly with FID, |r|=0.26).

## Reference encoders (optional convenience layer)

Lives in `torch_align_med.encoders.*`, gated behind `extras_require`. Core metrics never depend on these.

| Modality | Wrapper | HF repo | License | Style |
|---|---|---|---|---|
| 2D natural | DINOv2-with-registers ViT-L/14 | `facebook/dinov2-with-registers-large` | Apache-2.0, ungated | DINOv2 (rich patches) |
| 2D medical | RAD-DINO ViT-B/14 | `microsoft/rad-dino` | MIT, ungated | DINOv2 (rich patches) |
| 2D medical | BiomedCLIP-PubMedBERT ViT-B/16 | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT, ungated | CLIP (global) |
| 3D medical | 3DINO-ViT (AICONSlab) | `AICONSlab/3DINO-ViT` | CC-BY-NC-ND 4.0 | DINOv2 (rich patches, 3D) |

Why this set:

- **DINOv2-with-registers** — canonical R4G alignment target; Apache-2.0 makes it CI-friendly.
- **RAD-DINO** — only DINOv2-style medical encoder that is both MIT-licensed and ungated. Radiology-specific (chest X-ray).
- **BiomedCLIP** — *complements* RAD-DINO rather than substituting for it. CLIP-style contrastive on PMC-15M (15M biomedical figures across modalities — radiology, histology, microscopy). Global features dominate, so it's a strong target for `REPA-cosine` on the pooled embedding and for CKA, but a weak target for `LDS`/`CDS`/`RMSC` (which exist to measure spatial structure). Document this in the wrapper docstring so users don't run spatial-structure metrics on its CLS-only embedding by accident.
- **3DINO-ViT** — DINOv2 extended to 3D, ~100k 3D scans across 10+ organs (CT + MRI). Produces patch tokens analogous to 2D DINOv2, so the *same* metric code works with a 3D `coords` grid. Two caveats: license is CC-BY-NC-ND 4.0 (research-only — flag in README), and the upstream repo pins Python 3.9. The wrapper should not force the whole package to 3.9 — load the state_dict directly into a minimal 3D-ViT module if feasible, or lazy-import the upstream package and let users opt into a 3.9 env.

Pathology (UNI2-h, Virchow2) and DINOv3 are obvious follow-ups but gated; ship them in a second pass with the auth flow documented.

### Pseudo-3D via PlaneCycle ✅ (shipped in v0.1)

[PlaneCycle](https://github.com/HINTLab/PlaneCycle) (Apache-2.0) is a *training-free* converter that rewrites a frozen DINOv3 ViT to process volumetric input by cycling its transformer blocks across the three orthogonal planes (HW, DW, DH). The result is a 3D-aware feature extractor with no learned 3D parameters.

**Implemented as `PlaneCycleEncoder`** in `src/torch_align_med/encoders/planecycle.py`. Six tests cover the shape contract, error paths, and an end-to-end self-alignment check via `calculate_alignment_metrics`.

**Architectural correction (from the source audit):** the upstream converter is *not* a generic 2D-encoder wrapper. It mutates a backbone in-place by replacing its `.blocks` `ModuleList` with `PlaneCycleBlock` wrappers, and the block forward signature `blk(x, shape)` is tightly coupled to the upstream DINOv3 ViT (constructed with `block_type="PlaneCycle"`). The wrapper therefore takes a *pre-loaded converted backbone* — loading the DINOv3 ViT and applying `PlaneCycleConverter` is the user's responsibility, documented inline. We do not vendor the upstream model code.

**Output contract:** `model.get_intermediate_layers(volume, n=1, reshape=True)` returns `[B, C_feat, D, h, w]`. The wrapper flattens this to `[B, D·h·w, C_feat]` and tracks the 3D grid shape so callers can build matching `coords: [D·h·w, 3]` via `make_grid_coords(enc.grid_shape())`.

## Packaging & repo layout

- **Tooling:** `uv` + `pyproject.toml`. Single source of truth for dependencies.
- **Layout:** `src/torch_align_med/` (src layout — keeps tests honest).
- **Linting:** `ruff` (format + lint), no separate black.
- **Tests:** `pytest`. Unit tests on synthetic features with closed-form expected values; smoke tests that load a tiny encoder once and run end-to-end.
- **Extras:** `[encoders-2d]`, `[encoders-medical]`, `[encoders-3d]`, `[srss]` (SAM2). Core install pulls only `torch` + `numpy`.

```
src/torch_align_med/
  __init__.py                # public re-exports
  api.py                     # calculate_alignment_metrics
  base.py                    # PairwiseMetric, EncoderDiagnostic ABCs
  metrics/
    pairwise/
      cosine.py              # REPA-style patch cosine
      cka.py                 # Linear CKA
      mdms.py                # VA-VAE L_mdms
    diagnostic/
      lds.py                 # Local vs. Distant Similarity
      cds.py                 # Correlation Decay Slope
      rmsc.py                # RMS Spatial Contrast
  encoders/                  # optional, lives behind extras_require
    dinov2.py
    raddino.py
    voco.py
  cli.py                     # `torch-align-med` mirrors `fidelity`
tests/
  unit/
  fixtures/                  # synthetic features with known answers
```

## Implementation phases

**Phase 0 — bootstrap (single PR):**
- `pyproject.toml` (uv), `src/` layout, ruff + pytest configured, GitHub Actions skeleton.
- `PairwiseMetric` and `EncoderDiagnostic` ABCs.
- A `synthetic_features` fixture (random tokens + grid coords for 2D and 3D).

**Phase 1 — Tier-1 metrics:**
- Implement the six Tier-1 metrics.
- Each metric: implementation + unit test on synthetic data with a closed-form known answer + a property test (e.g. CKA(X, X) == 1, RMSC(constant_features) == 0).
- **Verify each formula against the source PDF before committing tests** — the iREPA formulas in particular came via web summaries and must be cross-checked.

**Phase 2 — public API + first encoder:**
- `calculate_alignment_metrics(...)` — the dispatcher.
- A `torch-align-med` CLI mirroring `fidelity`.
- DINOv2-with-registers wrapper (smallest dependency footprint).
- README usage example: extract DINOv2 features for two image folders → compute all Tier-1 metrics.

**Phase 3 — Tier-2 metrics:** CKNNA/mutual-kNN, SRSS (SAM2 extra), SE-CKNNA, `L_mcos`.

**Phase 4 — medical encoders:** RAD-DINO wrapper; BiomedCLIP wrapper (pooled-embedding only — document spatial-metrics caveat); 3DINO-ViT wrapper. The 3DINO wrapper is the non-trivial one — confirm token-output shape from `notebooks/basic_model_use.ipynb`, decide on direct state-dict loading vs. lazy import, and verify the volume-coords convention used by their 3D random-resized-crop pipeline before locking the API.

**Phase 5 — reproduction:** notebook reproducing iREPA's correlation table — LDS/CDS/RMSC computed on a pool of encoders, plotted against published FID numbers from REPA/REPA-E. This is the most credible validation we can offer.

## Constraints / decisions locked in

1. **Inputs only.** Metrics never accept models or images. Convenience extractors are a separate, optional submodule.
2. **Dimensionality-agnostic.** Spatial metrics use `coords`, not assumed 2D grids. 3D works on day one.
3. **Verify formulas against papers** before locking unit tests. Web summaries (including the ones used to draft this plan) drift from the published math.
4. **Permissive-license-first** for shipped encoder wrappers. Gated/non-commercial models come later, behind clear documentation.
5. **No comment noise in code.** Module docstrings cite the source paper and the formula; no inline narration of what the math does.

## Open items

- The exact CKNNA / mutual-kNN formulation needs verification against the Platonic Hypothesis paper PDF.
- SE-CKNNA needs the Bi et al. 2510.18457 PDF read — its formula is not in any web summary I've seen.
- 3DINO-ViT's exact patch-token output shape and CLS/register-token convention needs confirmation from `notebooks/basic_model_use.ipynb` before writing the wrapper.
- 3DINO's Python 3.9 pin: decide between (a) writing a minimal independent 3D-ViT loader that consumes the published state_dict, or (b) lazy-import + document a separate 3.9 env. (a) is preferable but only viable if the architecture is straightforward.
