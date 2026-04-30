# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

v0.1 is in. Six Tier-1 metrics implemented with 27 passing tests, public dispatcher (`calculate_alignment_metrics`), CLI, and four optional encoder wrappers (DINOv2, RAD-DINO via the same class, BiomedCLIP, 3DINO). See `REPORT.md` for the full status and `PLAN.md` for the roadmap.

## Commands

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]" "torch>=2.0" "numpy>=1.24"
pytest -q                   # run the test suite (27 tests, ~0.05s)
ruff check .                # lint
torch-align-med --help      # CLI smoke
```

Optional encoder backends are pulled in lazily — install with `pip install ".[encoders-2d]"` etc., or just run the metrics on already-extracted features and skip them entirely.

## Project goal

`torch-align-med` is a PyTorch library of **semantic-alignment metrics between latent spaces**. It is the alignment-evaluation analog of [`torch-fidelity`](https://github.com/toshas/torch-fidelity) (which measures generative fidelity, e.g. FID/KID/IS): same ergonomic feel, different problem.

The motivating context is the "Representations for Generation" (R4G) line of work — REPA, VA-VAE, REPA-E, iREPA, RAE, etc. — where diffusion / VAE latents are aligned to frozen vision foundation model features (DINOv2, CLIP, …). Reference write-up: https://kdidi.netlify.app/blog/ml/2025-12-31-r4g/. Reading these papers requires a small zoo of alignment metrics that are currently re-implemented per-paper; this package consolidates them.

## API shape (intended)

Mirror `torch-fidelity`'s surface so users coming from FID-style evaluation feel at home:

- A single Python entry point along the lines of `calculate_alignment_metrics(input1=..., input2=..., **flags)` returning a `dict[str, float]`.
- Inputs are two sources of features/latents (tensors, dataloaders, or wrapped models that emit them) — typically one generative-side latent and one frozen-encoder feature.
- A `align` (or similar) CLI that mirrors the `fidelity` CLI: `--metric-name` flags toggle individual metrics.
- Per-metric modules under a single package, each implementing a small interface (compute features → compute metric). Share feature extraction across metrics and cache intermediates, the way `torch-fidelity` does — alignment metrics on the same pair of inputs should not re-encode.

## Candidate metrics to support

From the R4G literature; treat as the initial backlog, not a fixed list:

- **Cosine alignment** — patch-wise / token-wise cosine similarity (REPA-style baseline).
- **Local Distance Similarity** and **Short-Range Spatial Similarity** — iREPA's spatial-structure metrics; reported to correlate strongly (|r| > 0.85) with downstream generation FID.
- **Cosine Distance Similarity** and **Relative Mean Spatial Contrast** — also from iREPA's spatial-structure family.
- **SE-CKNNA** — semantic-structure-preservation metric across noise levels.
- **Pairwise relational alignment** (VA-VAE's `L_mdms`) — preserves pairwise distances between latents under the encoder map.
- **Self-similarity structure** — patch-token self-similarity matrices compared between the two spaces.

When adding a new metric, cite the source paper in the module docstring and note what input shapes it expects (e.g. `[B, N_patches, D]` token features vs. `[B, D]` pooled).

## Conventions to establish early (before code grows)

- **Pick a packaging tool and stick with it.** `.gitignore` already has entries for `pyproject.toml`-based tools (uv, poetry, pdm, pixi). Choose one and add `pyproject.toml` before the second metric lands.
- **Frozen encoders are dependencies, not vendored code.** Load DINOv2/CLIP via `torch.hub` or `transformers`; do not copy model code in.
- **Determinism matters for metrics.** Seed everything that touches sampling, and document any metric whose value depends on batch order.

## Repo layout (load-bearing files only)

- `src/torch_align_med/api.py` — `calculate_alignment_metrics`, the public dispatcher.
- `src/torch_align_med/base.py` — `PairwiseMetric` and `EncoderDiagnostic` ABCs. New metrics inherit from one of these.
- `src/torch_align_med/_grids.py` — `make_grid_coords`, Manhattan distances, cosine self-similarity. Shared by all spatial metrics.
- `src/torch_align_med/metrics/{pairwise,diagnostic}/*.py` — one metric per file, registered in `metrics/__init__.py`.
- `src/torch_align_med/encoders/*.py` — optional, lazy-import. Never imported by metric code.
- `tests/conftest.py` — synthetic feature fixtures (`smooth_tokens_2d` is the one that actually exercises spatial-structure metrics; random fixtures are sanity checks).

When adding a new metric: pick the family (pairwise / diagnostic), inherit from the ABC, register it in `metrics/__init__.py` *and* in `api.py`'s registry dict, and add property tests that assert closed-form values on synthetic inputs.
