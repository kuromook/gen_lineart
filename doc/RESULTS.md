# Results Layout

Updated: 2026-07-29 JST

`results/` now keeps metrics, manifests, CSV/JSON outputs, and only a small set
of currently referenced montage images.

Large resolved image outputs were deleted during the 2026-07-25 cleanup.
Per-sample output images and old QC panels should be regenerated from scripts,
manifests, and checkpoints when needed.

Use `config/results_manifest.json` as the current lightweight index. It has
been pruned to paths that still exist after cleanup.

## Per-Source Folders (2026-07-29)

Raw-manuscript-source-specific koma panel-detection/tile-extraction outputs
(`ako5ver2`, `hamlabi`, `fitness`, `gakuen`, `housei`, `fighting`) live under
`results/<source>/` instead of as flat top-level files. Each folder holds only
the latest accepted panels manifest + overlays and the final materialize/
subregion/tile-extraction outputs; earlier dated/versioned iterations and
pre-koma legacy approaches (region matches, native_strict tiles, keep281/
varregion/strict88 experiments, etc.) were deleted as superseded — see
`doc/work_log.md` 2026-07-29 entry for the before/after counts. New per-source
pipeline runs should write directly into `results/<source>/` going forward.

Everything else (cross-source model/architecture-comparison experiments —
`halo_*`, `haze_uncertainty_*`, `router_*`, `line_refiner_*`,
`fixed_output_metrics_*`, `pair_dataset_integrity_*` not tied to one source,
`combined_20260726_*`, `dataset_gate_*`, etc.) was left in place at the
top level; it isn't organized "per source" and wasn't reviewed for deletion
in this pass.

## `results/lessons/` (2026-08-04)

Curated, small folder for outputs that carry a specific documented lesson
forward -- distinct from `results/archive/` (old/historical, kept for
audit only, not actively cited) and from the general top-level clutter
(regenerable per-run outputs). Something belongs in `lessons/` only if a
current doc (`doc/architecture_decisions.md`, `doc/work_log.md`) explicitly
cites it as evidence for a stated finding, not just because it was a
notable run. Named `lessons/` rather than `artifacts/` deliberately --
everything under `results/` is technically an "artifact," so that name
would not distinguish this folder's curated intent.

Current contents: the direct-regression (`unet`/`unet_skip0`) epoch-
trajectory lineage that grounds the ongoing stroke-continuity work --
3-epoch smoke test, 100-epoch/200-epoch/dense-28-epoch comparisons, the
epoch-trajectory montages, and the rough-fidelity-vs-binarization crop
comparison. See `doc/architecture_decisions.md`'s "単段直接回帰" section
for what each one demonstrates. When a doc reference to a `results/`-root
path is moved into `lessons/`, update the citing doc's path in the same
edit -- do not leave dangling references.

## Current Kept Images

Remaining top-level image files under `results/`:

- `results/compare_badrough_lucy_thin_threshold_e3.png`
- `results/compare_clean_baselines_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `results/compare_linefield_initial_e2.png`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`

Now under its source folder:

- `results/hamlabi/hamlabi_filtered398_unet480_epoch040_compare.png`

Current ako5ver2 review image is outside `results/`:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/valid_mask_qc.png`

## What To Keep

Keep:

- active montage images explicitly referenced by current docs
- metrics CSVs
- review CSVs
- candidate manifests
- integrity audit summaries/findings
- lightweight JSON route/manifest outputs

Delete after resolution:

- per-sample inference image directories
- old smoke-test images
- old QC sheets once review decisions are recorded
- archived/leak-era images that are not needed for active docs

## Cleanup Policy

1. Search active docs for direct `results/*.png|jpg|jpeg` references.
2. Protect only those current references.
3. Delete unreferenced `results/` image files.
4. Keep non-image metrics/manifests unless a separate cleanup asks to remove
   them.
5. Update `config/results_manifest.json` after deletion so it contains only
   existing paths.
6. Record major cleanup counts in `doc/work_log.md`.
