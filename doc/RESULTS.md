# Results Layout

Updated: 2026-07-25 JST

`results/` now keeps metrics, manifests, CSV/JSON outputs, and only a small set
of currently referenced montage images.

Large resolved image outputs were deleted during the 2026-07-25 cleanup.
Per-sample output images and old QC panels should be regenerated from scripts,
manifests, and checkpoints when needed.

Use `config/results_manifest.json` as the current lightweight index. It has
been pruned to paths that still exist after cleanup.

## Current Kept Images

Remaining image files under `results/`:

- `results/compare_badrough_lucy_thin_threshold_e3.png`
- `results/compare_clean_baselines_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `results/compare_linefield_initial_e2.png`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/hamlabi_filtered398_unet480_epoch040_compare.png`

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
