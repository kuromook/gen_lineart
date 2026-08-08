# Work Log - Clean Rebuild

Updated: 2026-07-25 JST

This active work log is intentionally compact. Detailed chronological history
from 2026-07-18 through the 2026-07-25 compaction is archived in:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

Read current state first:

- `doc/CURRENT.md`

## Current Pointers

Use focused docs for reusable knowledge:

- extraction rules and gates: `doc/EXTRACTION_RULES.md`
- dataset status and current manifests: `doc/dataset_status.md`
- dataset-specific raw extraction knowledge: `doc/raw_dataset_extraction_knowledge.md`
- raw zip storage: `doc/raw_dataset_storage_policy.md`
- region extraction policy: `doc/region_dataset_extraction_policy.md`
- region materialization policy: `doc/region_materialization_policy.md`
- region search loop: `doc/region_search_loop.md`
- model-family conclusions: `doc/model_results_summary.md`
- model direction survey: `doc/model_directions.md`
- result layout: `doc/RESULTS.md`
- worktree policy: `doc/worktree_policy.md`
- documentation maintenance: `doc/documentation_maintenance_policy.md`

## Current Active Goal

Build a clean, non-leaky rough-to-line training/evaluation path by improving
raw manuscript pair extraction, review, masking, and dataset-specific filtering.

Do not use leak-era `shape1` metrics as adoption targets. Clean eval metrics
and montage review are supporting references; visual quality and data integrity
remain the adoption gates.

## Compact History

### Clean Baseline Reset

Pre-clean-eval and leak-era assumptions were reset on 2026-07-18. Old `shape1`
scores are historical only.

Clean BCE references:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 |
| `shape1_clean_split_bce_milddup800_ft10_lr1e5` | 0.3010 | 7.079 | 1.333 |

Interpretation:

- clean BCE is a baseline reference, not a finished line-art direction
- milddup800 is only a small clean-baseline variant

Detailed history:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

### Model / Halo Survey Summary

Current model conclusions are in:

- `doc/model_results_summary.md`

Current important candidates:

- `lucy_mild_aux_msgan`: safer balanced Lucy candidate
- `lucy_thin_aux_msgan`: higher-recall candidate requiring artifact scrutiny
- `dog_aux_msgan`: controlled white/hint candidate
- `flowdog_aux_msgan`: high-recall / high-ink expert candidate
- cleanup + MSGAN/FM: promising but artifact-prone
- router/MoE oracle: useful upper-bound probe, not a deployed router
- initial line-field: runs end to end but overproduces ink

Detailed run-by-run model history:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

### Bad-Rough / Lucy-Thin Summary

The bad-rough cleanup and Lucy-thin threshold work is summarized in:

- `doc/badrough_lucy_thin_threshold_notes.md`

Current interpretation:

- bad-rough exclusion changed model behavior
- Lucy-thin / threshold-aware loss is a useful reference result
- thresholded output remains too grainy for production
- next useful work is preprocessing and cleaner region-matched data expansion

### hamlabi Summary

Current useful hamlabi dataset state is in:

- `doc/dataset_status.md`

Current useful hamlabi manifest:

- `dataset/regions_hamlabi_loop_auto_review_postalign12_masked_line_conservative_filtered398/manifest.csv`

Rows:

- 398

Latest hamlabi model review:

- `hamlabi_filtered398_unet480_e40` completed and produced review outputs
- the extraction workflow is useful
- plain U-Net e40 is technically valid but not a finished line-art expert
- hamlabi remains complex because roughs are dense, sketchy, and often
  black-fill-heavy

Key artifacts:

- `checkpoints/hamlabi_filtered398_unet480_e40/best.pth`
- `checkpoints/hamlabi_filtered398_unet480_e40/epoch040.pth`
- `results/hamlabi_filtered398_unet480_epoch040_compare.png`
- per-sample output images under `results/` were later removed during results
  image cleanup; the montage and checkpoints are the durable references

Detailed hamlabi exploration history:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

## Current Active Dataset Work

### Raw Dataset Zip Storage

Raw dataset archives are stored under:

- `dataset/raw_zips/`

Current archives:

- `dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset/raw_zips/dataset_hamlabi.zip`

Root symlinks exist only for compatibility:

- `dataset_ako5ver2.zip -> dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset_ako5.zip -> dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset_hamlabi.zip -> dataset/raw_zips/dataset_hamlabi.zip`

Policy:

- `doc/raw_dataset_storage_policy.md`

### ako5ver2 Fixed-Tile Baseline

The strict 88-pair 480px extraction is retained as a fixed-tile baseline only.

Train list:

- `dataset/pairs_480/valid_train_ako5ver2_region_20260725_strict.txt`

Line-dir override:

- `ako5r=dataset/pairs_480/train/line_ako5ver2_region_20260725_strict`

Integrity audit:

- findings: 0
- summary: `results/pair_dataset_integrity_summary_ako5ver2_region_20260725_strict.csv`
- findings CSV: `results/pair_dataset_integrity_findings_ako5ver2_region_20260725_strict.csv`

Status:

- do not treat this as the preferred ako5 workflow
- prefer the variable-region manifest below for current review/training design

### ako5ver2 Variable-Region Extraction

ako5ver2 has been processed through the hamlabi-style variable-region workflow:

- line-anchored variable-aspect parent proposals
- rough-side translation/scale search
- recursive child search inside large parent regions
- variable-aspect materialization
- conservative post-align
- conservative valid-mask generation

Current review target:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/`

Current QC:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/valid_mask_qc.png`

Current manifest before user review flags:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest.csv`

Rows:

- 291

Current filtered manifest after user review flags:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv`

Rows:

- 281

Review files:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/removed_user_mismatch.csv`
- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/held_user_mask_insufficient_umbrella.csv`
- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/flagged_user_review_20260725.csv`
- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/user_review_20260725_summary.txt`

### ako5ver2 User Review Flags

Excluded as rough/line mismatches:

- review indices: 42, 58, 59, 81, 90, 98, 112

Held out as mask-insufficient umbrella / layer-difference cases:

- review indices: 69, 79, 86

Reason for umbrella hold:

- rough contains umbrella but line art does not
- likely added later by 3D, separate layer, or compositing
- current conservative valid mask is insufficient to train these rows safely

Dataset-specific knowledge:

- `doc/raw_dataset_extraction_knowledge.md`

## Documentation Maintenance

The 2026-07-25 compaction was accepted by the user with this policy:

- archive clean-baseline history
- archive model/halo survey history
- archive Lucy-thin detailed history
- archive hamlabi exploratory workflow details
- keep 2026-07-25 ako5ver2 current work active

Compaction review document:

- `doc/work_log_compaction_review_20260725.md`

Archived detailed active-log source:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

## Results Image Cleanup

Applied cleanup for resolved `results/` images on 2026-07-25.

Policy used:

- delete image files under `results/` that are not referenced by active,
  non-archive docs
- keep CSV/JSON/metrics and current referenced montage images
- update `config/results_manifest.json` to drop deleted paths

Result:

- deleted images: 47,741
- reclaimed: about 3.8 GB
- remaining `results/` size: about 60 MB
- remaining image files: 6

Remaining images:

- `results/compare_badrough_lucy_thin_threshold_e3.png`
- `results/compare_clean_baselines_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `results/compare_linefield_initial_e2.png`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/hamlabi_filtered398_unet480_epoch040_compare.png`

## Next Actions

## 2026-07-25 ako5ver2 Main Masked Run

After user review, ran the first variable-region masked ako5ver2 training pass
from the 281-row kept manifest.

Code changes:

- `scripts/train_i2i_survey.py`
  - added `--region-rough-key` and `--region-line-key`
  - used to force masked source/target columns for masked manifests
- `tools/compare/make_region_manifest_compare.py`
  - added `--rough-key` and `--line-key`
  - keeps comparison montage aligned with the same manifest columns used during
    training
- `experiments/run_ako5ver2_varregion_keep281_masked_unet480_e20.sh`
  - trains masked keep281 U-Net at 480px, square-pad, 20 epochs
  - generates a 12-sample montage from `best.pth`

Verification:

- `bash -n experiments/run_ako5ver2_varregion_keep281_masked_unet480_e20.sh`
- `venv/bin/python -m py_compile scripts/train_i2i_survey.py tools/compare/make_region_manifest_compare.py lineart/region_dataset.py`
- manifest loader check:
  - rows: 281
  - first samples: rough/target/mask tensors all `(1, 480, 480)`

Training result:

- checkpoint dir:
  - `checkpoints/ako5ver2_varregion_keep281_masked_unet480_e20/`
- log:
  - `logs/train_ako5ver2_varregion_keep281_masked_unet480_e20.log`
- montage:
  - `results/ako5ver2_varregion_keep281_masked_unet480_e20_compare.png`
- individual outputs:
  - `results/ako5ver2_varregion_keep281_masked_unet480_e20_outputs/`
- final epoch:
  - `Epoch 020/20: G=0.2632 D=0.0000`

Visual interpretation:

- The run completed technically on CUDA.
- From-scratch U-Net on keep281 currently looks too density-map / black-fill
  oriented, not yet like a useful line-art expert.
- This suggests the next main attempt should use a clean-line checkpoint as a
  warmstart rather than treating keep281 from-scratch as the production path.

Follow-up warmstart:

- runner:
  - `experiments/run_ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10.sh`
- base checkpoint:
  - `checkpoints/shape1_clean_split_bce/best.pth`
- checkpoint dir:
  - `checkpoints/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10/`
- montage:
  - `results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_compare.png`
- threshold montage:
  - `results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_threshold52_compare.png`
- final epoch:
  - `Epoch 010/10: G=0.2107 D=0.0000`

Warmstart interpretation:

- Strict resume worked and loss improved quickly.
- Visual output is still too soft / density-map oriented on the variable-region
  keep281 masked manifest.
- Threshold52 mainly turns the density map into black islands rather than clean
  lines.

## 2026-07-25 ako5ver2 strict88 Control

Ran the fixed 480px strict88 pair-list extraction after the main masked runs, as
a controlled follow-up.

Runner:

- `experiments/run_ako5ver2_strict88_480_warm_clean_bce_e10.sh`

Source:

- file list:
  - `dataset/pairs_480/valid_train_ako5ver2_region_20260725_strict.txt`
- rough dir:
  - `dataset/pairs_480/train/rough`
- line dir:
  - `dataset/pairs_480/train/line_ako5ver2_region_20260725_strict`

Training result:

- checkpoint dir:
  - `checkpoints/ako5ver2_strict88_480_warm_clean_bce_e10/`
- output dir:
  - `results/ako5ver2_strict88_480_warm_clean_bce_e10_outputs/`
- threshold montages:
  - `results/ako5ver2_strict88_480_warm_clean_bce_e10_threshold52_compare.png`
  - `results/ako5ver2_strict88_480_warm_clean_bce_e10_threshold_sweep_compare.png`
- final epoch:
  - `Epoch 010/10: G=0.1810 D=0.0000`

Strict88 interpretation:

- Much more line-like than the variable-region keep281 runs.
- Still has strong thickening, black-fill growth, and crop/scale dependency.
- Threshold 52/54/58/62 changes little, meaning the model is already confident
  where it inks.
- Useful as a compatibility/control signal, but not enough by itself to replace
  the variable-region workflow.

## 2026-07-25 keep281-derived 480px Tiles

Created a new 480px tiler for reviewed region manifests:

- `tools/pair_extraction/tile_region_manifest_480.py`

Purpose:

- derive fixed 480px local tile pairs from the reviewed keep281 masked manifest
- keep the review/mask provenance while testing whether smaller crops reduce
  the keep281 density-map / black-blob failure mode

Default source:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv`

First top300 extraction:

- CSV:
  - `results/ako5ver2_keep281_tiles_480_top300.csv`
- list:
  - `dataset/pairs_480/valid_train_ako5ver2_keep281_tiles_top300_20260725.txt`
- line dir:
  - `dataset/pairs_480/train/line_ako5ver2_keep281_tiles_top300_20260725`
- QC:
  - `results/ako5ver2_keep281_tiles_480_top300_qc.png`
  - `results/ako5ver2_keep281_tiles_480_top300_qc_tail.png`
- extraction result:
  - raw tiles: 1694
  - accepted: 300
- integrity audit:
  - findings: 0
- training:
  - `experiments/run_ako5ver2_keep281_tiles_top300_480_warm_clean_bce_e10.sh`
  - final epoch: `Epoch 010/10: G=0.3252 D=0.0000`
  - montage:
    - `results/ako5ver2_keep281_tiles_top300_480_warm_clean_bce_e10_threshold_sweep_compare.png`

Top300 interpretation:

- Smaller crops did not solve the problem by itself.
- Score ranking selected too many black-fill / dense panel regions, so output
  remained blob-like.

Line-focused top300 extraction:

- additional filter:
  - `--ink-max 0.08`
- CSV:
  - `results/ako5ver2_keep281_tiles_480_linefocus_top300.csv`
- list:
  - `dataset/pairs_480/valid_train_ako5ver2_keep281_tiles_linefocus_top300_20260725.txt`
- line dir:
  - `dataset/pairs_480/train/line_ako5ver2_keep281_tiles_linefocus_top300_20260725`
- current tiles QC:
  - `results/ako5ver2_keep281_tiles_480_linefocus_top300_qc.png`
  - `results/ako5ver2_keep281_tiles_480_linefocus_top300_qc_tail.png`
- extraction result:
  - raw tiles: 1199
  - accepted: 300
  - line ink max: below 0.08
- integrity audit:
  - findings: 0
- training:
  - `experiments/run_ako5ver2_keep281_tiles_linefocus_top300_480_warm_clean_bce_e10.sh`
  - final epoch: `Epoch 010/10: G=0.2477 D=0.0000`
  - montage:
    - `results/ako5ver2_keep281_tiles_linefocus_top300_480_warm_clean_bce_e10_threshold_sweep_compare.png`

Line-focused interpretation:

- This is the current keep281-derived tile set.
- It is a true 480x480 crop from the 768x768 masked region images, not a simple
  768-to-480 resize.
- It is better filtered than the first top300 and avoids much of the black-fill
  mass, but the trained output still behaves like a sparse density/ink-region
  predictor more than a clean line-art model.
- The likely issue is that many 480 crops still contain panel-scale composition
  or black-fill-heavy targets, so the model can still learn blob placement.
- Keep smaller-source-crop-then-upscale as a separate future experiment; do not
  conflate it with the current 480-from-768 crop test.
- Next filtering should explicitly prefer close rough/line edge correspondence
  and local line strokes, and should first try black-fill exclusion as a clean
  cut for diagnosis.

Black-fill-excluded top300 extraction:

- additional filters:
  - `--ink-max 0.08`
  - `--max-black-component-ratio 0.025`
  - `--max-thick-ink-ratio 0.015`
- CSV:
  - `results/ako5ver2_keep281_tiles_480_nobeta_top300.csv`
- list:
  - `dataset/pairs_480/valid_train_ako5ver2_keep281_tiles_nobeta_top300_20260725.txt`
- line dir:
  - `dataset/pairs_480/train/line_ako5ver2_keep281_tiles_nobeta_top300_20260725`
- QC:
  - `results/ako5ver2_keep281_tiles_480_nobeta_top300_qc.png`
  - `results/ako5ver2_keep281_tiles_480_nobeta_top300_qc_tail.png`
- extraction result:
  - raw tiles: 765
  - accepted: 300
  - largest black component max: below 0.025
  - thick ink max: below 0.015
- integrity audit:
  - findings: 0
- training:
  - `experiments/run_ako5ver2_keep281_tiles_nobeta_top300_480_warm_clean_bce_e10.sh`
  - final epoch: `Epoch 010/10: G=0.2038 D=0.0000`
  - montage:
    - `results/ako5ver2_keep281_tiles_nobeta_top300_480_warm_clean_bce_e10_threshold_sweep_compare.png`

Black-fill-excluded interpretation:

- Loss improved substantially versus the earlier keep281-derived tile runs.
- Visual output still does not become clean line art; it is more sparse, but
  remains an ink/edge-fragment predictor.
- This confirms black-fill mass was one blob driver, but not the only issue.
- The next 480-from-768 filter should reduce panel-scale composition and favor
  tiles where rough and line edge structures are locally close and stroke-like.

## 2026-07-25 Strict Stroke-Scale Tile Filter

Built the stricter keep281-derived tile subset requested by the previous next
action. The measurement pass changed the design premise, so record both.

Code changes:

- `tools/pair_extraction/tile_region_manifest_480.py`
  - region-level source-scale gate: `--min-src-per-out`, `--max-src-per-out`
  - region-level review gates: `--exclude-feature-tags`, `--min-align-f1`,
    `--max-unsupported-line-edge-ratio`
  - tight-tolerance correspondence: `--strict-close-px` with `--min-strict-f1`,
    `--min-strict-line-recall`, `--min-strict-rough-precision`
  - stroke-likeness metrics: `line_width_p50` / `line_width_p95` from the ink
    distance transform, `long_line_ratio` from Hough segments, `soft_ink_ratio`
    as the gray share of drawn area
  - `--score-mode strict` alongside the preserved `legacy` score
  - all new gates default to off, so previous runs reproduce

Measurement pass:

- command: permissive gates, `--duplicate-overlap 1.01 --max-per-region 0`
- output: `results/ako5ver2_keep281_tiles_480_measure_all.csv` (2,389 tiles)
- scale diagnostic: `results/ako5ver2_keep281_scale_band_diagnostic.png`

Key finding, which inverted the original plan:

- Strict edge F1 *increases* with downscaling (0.169 below `src_per_out` 0.8,
  0.511 above 6), because a shrunken page turns rough and line into dense edge
  mush that matches everywhere.
- Ranking tiles by edge correspondence therefore promotes exactly the
  page-scale composition crops the next action wanted to remove.
- Visual check showed the low end is equally bad: `src_per_out` below about 1.0
  gives blurred upscaled rough against a near-empty line target.
- Usable band is roughly `src_per_out` 1.2 to 3.5, so scale needs a band gate
  rather than a cap, and must be gated rather than scored.

Details recorded in `doc/raw_dataset_extraction_knowledge.md` and the scale-band
rules in `doc/region_dataset_extraction_policy.md`.

Strict extraction result (dry-run, not saved):

- filters:
  - `--min-src-per-out 1.2 --max-src-per-out 4.0`
  - `--ink-min 0.012 --ink-max 0.08`
  - `--max-black-component-ratio 0.025 --max-thick-ink-ratio 0.015`
  - `--max-soft-ink-ratio 0.45 --max-line-width-p50 6.0`
  - `--min-strict-line-recall 0.40 --min-strict-rough-precision 0.15`
  - `--max-long-line-ratio 0.25 --min-support 0.90`
  - `--score-mode strict --duplicate-overlap 0.50 --max-per-region 4`
- CSV: `results/ako5ver2_keep281_tiles_480_strokescale.csv`
- QC:
  - `results/ako5ver2_keep281_tiles_480_strokescale_qc.png`
  - `results/ako5ver2_keep281_tiles_480_strokescale_qc_tail.png`
  - `results/ako5ver2_keep281_tiles_480_strokescale_qc_sample.png`
  - `results/ako5ver2_keep281_tiles_480_strokescale_review.png`
- counts:
  - regions used: 117 of 281
  - region rejects: `src_per_out_low` 114, `src_per_out_high` 50
  - raw tiles: 164
  - accepted: 90 tiles from 45 regions

Visual interpretation:

- Top-ranked tiles are face / figure / drapery scale with genuine rough-to-line
  correspondence and crisp strokes.
- Panel composition and black-fill mass, which dominated the earlier top300 and
  linefocus sets, are gone.
- The tail (about the last 10 percent) is still weak and near-empty.
- 90 tiles is clean but small; do not expect a finished model from it alone.

Normalization bottleneck:

- The 768 px long-side normalization, not the filter, is the limiting factor.
- At native source scale the same 281 regions could yield about 16,700 candidate
  480 px tiles instead of about 466.
- 113 of 281 regions are downscaled by 2x or more.

## 2026-07-25 ako5ver2 Native Re-Materialization

User chose to re-materialize keep281 near native scale, following the tile
budget finding above (about 466 tiles at 768 px normalization versus about
16,700 at native resolution).

### Native Materializer

New tool: `tools/pair_extraction/materialize_region_manifest_native.py`

- re-crops each reviewed region from the source page at a fixed
  source-to-output ratio instead of a fixed normalized long side
- redoes alignment at output scale using the reviewed 768-space `align_dx` /
  `align_dy` only as a starting offset; a two-stage coarse-then-fine translation
  search refines it, using cached edge-distance fields so a wide search range
  stays affordable at native resolution
- `--min-output-size` drops regions too small to fill one tile natively

Divisor choice: measured stroke width on matched source footprints across
divisors 1, 1.5, 2, 3+; divisor 1 (native) won on every axis (stroke width,
tile yield, gray fringe). See `doc/raw_dataset_extraction_knowledge.md`.

Materialization result:

- source rows: 281, kept: 165, dropped (below 480 px native): 116
- output: `dataset/regions_ako5ver2_native_20260725/`, about 427 MB
- alignment: 137 of 165 regions shifted, median correction 21 px, p90 74 px,
  3 boundary hits (down from a 768-space check showing this would have been 6
  of 20 sampled regions before the wider two-stage search)

### Native Masks And Downstream Pipeline

`tools/pair_extraction/build_region_valid_masks.py` gained `--image-size 0`
(skip square-fit, keep native variable aspect) and `--reuse-source-images`
(avoid a second full-size image copy).

Pixel-unit mask thresholds had to be recalibrated for native scale; the 768
default `--support-px 5` marked most real strokes as unsupported. Adopted
`--support-px 20 --window 61 --expand-ignore 16 --close-ignore 16`. Full sweep
in `doc/raw_dataset_extraction_knowledge.md`.

- target: `dataset/regions_ako5ver2_native_20260725_masked_line_conservative/`
- rows: 165

`tools/pair_extraction/tile_region_manifest_480.py` changes to support native
scale:

- reads `native_long_side` for the `src_per_out` calculation
- re-reads tiles from disk on demand instead of holding every candidate image
  in memory, since native-resolution regions produce far more candidates than
  the 768-normalized manifest could
- `--dedup-scope page`: deduplicates in source-page coordinates instead of
  per-region, so overlapping parent/child regions of the same page cannot
  contribute the same content twice; `page_bbox` recorded in the CSV

New review tool: `tools/compare/make_tile_review_sheet.py` pastes tiles at
native 480 px size instead of thumbnail scale, needed because thumbnails hid
exactly the gray-fringe and content-mismatch signals this review depends on.

### Native Strict Extraction Result

Measurement pass mirrored the 768-normalized one:
`results/ako5ver2_native_tiles_480_measure_all.csv` (3,876 tiles, 165 regions,
36 pages). Same edge-tolerance rescale applied: `--close-px 22
--strict-close-px 8` at native versus `8` / `3` at 768.

First strict pass (no score cutoff): 498 tiles from 636 candidates, all 165
regions used. Full-resolution review
(`results/ako5ver2_native_tiles_480_strict_review_{top,spread,tail}.png`) found
the top and middle ranks strong (clean face/figure correspondence, sharp
strokes) but the tail weak, including one tile that passed every individual
gate (edge F1/recall/precision, ink, width all in range) while showing
unrelated rough scratch marks matched to an unrelated line "stitch" symbol.
Recorded as a policy note in `doc/region_dataset_extraction_policy.md`: tile
score is not a content-match guarantee, and thumbnail QC can hide this failure
mode.

That specific case scored 2.01, near the set minimum (1.88 to 4.80), so a score
cutoff was applied:

- `--min-tile-score 2.5`
- CSV: `results/ako5ver2_native_tiles_480_strict_cut25.csv`
- QC: `results/ako5ver2_native_tiles_480_strict_cut25_qc*.png`
- result: 422 tiles from 112 regions (down from 498 tiles / 119 regions
  uncut)
- full-resolution tail re-check
  (`results/ako5ver2_native_tiles_480_strict_cut25_review_tail.png`): no more
  wild semantic mismatches; remaining weak tiles are sparse/faint rather than
  content-mismatched
- status: dry-run only, not yet `--save`d

For scale: the previous 768-normalized strict subset was 90 tiles from 45
regions. The native cut25 subset is about 4.7x more tiles from about 2.5x more
regions, using the same filter design.

User reviewed the `cut25` QC: top ranks were clean, tail showed correct
semantic content with loose alignment rather than mismatch. Accepted the tail
alignment looseness for now and approved proceeding to a training pass.

### Save And Integrity Audit

Saved with the reviewed `cut25` filter settings:

- list: `dataset/pairs_480/valid_train_ako5ver2_native_strict_cut25_20260725.txt`
  (422 rows)
- line dir: `dataset/pairs_480/train/line_ako5ver2_native_strict_cut25_20260725`
- rough dir: shared `dataset/pairs_480/train/rough` (tile names are prefixed
  `ako5nat_`, so no collision with other sources)

Integrity audit:

```bash
./venv/bin/python tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists dataset/pairs_480/valid_train_ako5ver2_native_strict_cut25_20260725.txt \
  --eval-lists dataset/pairs_480/valid_test.txt dataset/pairs_480/eval_fixed_clean_lineart004.txt \
  --line-dir-override ako5nat=dataset/pairs_480/train/line_ako5ver2_native_strict_cut25_20260725 \
  --output-summary results/pair_dataset_integrity_summary_ako5ver2_native_strict_cut25.csv \
  --output-findings results/pair_dataset_integrity_findings_ako5ver2_native_strict_cut25.csv
```

- findings: 0

### Training

Runner: `experiments/run_ako5ver2_native_strict_cut25_480_warm_clean_bce_e10.sh`

Mirrors the earlier `nobeta_top300` warmstart control run: `unet`, warmstart
from `checkpoints/shape1_clean_split_bce/best.pth`, strict resume, 480px,
10 epochs, `lr 1e-5`, `pos-weight 3.0`.

- checkpoint dir: `checkpoints/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10/`
- log: `logs/train_ako5ver2_native_strict_cut25_480_warm_clean_bce_e10.log`
- output dir: `results/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10_outputs/`
- final epoch: `Epoch 010/10: G=0.2617 D=0.0000` (monotonic decrease from 0.3048)
- quicklook (rough / GT line / output):
  `results/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10_quicklook.png`

Visual interpretation:

- Loss decreased cleanly and training/inference completed without error on all
  422 tiles.
- Output is not clean line art. It reproduces the same soft / density-map
  failure mode seen in every earlier keep281-derived run (768-normalized
  top300, linefocus, nobeta, and the variable-region masked runs): gray,
  smeared texture that follows rough stroke direction without converging to
  binary lines.
- This isolates the cause: fixing the source-scale normalization and the
  tile-level content-mismatch problem did not fix this failure mode. The
  10-epoch BCE-heavy warmstart recipe itself is the likely limiting factor, not
  data quality, at least at this data scale (422 tiles).
- Do not read this run as validating or invalidating the native
  re-materialization work; it only shows this particular short recipe is not
  sufficient. `strict88` and other keep281 variants hit a similar ceiling under
  the same recipe family, per `doc/model_results_summary.md`.

## 2026-07-26 fighting (formerly lineart) Dataset

Applied the ako5ver2-validated strict stroke-scale filter to a third source,
originally uploaded and referred to as `lineart`, then renamed to `fighting`
per user request: the name was too generic and collided with an unrelated
legacy `lineart_`-prefixed category already in
`lineart/dataset_gate.py`'s `DEFAULT_LABELS` (168 pre-existing files, unrelated
to this new source).

Source: `dataset/raw_zips/dataset_fighting.zip`, 192 pre-cropped native 480x480
rough/line pairs (8 pages x 24 tiles), no region matching needed since pairs
are already 1:1 and visually well aligned.

New tool: `tools/pair_extraction/filter_fighting_pairs.py` (reuses
`analyze_tile` / `build_mask` exactly as validated on ako5ver2 native tiles).

Result: 192 source pairs, 40 accepted (21%), integrity audit 0 findings. Full
details in `doc/dataset_status.md`.

Status: reviewed, saved, audited; not yet trained on.

## Next Actions

1. The native strict cut25 extraction and materialization pipeline is
   considered validated: clean top/tail QC, 0 integrity findings, correct
   scale/mask calibration. The remaining gap is model-side, not data-side.
2. Do not conclude anything about data quality from the soft/density-map output
   above; it matches the ceiling seen across every keep281-derived recipe of
   this family. Model-side next steps belong in `doc/model_results_summary.md`
   directions (longer training, non-BCE-heavy loss, or one of the existing
   halo/Lucy/cleanup candidates) rather than further data changes.
3. If data-side work continues on ako5ver2, prefer loosening
   `--max-per-region` / stride on the native pipeline now that per-tile
   filtering is scale-correct, since the filter design itself held up.
4. Keep strict88 and the 768-normalized keep281 tile sets separate from the
   native tile set for now; do not mix sources until target semantics are
   normalized.
5. Decide whether umbrella/layer-difference rows should stay held, get manual
   masks, or become tagged future routing examples.
6. Remove or archive per-sample output directories after the current review is
   done; keep only the referenced montage images as durable result artifacts.

## 2026-07-25/26 lineart→fighting, kurip→fitness Renames; housei Added

User asked to apply the ako5ver2 native-strict pipeline to three more raw
sources (`kurip`, `housei`, and the originally-uploaded `lineart`), to broaden
the training pool for future model-side work while treating extraction
methodology itself as settled.

`lineart` renamed to `fighting` immediately (too generic; collided with an
unrelated legacy `lineart_`-prefixed category). Result: 40 tiles from 192
pre-cropped native pairs, 0 integrity findings. Details in
`doc/dataset_status.md`.

Two overnight background agents were launched for `kurip` and `housei`, with
explicit instructions to review, save, audit, and write a standalone
`doc/agent_report_*.md` (not touch shared docs directly, to avoid collision).
Both agents were later found dead: `SendMessage` to their agent IDs returned
"No transcript found." Picked up their partial work directly instead.

While resuming, discovered this environment silently kills long-running
background processes around 10-13 minutes, with no traceback, regardless of
execution method (detached `nohup`, harness-tracked `run_in_background`, or
an autonomous agent). A command piped through `tail`/`tee` can report exit
code 0 even when the upstream process was killed, so file existence had to be
checked directly rather than trusting task-completion notifications. Recorded
as a standing operational note in `doc/raw_dataset_extraction_knowledge.md`.
Workaround: added `--offset`/`--limit`/`--append` to (renamed)
`filter_matched_region_tiles.py` and ran full passes in ~250-row chunks.

kurip (as `fitness`): `diagnose_pair_alignment.py` recommended
`needs_global_or_local_alignment`. Used `match_kurip_regions.py` (generic
despite the name) for line-anchored local-offset matching, then the strict
filter. 1,509 raw matches; a first pass at the default `--min-tile-score 2.5`
needed raising to `2.9` after tail review; final: 271 tiles from 37 regions,
full-range re-review clean, 0 integrity findings.

housei: same route (`needs_global_or_local_alignment`). Required a small fix
to `diagnose_pair_alignment.py` (missing bare-filename fallback for empty
`zip_root`, since housei's zip is flat). 602 raw matches; a full-permissive
measurement pass showed the strict content gates (not the score cutoff) are
the yield bottleneck (~51 tiles regardless of score threshold 1.5-2.9); final
production filter gives 25 tiles from 10 regions, 0 integrity findings.

Renamed `kurip`→`fitness` since `kurip` was a person's username. Per explicit
user decision, all pre-session leak-era `kurip`-named artifacts were deleted
outright rather than renamed: 2,526 same-coordinate-era tiles, 11
`line_kurip_*` directories, 29 training lists, ~40 `results/kurip_*` files, 9
checkpoints (~1.4 GB), 8 one-off `make_kurip_*_compare.py` scripts, and 4 dead
experiment runners. Reclaimed disk space; also removed most literal username
occurrences from the repo. 5 still-active infra scripts
(`match_kurip_regions.py`, `extract_kurip_matched_tiles.py`,
`prepare_kurip_tiles.py`, `refine_kurip_vlm_tiles.py`,
`vlm_review_kurip_matches.py`) were deliberately left un-renamed — still
functional, referenced in `doc/EXTRACTION_RULES.md`, and one
(`prepare_kurip_tiles.py`) is shared with hamlabi — pending a separate user
decision on that broader rename.

Full per-source detail: `doc/dataset_status.md`
(`## fitness`, `## housei`, `## fighting` sections).

## 2026-07-26 Cross-Dataset Strategy Note; housei Gate Diagnosis And Relaxation

User asked to step back and confirm what is universal across all sources
processed so far versus what must be tuned per source, before continuing.
Written up as a new top section in `doc/region_dataset_extraction_policy.md`
("Cross-Dataset Strategy"). Key correction during that discussion:
`fighting` is a target-quality reference, not an extraction-procedure
template, since its source data arrived already cropped and aligned before
this session touched it; the "heavyweight" (ako5ver2, full region proposal +
recursive child search) and "lightweight" (fitness/housei, grid + local
offset only) matching routes are two genuinely different procedures selected
by `diagnose_pair_alignment.py`'s recommendation, not one universal procedure.

New tool: `tools/pair_extraction/diagnose_gate_funnel.py` — evaluates every
strict-filter gate independently per tile (unlike `analyze_tile()`, which
returns `None` at the first failed gate, hiding later-stage metrics for
rejected tiles), to identify which specific gate is a source's yield
bottleneck.

Run on housei's full 602 raw matches: `soft_ink_ratio` accounts for 168 of 214
rejections at the point it's checked (78%), far above every other gate
(next-worst was `ink_range` at 31%). This matches housei's rough style being
more pencil/gray-textured than ako5ver2 or fitness. Confirmed the score cutoff
was not the bottleneck (yield stayed ~51 across `--min-tile-score` 1.5-2.9
under partial gates).

Visually verified (23-tile spot check at `--max-soft-ink-ratio 0.50`, then the
full run) that relaxing this one gate from the ako5ver2-derived default `0.40`
to `0.50` holds quality: no semantic mismatches, no visible gray/density-map
degradation. Re-saved housei at this setting: 65 tiles from 14 regions (up
from 25 from 10), integrity audit 0 findings. Further relaxation has more
room (~97 tiles at 0.60, ~134 at 0.70) but was intentionally not pursued
further, per user direction, to move on to training/inference results.

Full detail: `doc/dataset_status.md` (`## housei`).

## 2026-07-26 Combined Training, Alignment Root-Cause, Panel-Boundary Plan

First combined training run across all four reviewed native-strict sources
(ako5ver2 422, fitness 271, housei 65, fighting 40 = 798 tiles; combined line
dir via symlinks, `dataset/pairs_480/valid_train_combined_20260726.txt`).
Same warmstart recipe as prior single-source runs (10 epochs, BCE-heavy,
warmstart from `shape1_clean_split_bce`). Loss decreased monotonically
(0.2793->0.2459). Runner: `experiments/run_combined_20260726_warm_clean_bce_e10.sh`.

Visual review (per-source sheets, `results/combined_20260726_perSource_*.png`)
found output crispness varies sharply and consistently by source: fighting
traces structure closely (with some thickening/blur at joints and dense
hatching); fitness partially recognizable; ako5ver2-native and housei stay
soft/gray/density-map-like. This revises the earlier ako5-only conclusion
("the ceiling is recipe-side, not data-side"): data cleanliness clearly
interacts with the recipe's ceiling.

Quantified the correlation: chamfer distance (already computed by the strict
filter but gated at an effectively-disabled `--max-chamfer 200`) tracks output
crispness almost exactly (fighting median 10.45 best, ako5ver2 17.52 worst).
A calibration check (4 lowest-chamfer tiles per source,
`results/chamfer_calibration_best4_per_source.png`) confirmed chamfer is a
valid, comparable metric across sources: every source's best tiles show tight
rough/line edge overlap, including ako5ver2's. The problem is the *rate* of
good tiles, not a ceiling on how good ako5ver2's alignment can get.

Tested a post-hoc `chamfer<=12` re-filter (217 tiles combined:
ako5nat 53/422, fitness 108/271, housei 31/65, fighting 25/40;
`experiments/run_alignfilt12_20260726_warm_clean_bce_e10.sh`, final loss
0.2016). Direct same-sample comparison against the 798-tile run
(`results/compare_798_vs_alignfilt12_20260726.png`) showed no visible output
quality improvement, despite the metric correlation holding across sources.
Recorded as an open question in `doc/region_dataset_extraction_policy.md`
("Determining The Alignment Threshold").

### Alignment Gate vs Style Gate (Architecture Change)

User's framing: alignment-agreement extraction (does rough sit on line) is a
purely geometric question and should be one fixed, cross-dataset procedure;
appearance-based extraction (stroke width, gray fringe, ink density) is
inherently per-source and should stay tunable. Refactored
`tools/pair_extraction/tile_region_manifest_480.py`'s `analyze_tile()`
accordingly: split into `alignment_metrics()` / `alignment_gate_pass()`
(chamfer, strict-tolerance edge F1/recall/precision, using new fixed
`ALIGNMENT_*` module constants, no CLI override) and `style_metrics()` /
`style_gate_pass()` (existing per-source CLI args, unchanged). Removed the
now-dead `--close-px`/`--strict-close-px`/`--truncate-px`/`--min-f1`/
`--max-chamfer`/`--min-strict-f1`/`--min-strict-line-recall`/
`--min-strict-rough-precision` CLI args from `tile_region_manifest_480.py`,
`filter_matched_region_tiles.py`, and `filter_fighting_pairs.py` so they can't
be silently re-tuned per source again. Regression-checked: fitness chunk1
still gives exactly 54 tiles under the new code. Full policy writeup:
`doc/region_dataset_extraction_policy.md` ("Alignment Gate vs Style Gate").

### Root Cause: Scale/Deformation, Not Just Imprecise Search

Tested whether extra local search (translation only, wider range/finer step)
closes the chamfer gap: modest gains only (~5% mean reduction on a 15-tile
fitness sample), suggesting the current translation search is already near a
local optimum for most tiles.

User's explanation, then confirmed by test: rough is inked from a printed
copy with no production reason to preserve pixel alignment, and finished line
art goes through a finishing pass that rescales/repositions content per
panel, per character, or occasionally a smaller partial region — a real
geometric transform, not noise. Added scale to the search (0.85-1.15) on
fitness's 5 worst-chamfer tiles (originally chamfer 20.8-21.8): reduced to
13.7-15.7 (25-38%), with 3 of 5 tiles picking a non-1.0 scale. Confirms real
scale mismatch that no current tool corrects for: `match_kurip_regions.py` is
translation-only; `match_hamlabi_regions.py` only tries a few discrete global
scales per parent region, not per-panel/per-character. Local mesh-level
(non-uniform) deformation is decided out of scope for now (too open-ended to
model generally).

### Panel-Boundary-First Region Segmentation (Planned, Paused)

Root cause of "one region contains 3 panels" / "region cuts across a panel
boundary" in `match_hamlabi_regions.py`'s output: its region proposal step
groups purely by line-ink connected-component proximity, with no concept of a
panel border. A first attempt at panel-border detection (long-line
morphology) failed on a real ako5ver2 page, flagging character hair as false
borders.

Key fact from the user: panel border lines are composited from a separate
layer in the original production file and are not present in the finished
line art layer at all; the rough sometimes has ruled or roughly-sketched
panel lines. Panel boundaries therefore cannot be recovered from the
flattened rough/line raster images alone.

Plan, paused here (panel-layer extraction happens on another machine): (1)
extract the panel-border layer as its own dataset, (2) segment pages into
clean single-panel regions using it (full-bleed panels deferred), (3) verify
alignment per panel (translation + one uniform scale expected, no mesh
deformation, more tractable than whole-page matching), (4) split into
per-character regions within a panel when multiple characters are present,
re-scoring alignment per character with only low-scoring cases needing manual
review, (5) defer finer sub-character regions (may have irregular/"special"
deformation). Full detail: `doc/raw_dataset_extraction_knowledge.md`
("Residual Misalignment").

## 2026-07-27/29: OOM Root-Cause, Coarse-To-Fine Koma Alignment, 5-Source Panel-To-Tile Pipeline

The panel-border-layer extraction unblocked for ako5ver2/hamlabi (delivered
as `dataset_ako5_koma.zip`/`dataset_hamlabi_koma.zip`), and two more koma
sources arrived: `fitness`'s own koma layer (as `dataset_kurip.zip`,
reconciled and renamed) and a wholly new source, `gakuen`.

### OOM Root-Cause And Fix

The "background jobs silently die around 10-13 min" note from 2026-07-26 was
wrong — re-checked via `journalctl -k` and found 5 genuine kernel OOM kills
(15-28 GB anon-rss) on the day it was first observed. Root cause: several
scripts stored raw numpy slice *views* of full-resolution source pages
directly in long-lived lists (`match_kurip_regions.py`,
`filter_matched_region_tiles.py`, `match_koma_panels.py`'s `crop_page`,
`materialize_koma_panels.py`'s never-evicted `page_cache`) — a "small"
stored tile secretly kept its whole ~35 MB parent page alive. Fixed by
copying at the point of storage / evicting the cache on page change. Full
detail and the exact mechanism: `doc/raw_dataset_extraction_knowledge.md`
("Long Background Jobs Died From Real OOM").

### Coarse-To-Fine Koma Alignment Search

hamlabi's panel detection showed poor alignment; diagnosed to the old flat
`--max-shift 160` (housei's already-corrected value) still being
insufficient — 48-83% of panels across all four then-available sources had
a true offset beyond 160px. Added `coarse_offset_estimate()` to
`match_koma_panels.py`: a cheap wide pre-search on a downscaled image
(default 480px range) finds the neighborhood before the existing
native-resolution fine search refines it (~544px effective combined reach).
A `matchTemplate`/cross-correlation variant was tried for speed and
rejected — it measurably picked worse neighborhoods on real panels than the
sparse-edge-coordinate F1 metric the fine search already uses. Added a
do-no-harm fallback (report the true zero-shift baseline if the coarse-guided
result isn't actually better by the same F1 metric) after finding 3-4
panels/source where the coarse stage locked onto a spurious distant optimum.
Also fixed a QC-display-only bug (wrong pad/center in the montage generation
code, unrelated to the correctly-computed CSV chamfer values) that had been
showing garbled/black crops for any panel with a large coarse-found offset —
this affected every already-generated QC montage and required regenerating
all of them for reliable visual review. Full detail:
`doc/raw_dataset_extraction_knowledge.md`.

### Symlink Incident And Storage Policy Change

Three root-level compatibility symlinks (`gakuen`, `dataset_ako5.zip`,
`dataset_hamlabi.zip`) were each destroyed the same day: an `scp` upload to
the symlink's path followed the link and overwrote the real archive under
`dataset/raw_zips/` in place, rather than replacing the link. Original bytes
unrecoverable each time. Renamed the resulting archives to versioned
filenames and, per explicit user decision, discontinued root-level symlinks
entirely going forward — this supersedes the old "allowed for sources with a
hardcoded legacy reference" exception. Full detail:
`doc/raw_dataset_storage_policy.md` ("Compatibility Symlinks —
Discontinued").

### Broader Page-Extraction Bug Fix, Final Panel Detection (5 Sources)

The corrected re-uploads (fixing gakuen's original page1/11/12
ラフ-vs-下絵 issue) turned out to be an incomplete fix for a broader
page-extraction bug on the user's side, affecting `line` files (not just
`sketch`) across gakuen, housei, and fitness. Final corrected archives:
`dataset_ako5_koma_v2.zip`, `dataset_hamlabi_koma_v2.zip`,
`dataset_fitness_koma_v2.zip`, `dataset_gakuen_v3.zip`,
`dataset_housei_v4.zip` (ako5ver2/hamlabi were confirmed final for the day;
gakuen/housei/fitness needed one more round). Panel detection re-run against
all five with the final coarse-to-fine + do-no-harm code:

| source | panels | pages | base chamfer median | best chamfer median | worse-than-baseline |
|---|---:|---:|---:|---:|---:|
| ako5ver2 | 175 | 44 | 29.50 | 19.38 | 2 |
| hamlabi | 65 | 13 | 32.75 | 23.07 | 0 |
| fitness | 110 | 36 | 25.25 | 21.29 | 2 |
| gakuen | 46 | 16 | 25.35 | 14.63 | 0 |
| housei | 81 | 18 | 15.19 | 13.01 | 3 |

Notably, `housei_010`/`011`/`012` — previously flagged (2026-07-26) as a
distinct "high-residual-misalignment cluster" and hypothesized to be a real,
possibly-irreducible non-uniform-deformation case — dropped from
chamfer 27.8-34.4 to 9.64-15.4 (in line with the rest of the source) once
re-run against the bug-fixed `housei_v4` data. That hypothesis is now
considered wrong: the cluster was a page-extraction artifact, not real
deformation.

### Panel-To-Tile Pipeline, Launched For All 5 Sources

Built `tools/pair_extraction/run_koma_tile_pipeline.sh`, a driver chaining
`materialize_koma_panels.py` (`--max-chamfer 45`, generous — the real
alignment gate is the fixed `ALIGNMENT_*` tile-level constants, this is only
an efficiency pre-filter) -> `split_koma_panel_subregions.py`
(`--refine-alignment` default on) -> `build_region_valid_masks.py` (native
settings: `--support-px 20 --window 61 --expand-ignore 16 --close-ignore 16`)
-> `tile_region_manifest_480.py` (ako5ver2-validated native-strict style
gates for every source; housei alone keeps its established
`--max-soft-ink-ratio 0.50` relaxation). Found and fixed one more manifest-
schema gap while testing: `materialize_koma_panels.py` still read
`entry["line"]`/`entry["sketch"]` directly instead of using
`match_koma_panels.py`'s `resolve_files()` helper, so it crashed on gakuen's
embedded-`files`-dict schema.

Piloted on gakuen (smallest, 46 panels): 117 sub-regions, 202 tiles, ~33
min end to end. Per user direction, launched the remaining four sources
(ako5ver2, hamlabi, fitness, housei) as one sequential unattended run via
`tools/pair_extraction/run_all_koma_pipelines_20260729.sh`, started fully
detached (`nohup ... & disown`) so it survives this CLI session ending;
progress logs to `results/koma_memtest/master_pipeline_20260729.log`, and
`experiments/send_autoloop_notification.sh` (existing ntfy config in
`config/autoloop_notify.env`) fires on completion with a per-source tile
count summary. Estimated total runtime ~5 hours by linear extrapolation from
the gakuen pilot.

## Next Actions

1. **Check the overnight panel-to-tile run's outcome** (ntfy notification or
   `results/koma_memtest/master_pipeline_20260729.log` directly if the
   session dropped): confirm all 4 remaining sources
   (ako5ver2/hamlabi/fitness/housei) completed, check each source's final
   tile count and QC (top and tail, at native resolution — this project's
   standing rule that thumbnail QC and gate scores alone are not a
   content-match guarantee still applies here).
2. Once all 5 sources have tiles, decide on a training experiment: train each
   source separately first, or design a deliberate mixing/curriculum
   experiment across ako5ver2/hamlabi/fitness/gakuen/housei — do not
   casually concatenate without a stated rationale, per this project's
   long-standing rule.
3. Revisit whether `--max-soft-ink-ratio` needs a per-source
   `diagnose_gate_funnel.py` pass for ako5ver2/hamlabi/fitness/gakuen (only
   housei has an established relaxed value so far); yield may be
   conservative for the others under the shared default.
4. Decide whether to rename the remaining `kurip`-named infra scripts
   (`match_kurip_regions.py` and others) — still open, unrelated to tonight's
   work.
5. Model-side experiment design using the now-much-larger reviewed pool
   remains open; the soft/density-map ceiling seen in earlier single-source
   runs should be re-evaluated once trained against this alignment-corrected
   data, not assumed to still apply as-is.

## 2026-07-29 (later): `results/` Per-Source Reorganization + Cleanup

Visual QC of hamlabi's final koma tiles (top/tail) came back clean. User
then flagged that `results/` (427 top-level entries, 1.3G) had become too
cluttered to tell which files were current, and asked for per-work folders
plus deletion of rarely-used old files. Confirmed scope (full `results/`,
keep-latest-version-only per work) before acting, since deletion is
one-way for this gitignored directory.

Discovered a 6th source, `fighting` (see `doc/dataset_status.md`), distinct
from `fitness`/`kurip` — a small already-fixed-tile source (8 pages, 192
tiles) that never went through region matching or the koma pipeline, so
nothing under it is superseded; it got its own folder with everything kept.

For the 5 koma-pipeline sources (ako5ver2/hamlabi/fitness/gakuen/housei),
kept only the latest accepted panels manifest (+overlays/qc) and the
completed 2026-07-29 materialize/subregion/tiles_480 outputs; deleted every
earlier dated/versioned panel-detection iteration and all pre-koma legacy
approaches (region matches, native_strict tiles, ako5ver2's
keep281/varregion/strict88 tile-selection-and-training-run experiments,
hamlabi's old region/codex/vlm review files, etc.) — those approaches are
fully superseded by the koma pipeline's alignment quality and yield, and
per `doc/RESULTS.md`'s own cleanup policy, per-sample inference directories
and old QC sheets are meant to be deleted once superseded.

Result: 427 → 126 top-level entries, 1.3G → 465M. Left untouched: cross-
source model/architecture-comparison experiment output (`halo_*`,
`haze_uncertainty_*`, `router_*`, `line_refiner_*`, `fixed_output_metrics_*`,
`combined_20260726_*`, `dataset_gate_*`, etc.) — these aren't organized
per-manuscript-source and weren't part of this ask; a separate pass would
be needed to judge which of those are still relevant.

Updated `doc/RESULTS.md` (new "Per-Source Folders" section, moved the
`hamlabi_filtered398_unet480_epoch040_compare.png` reference to its new
path) and pruned `config/results_manifest.json` (21 stale entries for
already-nonexistent paths removed; none of its tracked paths were among the
files moved into source folders).

New per-source koma pipeline runs should write directly into
`results/<source>/` going forward (update `run_koma_tile_pipeline.sh`'s
`--out`/`--csv-out`/etc. paths next time it's touched, rather than
retrofitting immediately).

## 2026-07-29 (later still): Combined 5-Source Koma Training Launch

User visually reviewed final tile QC (top/tail, native resolution) for all 5
koma sources and confirmed all okay. Asked whether to train mixed or
per-source; recommended mixed — per-source counts (132-536 tiles) are small
enough to risk overfitting alone, and the pool only reaches 1489 tiles
combined — with the tradeoff noted (mixing can average out a source's style
idiosyncrasies; a per-source fine-tune from the mixed base stays an option
later if one source's fidelity needs attention).

Built the combined training set the same way as the 2026-07-26 combined run
(`## 2026-07-26 Combined Training...` above): per-file symlinks from a new
`dataset/pairs_480/train/line_combined_koma_20260729/` into each source's own
`line_<source>_koma_20260729/` dir (tile name prefixes like `ako5ver2koma_`
already prevent collisions), and `valid_train_combined_koma_20260729.txt` as
the straight concatenation of the 5 per-source lists (536+164+455+202+132 =
1489).

New runner `experiments/run_combined_koma_20260729_warm_clean_bce_e10.sh`
reuses the exact 2026-07-26 combined recipe unchanged (unet, warmstart from
`shape1_clean_split_bce`, 10 epochs, bce-heavy, lr 1e-5) specifically so this
result is a direct, recipe-controlled comparison against that earlier 798-tile
run — isolating the effect of the larger, alignment-corrected koma dataset.
Launched via `nohup ... & disown` (PID 609254; training log
`logs/train_combined_koma_20260729_480_warm_clean_bce_e10.log`, checkpoint
dir `checkpoints/combined_koma_20260729_480_warm_clean_bce_e10`, inference
output `results/combined_koma_20260729_480_warm_clean_bce_e10_outputs`).

Also launched a second, independently `nohup`+`disown`'d wrapper process that
waits on PID 609254 and then calls `send_autoloop_notification.sh` — done as
a separate detached process rather than a Bash-tool `run_in_background` call,
because (per the earlier finding this session) an in-session background
monitor can die with the CLI session itself even though the actual work it's
watching survives; only a fully OS-detached process is guaranteed to still
fire the ntfy notification if the session drops before training finishes.

## 2026-07-29 (later still): combined_koma_20260729 Result, Recipe Pivot, Branch Merge

`combined_koma_20260729_480_warm_clean_bce_e10` finished (loss 0.2413 ->
0.2183 monotonic, `checkpoints/combined_koma_20260729_480_warm_clean_bce_e10/best.pth`).
Ran inference on the standard 8-sample clean eval list
(`dataset/pairs_480/eval_fixed_clean_lineart004.txt`) and built a rough/
output/GT montage:
`results/compare_combined_koma_20260729_480_warm_clean_bce_e10_cleaneval.png`.

Visual result: same soft/gray "marbled" density-map texture seen in every
prior single-source and the 2026-07-26 combined run — follows rough stroke
direction and picks up general density but never converges to binary line
strokes, on all 8 samples including ones with crisp GT lines. No content
mismatches, just the same known ceiling.

Diagnosis (why this run isn't very informative on its own): the recipe's
loss weights (`bce=0.6 l1=0.2 tolerant=0.05 ink=0.02`, every structure/
binary/skeleton/width/threshold/bg-haze/feature-match weight at `0.0`,
`adv=0.02`) are essentially bare BCE+L1 — the same weak recipe that already
produced this exact failure mode on every earlier single-source and
2026-07-26 combined run, regardless of data quality. This run mostly
reconfirms the recipe ceiling rather than telling us something new about
the koma pipeline's data quality; per `doc/model_results_summary.md`, the
`lucy_mild_aux_msgan`/`lucy_thin_aux_msgan`/msgan family (structure loss +
adversarial + atari/lucy aux hint) already produces real binary-ish line
output (F1@2px ~0.40-0.41) under the *old* pre-koma data, so that family had
never been tested against the new alignment-corrected koma dataset.

Decision: retrain the `lucy_mild_aux_msgan` recipe (from
`experiments/run_lucy_mask_deep_survey.sh` / `run_badrough_retrain_survey.sh`:
`cleanup` model, `--gan --multiscale-gan`, `bce=0.75 l1=0.03 shape=0.08
ink=0.14 binary=0.10 structure=0.04 adv=0.03 fm=0.08`, lucy_mild atari-hint
aux channel) on the combined 1489-tile koma dataset, to isolate whether the
soft/density-map ceiling is recipe-side (as suspected) rather than
data-side, now that the data itself is the largest and most alignment-clean
pool built so far.

New runner: `experiments/run_combined_koma_lucy_mild_msgan_20260729.sh`.
Chains: (1) inference of the already-trained plain-bce koma checkpoint on
the `eval_clean_lineart004_8.txt` list for a same-sample before/after
baseline (the pre-existing `lucy_mask_deep_e2_lucy_mild_aux_msgan` reference
outputs no longer exist post-2026-07-25 results cleanup, so this run only
compares against the plain-bce koma baseline, not the old pre-koma
lucy_mild); (2) atari-hint generation
(`checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`) over
all 1489 koma train tiles + 8 eval tiles; (3) `lucy_mild` aux preprocessing;
(4) 3-epoch `cleanup`+GAN+multiscale-GAN training with the koma train
list/line-dir; (5) eval inference, montage
(`results/compare_combined_koma_lucy_mild_msgan_20260729.png`), and fixed
metrics (`results/fixed_output_metrics_combined_koma_lucy_mild_msgan_20260729_compare.csv`).

Launched detached (`nohup ... & disown`, PID 615641) so it survives the CLI
session ending; completion fires the existing
`experiments/send_autoloop_notification.sh` ntfy notification, same as the
2026-07-29 combined-bce launch above. Status as of this note: still running
(atari aux generation done, lucy_mild preprocessing/training in progress).

### hamlabi-region-extraction Branch Closed Out

Per user direction, committed and fast-forward-merged this branch into
`main` to close out the region-extraction line of work, since the next
branch (architecture improvement vs router/MoE) depends on this retrain's
result and shouldn't be decided or started on this branch.

Commit `b47b226` bundled: the koma alignment do-no-harm fix (guards against
a nominally-higher-F1 offset that actually lands on a blank/void crop, found
on gakuen) and a QC padding fix in `match_koma_panels.py`; a
`resolve_files()` fix in `materialize_koma_panels.py` for gakuen's
embedded-files-dict manifest schema; the new
`run_koma_tile_pipeline.sh`/`run_all_koma_pipelines_20260729.sh` driver
scripts; the `doc/raw_dataset_storage_policy.md` update recording the final
corrected archive versions and the SCP-through-symlink incident; the
`results/` per-source reorganization (`doc/RESULTS.md`,
`config/results_manifest.json`, deletion of superseded
`results/ako5_region_matches*`/`ako5_region_tiles.csv`); and both new
combined-training runners (plain-bce baseline and this lucy_mild msgan
retrain).

Since `main` was already a strict ancestor of this branch (0 main-only
commits), the merge was a clean fast-forward. To avoid any risk to the
lucy_mild retrain running detached from this same working directory (a
`git checkout main` would rewrite the working tree, and the running bash
process was mid-execution of a script file only present on this branch),
the merge was done without ever checking out `main`: `git branch -f main
hamlabi-region-extraction` followed by `git push origin main`, staying on
`hamlabi-region-extraction` throughout. Both branches pushed to origin at
`b47b226`.

### Next Actions

1. Wait for `combined_koma_lucy_mild_msgan_20260729` to finish (ntfy
   notification or `logs/combined_koma_lucy_mild_msgan_20260729.log`).
   Review the montage and metrics at full resolution before drawing any
   conclusion — per this project's standing rule, visual review is the
   adoption gate, not F1/chamfer alone.
2. If it produces real binary-ish line output (unlike the bare-bce runs),
   that confirms the ceiling was recipe-side: next branch should be
   architecture/recipe improvement (per `doc/model_directions.md`'s
   still-open survey directions) applied to this koma dataset, or a
   deliberate per-source curriculum/MoE split if source-style diversity
   turns out to matter.
3. If it still shows the same soft/density-map ceiling even with structure
   + adversarial + aux-hint losses, that would be new information — the
   problem may not be pure recipe weakness, and router-moe (already
   scaffolded in `../lineart-router-moe`) or a source-specific
   split/expert approach may need to move up in priority.
4. Decide the next branch (architecture-improvement vs router-moe) only
   after that review, per explicit user direction — not preemptively.

## 2026-07-31: Direction 6/8/9 Survey Concluded (Same Ceiling); Unpaired-Rough Integration; Direction 4 Branch Starting

Fixed a missed argparse-choices bug (`cleanupdark` was wired into
`build_generator()` but not into `--model`'s choices list in
`scripts/train_i2i_survey.py`, so `combined_koma_cleanupdark_20260730` had
silently failed at launch). Fixed, relaunched, and gave an honest
metrics+visual verdict: `cleanupdark` (darken-only correction) was not a net
improvement over the bidirectional `cleanup`/`lucy_mild` result (F1@2px
0.4090 vs 0.4175, chamfer worse) — not adopted.

**Direction 6 (confidence/thickness dual-head) and Direction 8 (HED-style
multi-scale side outputs), first attempt:** both new generators
(`DualHeadRefinerGenerator`, `HedUNetGenerator` in `lineart/model_zoo.py`)
converged to a chronically under-inked output after 3 epochs (F1@2px 0.198
and 0.190 respectively, vs ~0.41-0.42 for the adopted cleanup family).
Root-caused by code comparison: the adopted `ResidualCleanupGenerator`/
`DarkenOnlyCleanupGenerator` predict a small bounded correction around the
aux (atari) input's own logits (`out = aux_logits + bounded_delta`), so they
start already close to a decent output; the new architectures reconstructed
ink_logits from scratch with no anchor to aux, starting at ~sigmoid(0)=0.5
everywhere, and 3 epochs (a budget calibrated for residual-anchored models)
was nowhere near enough to learn ink density from nothing.

**Fix applied and retried:** added a shared `aux_channel_logits()` helper and
rewired both generators to predict `aux_logits + bounded_correction` (with
zero/negative-bias-initialized final layers so training starts at exactly
the aux baseline — verified via smoke test, initial output within 0.4% of
aux). Retrain results: `combined_koma_dualhead_v2_20260731` reached F1@2px
0.4096 / chamfer 4.530 (best chamfer of the whole family) but with
precision 0.279 / ink_ratio 2.609 (over-inking) and a visible dark-blob
artifact on one ambiguous sample — same soft/marbled texture family, not a
qualitative jump. `combined_koma_hed_v2_20260731` (still 3 epochs) only
reached F1@2px 0.235; a 10-epoch retry (`combined_koma_hed_v3_20260731`,
this project's standard from-scratch-unet budget) reached F1@2px 0.328,
still visually thinner than the adopted models and judged to be converging
toward the same ceiling rather than a different one — not worth chasing
further epochs.

**Direction 9 (bottleneck self-attention refiner):** implemented directly
with the residual-anchor lesson baked in from the start
(`AttentionUNetGenerator`, `SelfAttention2d` with zero-initialized `gamma`
so the block starts as a no-op) and trained for 10 epochs immediately
(`combined_koma_attn_20260731`). F1@2px 0.348, chamfer 5.841, ink_ratio
1.016 (best-balanced ink ratio of the family) but still below the adopted
~0.41-0.42 ceiling, with no visible long-range coherence benefit from the
attention block specifically.

**Verdict: Directions 5, 6, 8, and 9 all converge to the same soft/marbled
F1@2px ~0.40-0.42 ceiling (or below it, when undertrained), none producing a
qualitatively sharper result.** Per `doc/model_directions.md`'s own decision
rule, this closes out the short architecture survey without a breakthrough;
`combined_koma_lucy_mild_msgan_20260729` (`cleanup` model) remains the
adopted best checkpoint.

**Unpaired-rough data (skima) prepared and tested.** User provided
`skima_text_removal.zip` (626 full manga pages, pencil-only originals with
no line-art counterpart ever produced, text auto-removed) at the repo root;
moved to `dataset/unpaired_rough_raw/skima_text_removal.zip` and extracted
to `dataset/unpaired_rough/skima/{cleaned,auto_mask,manifest.json}` (kept
deliberately separate from the paired `dataset/raw/`/`dataset/raw_zips/`
folders). Tiled via new `tools/pair_extraction/tile_unpaired_rough.py`
(grid split + autocontrast/std blank filter, no alignment needed since
there's no line-art target) — 626 pages -> 4917 tiles, QC'd visually
(some tiles retain un-erased dialogue text, noted but not blocking for the
adversarial-only use case below).

Implemented a low-cost integration path in `scripts/train_i2i_survey.py`:
new `UnpairedRoughDataset` (rough+aux only, no target) and
`--unpaired-rough-file-list`/`--unpaired-rough-dir`/
`--unpaired-rough-aux-dir`/`--unpaired-weight` flags. Each training step, an
extra unpaired batch is passed through G and only
`adversarial_mse(D(rough, G(rough)), 1.0)` is added to the generator loss
(weighted by `--unpaired-weight`) — legal without a GT line target since the
discriminator only judges plausibility, not a specific match. Generated the
aux/atari pass for skima's tiles via the existing
`model_resnet_binft_e3_resnet_gan_advsharp_binft` checkpoint +
`preprocess_atari_aux.py --mode lucy_mild` (matching the adopted recipe's
input format).

Retrained the adopted `lucy_mild`/`cleanup` recipe with this branch added,
isolating `--unpaired-weight` as the one variable:

- `combined_koma_lucy_mild_unpaired_skima_20260731` (weight=0.03, matching
  `--adv-weight`): clear metric regression (F1@2px 0.4175 -> 0.2381,
  chamfer 4.680 -> 8.006, recall 0.597 -> 0.197) but a qualitatively
  *different* failure mode from every other experiment this session — sparse,
  high-contrast black fragments instead of the usual soft/marbled gray.
  Hypothesis: the unpaired branch has no shape/structure/continuity loss (only
  paired data gets those), and the PatchGAN discriminator judges local
  patches only, so G found a cheap local trick (isolated high-contrast
  speckles) that satisfies D without needing continuous strokes, and this
  bled into the paired output through the shared weights.
- `combined_koma_lucy_mild_unpaired_skima_w003_20260731` (weight=0.003, 10x
  lower): F1@2px 0.4076 / chamfer 4.813 / recall 0.599 — visually and
  metrically almost indistinguishable from the unmodified baseline. The
  transition between "negligible effect" (0.003) and "dramatic disruptive
  effect" (0.03) is sharp/nonlinear rather than a smooth scaling, consistent
  with the cheap-fooling-trick hypothesis (small weight can't overcome the
  reconstruction loss's dominance; once it can, the trick take over rather
  than gradually blending in). Not adopted at either weight as tested; a
  next step (not yet attempted) would be adding some form of GT-free
  continuity/self-consistency regularizer to the unpaired branch itself
  before revisiting the weight sweep.

**Direction 4 (diffusion/ControlNet) groundwork.** User confirmed several
locally-available SD1.5-family checkpoints at
`~/disk/checkpoint/Stable-diffusion/` (`v1-5-pruned-emaonly.safetensors`
plain SD1.5, plus anime-tuned merges `AOM3A1B_orangemixs.safetensors` /
`Counterfeit-V2.5.safetensors` / `BloodOrangeMix.safetensors` — the anime
merges are likely a better line-art starting point than plain SD1.5).
Installed `diffusers`/`transformers`/`accelerate`/`peft`/`safetensors` into
the project venv (added to `requirements.txt`) and confirmed feasibility:
`StableDiffusionPipeline.from_single_file()` loads `AOM3A1B_orangemixs`
correctly (unet in_channels=4, cross_attention_dim=768, standard SD1.5
config), and `ControlNetModel.from_unet()` builds a fresh ~361M-parameter
ControlNet adapter from it. The actual training script (data format
conversion, denoising training loop, VRAM/precision tuning for the 12GB
GPU) is not yet implemented — diffusers ships the model classes but not the
`examples/controlnet/train_controlnet.py` reference script, which lives in
the diffusers GitHub repo, not the pip package.

User separately confirmed the unpaired-rough pool (skima-style, pencil-only
manuscripts) could grow ~4x from material already on hand. Discussed which
scenarios that would matter for: it would matter for (a) self-supervised
domain-adaptation pretraining of the SD1.5 backbone on rough-sketch imagery
before attaching ControlNet (unsupervised, scales with image count), and
(b) a future CycleGAN-style setup once a line-only unpaired pool also
exists; it would *not* directly fix the adversarial-branch fragmentation
issue above (that's a loss-design problem, not a data-scarcity one), and it
doesn't add usable data to the current paired-refiner training (which needs
line-art GT that pencil-only manuscripts don't have).

**Decision: start a new branch for Direction 4** (diffusion/ControlNet),
since it is architecturally unrelated to the CNN+GAN refiner family this
branch (`cleanup-refiner`) has been developing. This entry is the closing
record for that family's short architecture survey before the switch.

### Next Actions

1. On the new Direction 4 branch: build the ControlNet training data format
   (rough tile as conditioning image, GT line art as target, a fixed/simple
   caption since per-tile captions don't exist) from the existing
   `dataset/pairs_480` paired tiles.
2. Adapt or write a training loop (diffusers' reference `train_controlnet.py`
   pattern) sized for the 12GB GPU (fp16, gradient checkpointing, small
   batch/grad-accum), starting from `AOM3A1B_orangemixs.safetensors` as the
   base checkpoint.
3. Before committing to full ControlNet training, consider the
   domain-adaptation pretraining step discussed above (unsupervised
   diffusion fine-tune on rough-only images, including the skima pool) if
   the 4x-larger unpaired-rough pool has materialized by then.
4. `combined_koma_lucy_mild_msgan_20260729` remains the production
   candidate on the `cleanup-refiner` line; the unpaired-adversarial-branch
   idea and the untried continuity-regularizer follow-up are recorded here
   for later revisit, not currently active.
5. Older open items (kurip-named script renames, ako5ver2 umbrella rows,
   per-source `--max-soft-ink-ratio` tuning) remain open and unrelated to
   either the concluded survey or the new Direction 4 branch.

## 2026-07-31: Direction 4 Training Script Landed, Smoke-Tested

Implemented `scripts/train_controlnet.py` on the `diffusion-controlnet`
branch, covering Next Actions items 1-2 from the prior entry:

- **Data format**: `ControlNetTileDataset` reads the existing
  `dataset/pairs_480/valid_train_combined_koma_20260729.txt` manifest
  (1489 tiles) directly — rough tile (`dataset/pairs_480/train/rough/`) as
  the ControlNet conditioning image, line tile
  (`dataset/pairs_480/train/line_combined_koma_20260729/`) as the diffusion
  target. No new dataset materialization needed. Since there is no per-tile
  caption, a single fixed caption (`"monochrome line art, clean linework,
  manga panel, black and white"`) is tokenized once and reused for every
  example (its text-encoder embedding is also computed once, not
  per-batch).
- **Training loop**: standard diffusers ControlNet recipe (VAE-encode
  target -> add noise -> ControlNet produces down/mid residuals from the
  rough conditioning -> frozen UNet predicts noise with those residuals ->
  MSE loss), built from `diffusers` model classes directly since the
  `train_controlnet.py` reference script isn't shipped in the pip package.
  Only the ControlNet adapter trains; VAE/text-encoder/UNet stay frozen.
  Uses `accelerate` for fp16 mixed precision + gradient accumulation,
  gradient checkpointing on both UNet and ControlNet. Base checkpoint:
  `~/disk/checkpoint/Stable-diffusion/AOM3A1B_orangemixs.safetensors`.
- **Smoke test**: ran on the real 12GB RTX 3060. At `--batch-size 2
  --grad-accum 4` (effective batch 8) and 512px resolution, peak VRAM was
  9.7GB/12GB (comfortable headroom) and loss dropped step to step (e.g.
  0.061 -> 0.033 -> ... -> 0.021 over 8 steps) with no errors. ~3.8s/step
  once warmed up, so a full 1489-tile epoch (186 steps at this batch
  config) is roughly 12 minutes; a saved ControlNet checkpoint is ~1.4GB.

### Next Actions

1. Launch a real (non-smoke) training run in the background — epoch count
   and whether to do the domain-adaptation pretraining step first (still
   not implemented) are open choices, not yet decided.
2. No inference/eval pipeline exists yet for the diffusion route (the
   existing `tools/evaluation/evaluate_fixed_outputs.py` assumes direct
   pixel-output CNN/GAN models, not a diffusion sampling loop) — will need
   a small `StableDiffusionControlNetPipeline`-based inference script
   before this can be compared against the `combined_koma_lucy_mild_msgan_20260729`
   baseline on the clean eval montage.
3. `combined_koma_lucy_mild_msgan_20260729` remains the production
   candidate on the `cleanup-refiner` line; unaffected by this branch.

## 2026-07-31: Direction 4 First Real Training Run + Eval — Qualitative Break, Not Yet Faithful

Ran `scripts/train_controlnet.py` for real (10 epochs, 1860 steps, batch
2/grad-accum 4/512px/fp16) in the background on the full 1489-tile
`combined_koma_20260729` manifest. Completed in ~113 min with no errors;
final checkpoint at `checkpoints/controlnet_koma_direction4_20260731/final/`
(1.4GB). Cleaned up the 10 intermediate `step_*` checkpoints afterward
(they had ballooned to ~15GB combined and pushed root-disk free space down
to 17GB) — only `final` is kept.

Wrote `scripts/infer_controlnet.py` (new): loads the trained ControlNet
onto `AOM3A1B_orangemixs.safetensors` via `StableDiffusionControlNetPipeline`
+ `UniPCMultistepScheduler`, samples from the 8-tile
`eval_clean_lineart004_8.txt` set, and writes outputs in the existing
`results/{tag}/{base}_out.png` convention — this let both
`tools/evaluation/evaluate_fixed_outputs.py` and
`tools/compare/make_multi_model_eval_compare.py` be reused unmodified for
scoring/montage against the `combined_koma_lucy_mild_msgan_20260729`
baseline (both tools needed zero changes, confirming the pixel-output
convention is diffusion-compatible as long as inference writes into it).

**Numeric result:** worse than baseline on the pixel metrics — F1@2px 0.20
vs baseline 0.42, chamfer 8.93 vs 4.68, ink_ratio 6.36 (baseline 1.56, so
~4x more ink than the baseline already over-inks relative to GT).

**Qualitative result (montage: `results/compare_controlnet_koma_direction4_20260731.png`):**
strikingly different failure mode from the CNN/GAN family. Baseline output
is the known soft/marbled gray "cleanup" look; ControlNet output is crisp,
confident, fully binary black ink with real anime-style linework (clean
eyes, hair strands, cloth folds) — the first result in this project to
visually break the soft/marbled texture ceiling. But it does this by
**hallucinating plausible content loosely keyed to the rough's rough
composition rather than faithfully tracing the specific input strokes** —
e.g. a generated face/expression that doesn't match GT's pose, invented
background elements, large solid-black fill regions not present in GT.
This is the likely cause of the bad pixel metrics: it's an unfaithful but
structurally coherent generation, not a faithful cleanup, after only 10
epochs on 1489 tiles.

**Interpretation:** consistent with expectations for this stage — 10
epochs / 1489 tiles is a small amount of ControlNet fine-tuning, and the
current inference settings (`--guidance-scale 3.0
--controlnet-conditioning-scale 1.0`) let the SD1.5 anime-merge's strong
prior dominate over the (still loosely-learned) conditioning. Untried
levers before drawing conclusions: raise `--controlnet-conditioning-scale`
(e.g. 1.5-2.0) and/or lower `--guidance-scale` (toward 1.0) to force
tighter adherence to the rough conditioning at inference time without
retraining; more training epochs; deterministic (non-CFG) sampling.

### Next Actions

1. Sweep `--controlnet-conditioning-scale` and `--guidance-scale` on the
   existing checkpoint (inference-only, cheap) before deciding whether more
   training epochs are needed.
2. If conditioning-scale sweep doesn't recover faithfulness, the
   domain-adaptation pretraining idea (unsupervised diffusion fine-tune on
   rough-only imagery before attaching ControlNet) becomes more relevant as
   a way to sharpen the model's rough-image "understanding" before the
   paired ControlNet stage.
3. Not adopted / not comparable to `combined_koma_lucy_mild_msgan_20260729`
   yet — this is a first probe, numeric metrics are currently worse despite
   the qualitative ceiling break.

**Sweep result (same day, immediately after):** ran the conditioning-scale
sweep from Next Action #1 — `controlnet_conditioning_scale` in
{1.0, 1.5, 2.0, 2.5} x `guidance_scale` in {1.0, 3.0} on the same 8-tile
eval set (`results/compare_controlnet_koma_direction4_20260731_sweep.png`,
`results/fixed_output_metrics_controlnet_koma_direction4_20260731_sweep_compare.csv`).
**Did not fix it.** F1@2px stayed flat at 0.20-0.21 across every
combination — inference-time knobs cannot recover faithfulness here.
Visually: low/default conditioning scale (1.0-1.5) keeps producing a
different, hallucinated illustration nearly ignoring the rough's specific
content; pushing conditioning scale up to 2.5 does start visibly pulling
output structure toward the rough's stroke directions, but at the cost of
turning the whole image into noisy chaotic hatching — quality collapses
rather than converging to clean faithful line art. Conclusion: this is not
an inference-tuning problem, it's an undertrained-ControlNet problem.

**Working hypothesis: "sudden convergence phenomenon."** ControlNet
training is documented (by the original ControlNet author) to often show
near-zero conditioning influence for a long stretch of training, then
transition somewhat abruptly to tight conditioning-following once training
progresses far enough — the zero-initialized output convolutions mean the
adapter's effective contribution ramps from nothing. 10 epochs / 1860
steps / 1489 tiles at lr=1e-5 is a small step count by the standards this
phenomenon is usually reported at; the current behavior (SD1.5 anime prior
dominating, conditioning only loosely shaping composition) is consistent
with being short of that transition rather than at a bad local optimum.
Not yet confirmed — untested whether substantially more steps (e.g.
5-10x, tens of thousands of steps territory) triggers the transition on
this dataset size, or whether 1489 tiles is simply too little data for the
phenomenon to kick in at all regardless of step count.

### Next Actions (revised)

1. Decide whether to commit to a much longer training run (many more
   epochs / repeated passes over the 1489 tiles) to test the
   sudden-convergence hypothesis, given the multi-hour GPU cost.
2. Domain-adaptation pretraining (unsupervised diffusion fine-tune on
   rough-only imagery, including the skima pool, before attaching
   ControlNet) remains an alternative/complementary lever, not yet tried.
3. Not adopted / not comparable to `combined_koma_lucy_mild_msgan_20260729`
   yet.

**Disk relocation + resume support (same day).** `checkpoints/` had grown
to 34GB and pushed root-disk free space down to 30GB; moved it to
`~/disk/lineart_checkpoints/` with `checkpoints` left as a symlink in this
worktree (root disk back to ~64GB free; `~/disk` has ~312GB free). Only
affects this worktree (`lineart-halo-loss`/`lineart-router-moe` are
separate worktrees with their own `checkpoints/`). Note:
`checkpoints/README.md` and `checkpoints/shape1/*.pth` were tracked in git
(pre-dating the `checkpoints/*` gitignore rule); after the move git sees
those as deleted, not yet resolved/committed — the physical files are
intact under `~/disk/lineart_checkpoints/`, this is just a pending
git-tracking cleanup decision.

Since the user wants to run long (many-hour, possibly multi-day) training
sessions on weekdays with the option to stop for the weekend, and the
initial run had no way to continue after a stop/crash, added
crash/pause-resume support to `scripts/train_controlnet.py`:
`--resume-from-checkpoint <output-dir|latest>` calls `accelerator.load_state()`
on a `resume_state/` directory (saved via `accelerator.save_state()` at
every `--save-steps`, overwritten each time, alongside a small
`trainer_state.json` tracking `global_step`) to restore model + optimizer
state and continue `global_step` counting correctly against
`--max-train-steps`. Verified end-to-end on a scratch run (steps 1-6, kill,
resume with `--resume-from-checkpoint latest --max-train-steps 10`,
continued cleanly at step 7 through 10, no errors).

### Next Actions

1. Launch the longer (5-10x step count) training run to test the
   sudden-convergence hypothesis, now with resume support so it can be
   stopped/restarted across weekday sessions without losing progress.
2. Resolve the `checkpoints/README.md` / `shape1/*.pth` git-tracking
   question (remove from tracking now that `checkpoints/` is a symlink, or
   restore them another way) — not urgent, doesn't block training.

**Corrected disk relocation + long-run launch prepared (same day, later).**
The single-symlink approach above broke `git add` for
`checkpoints/README.md` / `checkpoints/shape1/*.pth` (git refuses to add
through a symlinked directory) -- and `checkpoints/README.md` itself
documents that those are intentional force-add exceptions to the
`checkpoints/*` gitignore rule, so silently dropping them from tracking
would violate stated policy. Fixed properly: `checkpoints/` is a real
local directory again with `README.md`/`shape1/` restored as real
git-tracked files; every other (large, untracked) subdirectory is
individually symlinked to `~/disk/lineart_checkpoints/<name>`. `git
status` is clean.

Found and worked around a real gotcha with this approach: `mkdir -p` /
`os.makedirs(exist_ok=True)` on a *new* subdirectory name (no existing
symlink) just creates a real directory on the root disk -- it does not
transparently redirect through a not-yet-existing symlink target.
Pre-created `~/disk/lineart_checkpoints/controlnet_koma_direction4_longrun_20260803/`
and symlinked `checkpoints/controlnet_koma_direction4_longrun_20260803`
to it ahead of time so the scheduled run lands on `~/disk` from its first
write; verified with `os.makedirs(exist_ok=True)` directly. See
[[disk_layout]] memory for the general procedure to follow before any
future new checkpoint run.

Per user request (weekdays free for long GPU runs, weekends reserved for
other PC use; today is Friday evening, next weekday start is Monday
2026-08-03 00:00 JST): scheduled `experiments/run_controlnet_direction4_longrun_20260803.sh`
via a one-shot system crontab entry (`0 0 3 8 *`, self-removing on fire)
rather than the in-session `CronCreate` tool, since `CronCreate` jobs are
session-only and would be lost if this Claude Code session ends before
Monday. The launcher runs `train_controlnet.py --max-train-steps 18600`
(10x the first run, ~19h at the observed ~3.7s/step) with
`--resume-from-checkpoint latest` (safe to re-invoke to extend the run
later), `--save-steps 300` (resume-state safety net) and
`--eval-snapshot-steps 1860` (10 accumulating eval snapshots across the
run, ~14GB total), and notifies via the configured ntfy webhook on
completion. Verified GPU/CUDA is reachable under a minimal cron-like
environment (`env -i` test) before relying on this.

### Next Actions

1. Nothing to do until the Monday 2026-08-03 00:00 JST cron fires; then
   monitor `logs/controlnet_koma_direction4_longrun_20260803.log` and use
   the `--eval-snapshot-steps` snapshots with `scripts/infer_controlnet.py`
   to check whether/when the "sudden convergence" signal (tighter
   adherence to the rough conditioning) appears.
2. Resolve the `checkpoints/README.md` / `shape1/*.pth` git-tracking
   question -- now resolved (see above), no longer open.

**Per-tile WD14 auto-tagging for Direction 4 captions, and orphaned rough/line
cleanup (2026-07-31, later).** User asked whether spending the weekend on
tagging (for either Direction 4 conditioning or the parked CNN/GAN MoE
track) would help. Recommendation given: Direction 4 conditioning tags are
the more directly useful lever right now, since the just-diagnosed
hallucination problem (SD1.5 anime prior dominating over the rough
conditioning) is plausibly worsened by the fixed caption giving the model
zero per-tile signal to distinguish tiles. Installed `onnxruntime` +
downloaded `SmilingWolf/wd-v1-4-moat-tagger-v2` (~312MB ONNX,
`~/disk/checkpoint/wd14_tagger/`) and wrote `scripts/tag_wd14.py`.
Spot-checked on 6 sample tiles: tagging the **line** (clean target) tiles
gives good, differentiated danbooru-style tags (1girl/multiple_girls,
expression, hair length, etc.); tagging the **rough** tiles is noisier and
sometimes contradicts the line tagging on the same content (e.g. detected
"no_humans" on a rough tile where the line version correctly got
"1girl, solo"), so captions should come from the line tiles, matching the
existing convention that the caption describes the diffusion target, not
the ControlNet conditioning image.

User approved tagging all 1489 `combined_koma_20260729` line tiles and
wiring per-tile captions into training. Launched
`scripts/tag_wd14.py --file-list valid_train_combined_koma_20260729.txt
--image-dir train/line_combined_koma_20260729 --output-csv
dataset/pairs_480/captions_combined_koma_20260729_wd14.csv` in the
background (CPU-only, ~1489 tiles at ~1.2-1.3s/tile once warmed up, took
noticeably longer than the ~32min estimate due to CPU contention with
concurrent smoke tests). Extended `scripts/train_controlnet.py` with
`--caption-csv` (falls back to the existing fixed `--caption` for any tile
missing from the CSV; per-tile `input_ids` now flow through the dataset
and `encoder_hidden_states` is computed per-batch inside the training loop
instead of precomputed once globally, since captions are no longer
uniform). Verified the default fixed-caption path still works unchanged
(regression check, since the scheduled Monday long run does not pass
`--caption-csv` and must not be affected).

Separately, investigated the user's observation that `dataset/pairs_480/train/rough`
(17,950 files total) is dominated by `housei`/`ako5`-prefixed files
(16,446 of them) accumulated across the project's entire history, not
specific to the current `combined_koma_20260729` set (confirmed zero
filename overlap with the old `std15`-family manifests). User asked about
deleting unreferenced ("orphaned") ones. **First pass had a real bug**:
searched only `valid_train_*.txt` (glob), which misses the base
`valid_train.txt` (no underscore before `.txt`) -- this file is still the
default `--file-list` for `scripts/train.py` and `scripts/train_gan.py`
and was not caught by the glob. Archived and deleted 5,635 "orphans" using
the incomplete list, then a broader re-check (all `*.txt` under
`dataset/pairs_480` including subdirectories, plus `pair_metadata.csv` and
the router-feature-probe label CSV, plus a whole-repo grep for the
candidate names outside `dataset/`) found 2,725 of those were in fact
referenced by `valid_train.txt`. **Immediately restored everything from
the archive** (verified byte-identical via checksum spot-checks both
before deleting and again before re-deleting) and redid the analysis
properly: true orphan count is 2,910 (not 5,635), archived to
`dataset/archive/orphaned_housei_ako5_rough_line_20260731.tar.gz` (14MB
compressed, verified via tar member count + checksum spot-checks before
and after deletion) and removed from
`dataset/pairs_480/train/{rough,line}`. Final sanity check: cross-referenced
every name in every `valid_train*.txt` manifest against the post-deletion
directory listing -- 4,946 were already missing beforehand (pre-existing,
unrelated manifest/data drift, not caused by this cleanup) and zero of the
deleted 2,910 orphans were referenced by anything. Net effect: freed ~115MB
of disk (rough+line combined) -- not disk-pressure-relevant at current
64GB/313GB free, this was purely a tidiness cleanup, reversible via the
archive.

**Process lesson:** when computing "is X referenced anywhere" for a
destructive operation, glob patterns like `valid_train_*.txt` are not
equivalent to "all manifests with this naming family" -- verify against a
`find`/`ls` listing of the actual files present, not an assumed pattern,
especially before deleting. Caught this time via checksum-verified
archive-before-delete (so recovery was immediate and lossless) plus a
mandatory broader re-check before the second (final) deletion pass.

**WD14 tagging run completed (2026-08-01, after a mid-run restart).** The
first attempt at the full 1489-tile batch ran ~103+ min against a ~32 min
estimate with no visible progress output (stdout fully buffered) and no
clear hang/crash signal; killed it and restarted after two fixes: (1)
capped the onnxruntime session to `intra_op_num_threads=4` /
`inter_op_num_threads=1` (previously unbounded, 18 threads observed on an
8-core machine), (2) switched to incremental per-row CSV writes with
`flush=True` progress logging every 20 images instead of collecting
everything in memory and writing once at the end. The restarted run still
took ~7281s (~121 min) at a stable 4.89s/img -- slower than the original
clean-machine benchmark (~1.2-1.3s/img), and `idle_inject` kernel threads
observed in `ps` at the time pointed to sustained CPU thermal throttling
(`uptime` load average 7-9 on an 8-core box) as the likely cause, not a
bug in the tagging script itself. Completed cleanly: 1489/1489 rows
written to `dataset/pairs_480/captions_combined_koma_20260729_wd14.csv`,
process exited normally.

Smoke-tested `scripts/train_controlnet.py --caption-csv
dataset/pairs_480/captions_combined_koma_20260729_wd14.csv` (6 steps,
scratch output-dir): loaded all 1489 per-tile captions (zero fallback to
the fixed caption), trained without errors. Also re-verified the default
fixed-caption path is unaffected (the scheduled Monday
2026-08-03 00:00 JST long run, `experiments/run_controlnet_direction4_longrun_20260803.sh`,
does not pass `--caption-csv` and was not touched by this work).

### Next Actions

1. Per-tile captions are ready but not yet used in any real training run
   -- decide whether/when to run a real ControlNet training pass with
   `--caption-csv` (e.g. after the Monday long run's fixed-caption result
   is in, to isolate the two variables) rather than conflating both
   changes in one run.
2. Separately recorded and deferred (see
   [[project_architecture_direction_order]] equivalent section below):
   revisit a plain-regression (no adversarial loss) CNN recipe on
   `combined_koma_20260729` after Direction 4 settles, motivated by
   `notebooks/gen_lineart.ipynb` showing the pre-leak-fix era model used
   `BCE+L1+edge_loss` with no GAN term and reportedly produced more
   line-art-like (less soft/marbled) output than the current adopted
   `lucy_mild_aux_msgan`.
3. New raw manuscript data `dataset_4th.zip` (33 pages, same artist as the
   existing 5 sources per user confirmation, CLIP-STUDIO-layer-derived
   line/sketch pairs with a manifest) has arrived but is not yet run
   through the koma-panel extraction pipeline -- not urgent per the
   earlier discussion (current bottleneck is diagnosed as undertraining,
   not data scarcity), but ready whenever it's needed.

## 2026-08-04: results/ Cleanup -- New `results/lessons/` Folder, ControlNet Raw-Output Trim

Per user direction (continuing the `cleanup-refiner` worktree's own
`results/` cleanup the same session), pruned redundant per-sample output
here too, and introduced `results/lessons/` as a curated folder for
outputs a current doc explicitly cites as evidence for a finding --
distinct from `results/archive/` (old/historical, audit-only). See
`doc/RESULTS.md` ("`results/lessons/` (2026-08-04)") for the naming
rationale (`lessons/` chosen over `artifacts/`, which wouldn't distinguish
itself from everything else in `results/`).

Moved into `results/lessons/` (all cited by path in
`doc/architecture_decisions.md`'s "単段直接回帰" section, updated in the
same edit so no reference went stale): the 3-epoch/100-epoch/dense-28-epoch/
200-epoch direct-regression montages, both epoch-trajectory montages
(200ep and finegrid), the rough-fidelity-vs-binarization crop comparison,
and their metrics CSVs. Deleted the now-redundant raw per-checkpoint
output directories behind them (each montage already aggregates what
those held).

**ControlNet/Direction 4 outputs were explicitly NOT moved into
`lessons/`** -- per user clarification, that work is a different research
axis (faithfulness/hallucination, not stroke continuity), so it doesn't
belong in a continuity-lesson folder. User's explicit call: keep the
ControlNet montages (`compare_controlnet_koma_direction4_20260731.png`,
`_sweep.png`, `compare_controlnet_koma_direction4_longrun_20260803.png`)
on disk exactly where they are -- they're this branch's own primary
research record, cited directly from `doc/architecture_decisions.md`'s
Direction 4 section, not incidental clutter. Only deleted the genuinely
redundant raw per-sample directories already aggregated into those
montages (the base run's output dir, the long-run's output dir, and all 5
conditioning/guidance-scale sweep variant dirs) -- none of those raw dirs
were cited by path anywhere.

`results/` total size effectively unchanged (2.5G before/after -- the
deleted directories were a small fraction of the total; most of the bulk
is unrelated per-source koma-extraction and other-family experiment
output, out of scope for this pass per user direction).

## 2026-08-04/05: ControlNet 10x Long-Run Verdict -- Undertraining Hypothesis Rejected

The `controlnet_koma_direction4_longrun_20260803` run (18,600 steps, 10x
the original 1860-step probe, fired via crontab 2026-08-03 00:00 JST)
completed and was inferred/evaluated the same day (18:50-18:56), but the
result was discussed verbally and never written back into
`doc/architecture_decisions.md`'s Direction 4 entry -- caught and fixed
this session. Recording the conclusion here too since it was missing.

Quantitative result barely moved versus the original 1860-step run:
F1@2px 0.202 -> 0.216, ink_ratio 6.36 -> 5.93 (baseline CNN+GAN
`combined_koma_lucy_mild_msgan_20260729`: F1 0.418, ink_ratio 1.56).
User visually reviewed `results/compare_controlnet_koma_direction4_longrun_20260803.png`
and confirmed the hallucination behavior (confident but rough-unrelated
plausible content) did not resolve with 10x more steps. This rejects the
"undertraining / sudden convergence phenomenon" diagnosis from the
original 1860-step entry -- more steps alone is not the fix. This
specific step-count axis is shelved; if Direction 4 is revisited, the
next candidate variables are data volume, conditioning method, or the
already-prepared but unused per-tile WD14 captions (`--caption-csv`),
not further training-length increases.

## 2026-08-04/05: New `diffusion` Branch -- Domain Generation Instead Of Conversion

Branched off `diffusion-controlnet` at `e5f72f6` (right after the 10x
long-run verdict above) rather than continuing on it or renaming it in
place, so the ControlNet conditional-translation work stays intact as
its own historical record.

User's reframing of the diffusion direction going forward: set the
rough-to-line-art *conversion* task aside for now. Instead, focus on
training/generation quality of the rough domain and the line-art domain
*separately and unconditionally* (no pairing) -- i.e. can a diffusion
model be adapted to genuinely understand what these rough sketches look
like, and separately what this line art looks like, before attempting
any cross-domain conditional task again. This directly motivates the
concurrent unpaired-rough-pool cleanup work (see main branch's
`dataset/unpaired_rough_candidates/`, tasks: filter out finished-ink
contamination, then visual QC) -- that pool is training material for
this domain-only generation goal, not ControlNet conditioning material
as originally framed when it was gathered.

Not yet started: no training code for unconditional/domain-only
generation exists yet on this branch (`scripts/train_controlnet.py` is
conditional-only). First concrete step is the rough-pool cleanup
(filter + QC), then deciding the actual training mechanism (LoRA/
DreamBooth-style fine-tune vs. something else) once clean domain data is
in hand.

## 2026-08-05: Unpaired-Rough Pool -- No Finished-Ink Contamination Found

Attempted an automated filter for the "finished ink mistakenly labeled as
rough" contamination the pool-gathering script's docstring flagged as an
open risk (`dataset/unpaired_rough_candidates/`, 2014 tiles across
ako5ver2/fitness/gakuen/hamlabi/housei). Two heuristics tried, both
inconclusive on visual follow-up:

1. Rough-side dark-pixel ratio (fraction of drawn pixels below a
   confident-black threshold, vs. the graphite-midtone majority expected
   in genuine pencil rough). Top outliers (up to 0.74) visually inspected
   (contact sheet) turned out to be legitimate confident/thick pencil-pen
   strokes or near-empty tiles, not finished ink.
2. Rough-vs-line pixel correlation, for the 4 sources with both saved
   (1345 tiles; ako5ver2 has rough-only). A pipeline bug duplicating the
   finished line art into the rough slot would show as near-identical
   pairs. Highest-correlation pairs (up to 0.85) turned out to be simple
   tiles with a single confident stroke that happens to closely match its
   own line counterpart -- not duplication. Exact byte-identical
   rough/line file check: 0 of 1345.

Fell back to direct visual QC instead of further metric engineering,
since neither heuristic cleanly separated a contamination cluster from
legitimate content. New tool
`tools/pair_extraction/make_unpaired_rough_qc_sheet.py` draws a
stratified-random contact sheet (proportional to each source's pool
size, thumbnail resolution -- sufficient for this specific "graphite vs.
finished ink" gross visual judgment, unlike fine positional
rough/line-correspondence review which needs full tile resolution):
`results/qc_unpaired_rough_candidates_20260804.png`, 199 of 2014 tiles
(ako5ver2 66, fitness 50, gakuen 25, hamlabi 20, housei 38).

User reviewed the full sheet: every tile reads as genuine graphite
pencil rough (construction lines, gray shading, multiple stroke passes)
across all 5 sources -- no finished-ink contamination visible anywhere
in the sample. **Conclusion: the pool is clean as gathered; no filter
needed.** Both the filter task and the QC task are closed on this
finding. The pool is ready to use as rough-domain training material for
this branch's domain-only generation direction.

## 2026-08-04 (later): Domain-Only LoRA Training Launched

First concrete step on the domain-generation-quality direction. Design
discussion with the user settled: same base checkpoint as the ControlNet
work (`AOM3A1B_orangemixs.safetensors`, frozen), LoRA injected into the
UNet's attention layers only (peft, rank 16, `to_k`/`to_q`/`to_v`/
`to_out.0`), one LoRA per domain trained via plain unconditional txt2img
diffusion loss (no conditioning image, no pairing) from a fixed trigger
caption per domain. Verification/sampling uses a plain
`StableDiffusionPipeline` + `pipe.load_lora_weights(...)`, matching how
`scripts/infer_controlnet.py` already loads the base checkpoint, just
without the ControlNet residual input. Explicitly scoped as a stepping
stone, not the end goal: if both domains genuinely come through, the
natural next experiment is img2img/SDEdit-style translation (rough
latent + partial noise, denoise with the line-domain LoRA) as a
fundamentally different mechanism from ControlNet's conditioning-residual
approach, which might sidestep the hallucination failure mode -- but
that's deferred until domain quality itself is confirmed.

New code: `scripts/train_domain_lora.py` (mirrors `train_controlnet.py`'s
model-loading/Accelerator/resume-state conventions, with the ControlNet
network and paired dataset removed -- single `--image-dirs` pooled with
no pairing key needed since there's no conditioning target),
`scripts/sample_domain_lora.py` (txt2img sampling + contact-sheet
montage for visual QC).

Data: line domain uses the existing 1489-tile
`dataset/pairs_480/train/line_combined_koma_20260729`. Rough domain
turned out much larger than assumed --
`dataset/pairs_480/train/rough` is the *shared cross-source* rough pool
(19,454 images, not scoped to combined_koma) plus the newly-confirmed-
clean `dataset/unpaired_rough_candidates/*/rough` (2,014) = 21,468
images total. Pairing/alignment constraints that limited the paired
training set don't apply here, so the full pool is valid domain
material.

Smoke-tested `train_domain_lora.py` (30 steps, line domain): trains and
saves correctly, ~3.0-3.2s/step (batch 2, grad-accum 4, 512px, RTX 3060
12GB) -- somewhat faster than ControlNet's 3.64s/step, as expected
(no second network's forward/backward). Given the large rough/line pool
size mismatch (~14x), matched epoch counts would be very time-unbalanced;
settled on line=10 epochs (~1860 steps, ~1.6h) and rough=4 epochs
(~10,732 steps, ~9.3h) as a first-look domain-quality check, not a
final-convergence run.

Launched `experiments/run_domain_lora_queue_20260804.sh` (line then
rough sequentially, each followed by 16-sample contact-sheet generation
via `sample_domain_lora.py` and an ntfy notification) via
`nohup ... & disown`. Verified detached: queue script PID 1270658,
PPID=1, no controlling terminal; nested training process PID 1270670
confirmed running with the correct line-domain arguments.

Next actions: once both complete (~11h total), review
`results/domain_lora_line_20260804/contact_sheet_domain_lora_line_20260804.png`
and the rough-domain equivalent -- does unconditional generation from
the trigger caption actually look like this project's line art / rough
sketch style, or does the LoRA fail to shift the base model's style
meaningfully at this rank/step count? That verdict decides whether the
SDEdit-style translation follow-up is worth attempting.

## 2026-08-05: Line-Domain LoRA Reviewed; New GT-Free Line-Art-Profile Tool

The line-domain LoRA (`domain_lora_line_20260804`) finished overnight
(1860 steps, loss 0.0128); the rough-domain LoRA is still training as of
this entry (~50% through its 10,732-step budget).

Visual review of the line-domain contact sheet
(`results/domain_lora_line_20260804/contact_sheet_domain_lora_line_20260804.png`,
16 unconditional txt2img samples): crisp, confident, fully binary anime
linework -- no soft/marbled texture, the CNN+GAN family's persistent
failure mode. Promising first signal that a domain-only diffusion LoRA
can learn "line-art-ness" as a style from unpaired data alone.

Discussed with the user what "line-art-ness" should even mean here,
given this project's repeated pattern of a single optimized indicator
hiding collapse on an unchecked axis (soft/marbled cleanup family, wobbly
single-stage regression, ControlNet hallucination -- see
`doc/architecture_decisions.md`'s Direction 4 entry and the direct-regression
section). Conclusion: treat it as a multi-axis structural profile, not one
scalar, and since domain-only generation has no GT/pairing, every axis
must be computable from a single image alone.

Built `tools/evaluation/measure_lineart_profile.py` (new): reuses
`evaluate_stroke_stability.py`'s skeleton/component-length code and
`tile_region_manifest_480.py`'s `line_width_stats`/`long_line_ratio`/
`orientation_entropy`, plus new intensity-histogram and "faint-pixel
locality" metrics. Full metric glossary and interpretation notes added to
`doc/architecture_decisions.md` ("線画らしさプロファイル指標").

**Calibration bug caught and fixed during first use:** the initial faint-
locality design anchored "confident ink" at an arbitrary `<30` threshold
and called the 30-220 band "midtone". A quick check of real ink-pixel
values (`gray < 128`, this project's standard ink threshold) showed the
median is 73 -- so most genuine ink was being miscounted as "midtone"
purely from the uncalibrated anchor, making real GT tiles look ~90%+
midtone-within-drawn-area, which is meaningless. Fixed by anchoring
locality on the actual ink threshold (128) instead of an arbitrary
stricter one; `deep_black_ratio` (`<30`) is kept only as a separate,
non-locality diagnostic.

First real comparison (koma_ref: full `combined_koma_20260729` 1489
line tiles as the reference distribution; lora_line: the 17-sample
contact sheet), full stats:
`results/lineart_profile_koma_ref_vs_domain_lora_line_20260805.csv`.
Medians:

| metric | koma_ref | lora_line |
|---|---:|---:|
| long_component_ratio | 0.814 | 0.781 |
| components_per_1k_ink_px | 10.85 | 8.38 |
| line_width_p50 | 3.82px | 5.73px |
| width_consistency (p95/p50) | 2.59 | 3.06 |
| faint_of_drawn_ratio | 0.384 | 0.445 |
| **faint_near_ink_ratio** | **0.899** | **0.663** |
| faint_mean_dist_to_ink | 2.40px | 3.27px |

Interpretation: stroke continuity is comparable or better than the
reference (no wobble/fragmentation regression like the single-stage
direct-regression case). Line width is clearly thicker and less uniform
than real koma tiles -- expected, matches the visually "bold" contact
sheet impression. The standout divergence is `faint_near_ink_ratio`: in
real line art, ~90% of ambiguous/faint pixels sit within 3px of actual
ink (a thin anti-aliasing halo hugging confident strokes); in the LoRA
samples, only ~66% do -- more of its faint pixels are floating free of
any confident stroke rather than hugging one. This is the kind of
specific, non-obvious axis this tool was built to surface instead of a
single pass/fail verdict.

Caveat: `ink_ratio`/`background_ratio` differ a lot between the two sets
(LoRA ~0.26 vs ref ~0.03) but this is confounded by framing -- the LoRA
contact-sheet samples are all face/eye close-ups with little blank
background, unlike the broader mix of full-panel real tiles -- not a
direct line-art-ness signal by itself.

### Next Actions

1. Once the rough-domain LoRA finishes, run the same profile tool against
   its contact sheet and review both visually and via this tool.
2. Consider whether `faint_near_ink_ratio` divergence is specific to this
   LoRA/rank/step-count combination or a more general property of
   diffusion-sampled line art; no controlled comparison across
   configurations exists yet.
3. Domain LoRA training itself is still a stepping stone (per the
   2026-08-04 entry above) -- the SDEdit-style translation follow-up
   remains the next real milestone once both domains are judged
   sufficiently "line-art-like"/"rough-like" by this profile plus visual
   review.

## 2026-08-05 (later): Rough-Domain LoRA Reviewed -- Content Mismatch, Not A Style Problem; Two Isolation Runs Launched

`domain_lora_rough_20260804` finished (10,732 steps, 4 epochs over the full
21,468-image cross-source rough pool, loss 0.2254). Contact sheet:
`results/domain_lora_rough_20260804/contact_sheet_domain_lora_rough_20260804.png`.

**Visual review, user's judgment:** looks like dense, repetitive
line-practice scribbles -- almost every one of the 16 samples is dominated
by parallel hatching/cross-hatch strokes, with only 1-2 tiles showing any
recognizable figure/face content. Stroke darkness itself is borderline
acceptable as pencil (some strokes are pen-dark, but that's within range
for a strong pencil pass). The real problem the user flagged: **content and
composition barely resemble the actual training tiles** -- this is a
content/composition failure, not a texture/darkness failure.

**Quantified with `measure_lineart_profile.py`** against the real rough
pool as reference (`dataset/unpaired_rough_candidates/*/rough`, n=1354
sampled, vs. the 16 LoRA samples):
`results/lineart_profile_rough_ref_vs_domain_lora_rough_20260805.csv`.
Medians:

| metric | rough_ref (real) | lora_rough | read |
|---|---:|---:|---|
| ink_ratio | 0.018 | 0.307 | 17x more ink than real rough pages |
| background_ratio | 0.90 | 0.51 | half the canvas covered vs. mostly blank paper |
| faint_of_drawn_ratio | 0.79 | 0.39 | real rough is mostly faint graphite; LoRA draws confidently |
| line_width_p50 | 2.74px | 4.23px | notably thicker strokes |
| long_line_ratio | 0.46 | 0.84 | dominated by long straight-ish lines |
| orientation_entropy | 0.79 | 0.73 | fewer distinct stroke directions (parallel-hatch signature) |

Confirms the visual read quantitatively: high ink density, long straight
strokes, low orientation diversity is consistent with convergence onto one
dominant local texture pattern (parallel hatching) rather than the full
rough-sketch content distribution (construction lines, faces, panel
layouts).

**Two candidate causes discussed, deliberately not conflated:**

1. **Epoch/exposure deficit.** Confirmed from each run's own log line:
   line domain = 1,489 images, 186 steps/epoch, 10 epochs = 10 passes/image.
   Rough domain = 21,468 images, 2,683 steps/epoch, 4 epochs = 4
   passes/image -- 2.5x fewer passes than line. Noted as a real but
   possibly partial explanation: classic diffusion undertraining usually
   shows as *weak* conditioning/style transfer (as with the ControlNet
   1860-step probe), not a *strong, consistent* convergence onto one
   specific texture, which is what's observed here.
2. **Data composition skew (found while investigating).**
   `dataset/pairs_480/train/rough` (19,454 of the pool's 21,468 images) is
   the legacy cross-source pool accumulated over the project's whole
   history for the CNN aux/atari generator, not curated for domain-LoRA
   diversity. Prefix breakdown: `ako*` 9,332 (48%) + `housei*` 5,070 (26%)
   = 74% from just two legacy sources, plus 2,603 images (13.5%) from
   `*komadense` variants -- the same duplicate-overlap-loosened retiling
   family the project already found reduces effective content diversity in
   a different training (2026-08-01 direct-unet dense-28ep result, this
   file above). The curated line-domain pool
   (`line_combined_koma_20260729`) has no equivalent skew.

**Decision (explicit user direction):** run both isolation experiments
rather than guess, starting with the cheaper one first:

- `experiments/run_domain_lora_roughclean_20260805.sh` (data-composition
  isolation): same rank/caption/epoch-count as the line-domain run (10
  epochs), but trained only on the 2026-08-05-confirmed-clean
  `dataset/unpaired_rough_candidates/*/rough` pool (2,014 images, balanced
  across all 5 sources, no legacy/dense duplication) -- 2,510 steps, ~2.5-3h.
  Launched detached (`nohup ... & disown`), smoke-tested first (6-step dry
  run into a scratch dir).
- `experiments/run_domain_lora_roughfull_e10_20260805.sh` (epoch-count
  isolation): resumes `domain_lora_rough_20260804`'s `resume_state` and
  continues on the full 21,468-image pool up to the line domain's 10-epoch
  equivalent (26,830 total steps, ~16,098 more from here, ~16-17h).
  Writes to a new output dir so the original 4-epoch checkpoint/contact
  sheet stays intact for comparison. Verified via smoke test that resume
  correctly picks up at step 10,732 without touching the original
  checkpoint dir.

Since the roughclean run (~3h) may finish after the user is asleep,
explicit direction: chain straight into the roughfull run on completion,
with no interactive review gate in between. Built
`experiments/run_domain_lora_chain_roughclean_then_roughfull_20260805.sh`,
a detached watcher (`nohup ... & disown`, verified `PPID=1`) that polls
`logs/domain_lora_roughclean_20260805.done` every 60s and launches
`run_domain_lora_roughfull_e10_20260805.sh` the moment it appears (i.e.
right after roughclean's training *and* contact-sheet generation both
complete). Safety net: if roughclean's process disappears without ever
writing that `.done` marker (crash), the watcher does not auto-launch
roughfull -- logs the failure and exits, so a broken premise doesn't waste
the whole overnight/daytime GPU window. All three scripts (`roughclean`,
`roughfull_e10`, and the chain watcher) still send the existing ntfy
notification on their own completion.

### Next Actions

1. Review `domain_lora_roughclean_20260805`'s contact sheet + profile-tool
   comparison once it finishes tonight -- if it looks like real rough
   content (unlike the full-pool run), that confirms data-composition skew
   as the primary cause.
2. Review `domain_lora_roughfull_e10_20260805`'s result the next
   morning/day (should auto-launch unattended right after #1, per the
   chain watcher above) -- if *this* fixes the content-mismatch problem
   instead, epoch count was the primary cause; if neither fixes it cleanly,
   both variables may matter together, or a third factor (fixed single
   caption giving zero content-disambiguation signal across a diverse
   pool, or rank-16 capacity) needs to be considered next.
3. Once both isolation results are in, update `doc/architecture_decisions.md`
   with the resolved verdict (currently only the line-domain LoRA and the
   original 4-epoch rough-domain LoRA are recorded there).
4. SDEdit-style translation follow-up remains blocked on both domains
   being judged sufficiently "line-art-like"/"rough-like" -- unchanged from
   the prior entry.

## 2026-08-06: Rough-Domain Isolation Results In -- Both Miss The Real Failure; Root Cause Is Missing Content Specification In The Caption

Both isolation runs from the prior entry finished (chain watcher worked as
designed): `domain_lora_roughclean_20260805` (data-composition isolation,
2,014-image clean pool, 10 epochs, completed 2026-08-06T00:15) then
`domain_lora_roughfull_e10_20260805` (epoch-count isolation, resumed the
original 4-epoch run to a 10-epoch-equivalent 26,830 steps on the full
21,468-image pool, completed 2026-08-06T18:48).

### Stroke-Metric Comparison: Data Composition Beat Epoch Count, But Both Missed The Point

`results/lineart_profile_rough_isolation_compare_20260806.csv` (`rough_ref`
n=2014 vs each LoRA n=16-17, medians):

| metric | rough_ref | lora_rough_e4 | roughclean_e10 | roughfull_e10 | closer to ref |
|---|---:|---:|---:|---:|---|
| ink_ratio | 0.020 | 0.305 | 0.253 | 0.284 | roughclean |
| background_ratio | 0.902 | 0.504 | 0.426 | 0.499 | roughfull (slight) |
| faint_of_drawn_ratio | 0.784 | 0.405 | 0.571 | 0.416 | roughclean |
| line_width_p50 | 2.74 | 3.82 | 3.82 | 5.73 | roughclean |
| long_line_ratio | 0.477 | 0.839 | 0.744 | 0.862 | roughclean |
| orientation_entropy | 0.790 | 0.723 | 0.783 | 0.728 | roughclean (near match) |
| components_per_1k_ink_px | 37.2 | 15.1 | 20.7 | 9.4 | roughclean |

7 of 8 stroke-level axes favored `roughclean` (data-composition isolation)
over `roughfull_e10` (epoch-count isolation) -- the legacy/dense-heavy
74%-of-pool skew identified in the prior entry really was hurting
stroke-level realism, more than the epoch deficit was.

But a human visual read of both contact sheets (user, before any of this
was quantified) judged it the opposite way on the dimension that actually
matters: "ep10 の方が少しだけ何か形らしいものを描こうとしてるようにみえる"
(`roughfull_e10` looks slightly more like it's attempting recognizable
shapes) -- both still read as overwhelmingly parallel-hatch texture, not
line art. This flagged that the profile tool's existing axes (all
stroke-level: width, faintness, continuity, orientation) cannot see the
dimension the user was actually judging: whether the image has real
macro composition at all, versus being a flat repeating texture that
merely has plausible-looking individual strokes.

### New Metric: `grid_ink_cv` / `blank_cell_fraction` (Structural Collapse, Not Stroke Quality)

Added to `tools/evaluation/measure_lineart_profile.py`
(`grid_heterogeneity()`): divide the image into an 8x8 grid, compute
per-cell ink_ratio, report the coefficient of variation across cells
(`grid_ink_cv`) and the fraction of near-empty "paper" cells
(`blank_cell_fraction`, threshold 1% local ink). Rationale: a page that is
one repeating hatch texture wall-to-wall can score fine on every stroke
axis while being content-free; real sketches mix near-blank paper with
locally dense subject regions (high variance), a uniform texture does not.

Result, run against all three rough LoRA variants plus `rough_ref`
(`results/lineart_profile_rough_isolation_compare_20260806.csv`, medians):

| metric | rough_ref | lora_rough_e4 | roughclean_e10 | roughfull_e10 |
|---|---:|---:|---:|---:|
| grid_ink_cv | 2.03 | 0.39 | 0.43 | 0.47 |
| blank_cell_fraction | **0.72** | 0.00 | 0.016 | 0.016 |

Real rough pages are ~72% near-blank grid cells with sharp local density
contrast (CV 2.0); all three LoRA variants are 0-1.6% blank with CV ~0.4 --
roughly a fifth of real. On this axis all three are equally collapsed;
neither isolation variable (data composition or epoch count) touched the
actual failure the user was pointing at. This directly explains why the
stroke-metric table above and the visual read diverged: they were
measuring different things, and the thing the user cared about had no
metric until this one.

### Root Cause Test: Caption Has Zero Content Specification

User's hypothesis: the fixed caption used for all 21,468 training images
(`"roughsketchstyle, pencil rough sketch, messy sketchy construction
lines, monochrome"`) never says *what* to draw, only *how* to render it --
so the model has no per-image signal for composition and falls back to
tiling a texture. Checked the training tiles themselves
(`dataset/unpaired_rough_candidates/{ako5ver2,hamlabi}/rough` samples):
confirmed they are tight single-character face/head crops with large
blank margins, i.e. real, specific content that the caption never
mentions.

Tested cheaply -- inference only, no retraining, existing
`domain_lora_roughclean_20260805/final` checkpoint:

1. **Motif added** (`"1girl, portrait, face closeup, ..."`,
   `results/domain_lora_roughclean_motiftest_20260806/`, 8 samples): all 8
   samples became recognizable anime-style faces (eyes, hair, expression)
   -- a dramatic, unambiguous visual break from the flat hatch texture.
   Hypothesis confirmed for content-recognizability.
2. **Full-body motif** (`"1girl, full body, standing pose, wide shot,
   ..."`, `results/domain_lora_roughclean_motiftest_fullbody_20260806/`):
   a recognizable standing figure appears in all 8 samples, but the
   surrounding canvas is still filled edge-to-edge with the same
   parallel-hatch texture -- `blank_cell_fraction` medians actually *lower*
   than the face test (0.008 vs 0.039). Content-specification and
   background-fill are separable problems; fixing one did not fix the
   other.
3. **Background-fill overcorrection**
   (`"... plain background, empty background ..."` +
   `--negative-caption "hatching, crosshatch, dense background lines,
   filled background"`,
   `results/domain_lora_roughclean_motiftest_fullbody_emptybg_20260806/`):
   overshot badly -- output became fully flat-shaded, finished-illustration
   anime renders with no line-art texture at all. Pushing the prompt hard
   enough against the LoRA's own domain overrides the rank-16 LoRA's grip
   on the rough-sketch style entirely, reverting to the base checkpoint's
   (AOM3A1B) native rendering prior.
4. **Hatch-phrase removal only** (dropped `"messy sketchy construction
   lines"` from the full-body caption, no background push,
   `results/domain_lora_roughclean_motiftest_fullbody_nohatchphrase_20260806/`):
   negligible change (`blank_cell_fraction` 0.008 -> 0.016,
   `grid_ink_cv` 0.469 -> 0.487, both still ~1/40 and ~1/4 of `rough_ref`
   respectively). Rules out that specific phrase as the driver -- the
   background-fill habit is baked into the LoRA weights (almost certainly
   from the training pool's own composition-heavy legacy tiles), not
   something caption wording at sampling time can undo.

### Conclusions

- The single fixed style-only caption used for rough-domain training is a
  real, confirmed root cause of the content/composition failure -- adding
  explicit motif words at *sampling* time (no retrain) fixes
  content-recognizability immediately.
- Canvas-filling/background-hatch is a separate failure mode, not fixed by
  motif specification and not undoable by prompt engineering at sampling
  time -- it is trained into the weights and needs to be addressed at
  training time (data composition and/or per-image captions), not
  inference time.
- Prompt engineering has reached its practical limit for this checkpoint:
  pushing further (explicit background suppression) breaks the LoRA's
  domain grip rather than fixing composition.
- This reframes both prior isolation runs: neither
  `domain_lora_roughclean_20260805` nor `domain_lora_roughfull_e10_20260805`
  addressed the actual dominant failure (structural/macro collapse); the
  stroke-metric wins for `roughclean` were real but on the wrong axis.

### Next Actions

1. Plan a per-image content-aware caption retrain of the rough domain --
   needs actual per-image content tags (motif/pose/framing), not one
   shared style caption across the whole pool. Likely needs a tagging
   pass (existing tagger/BLIP-style tool or heuristic) over the training
   images before retraining.
2. Investigate whether the background-fill habit traces to the same
   legacy/dense-tile composition skew flagged in the prior entry (`ako*`
   48% + `housei*` 26% + `*komadense` 13.5% of the full pool) -- if so, a
   retrain combining the clean-pool data composition *and* per-image
   captions may be needed together, not caption diversity alone.
3. Update `doc/architecture_decisions.md` with the resolved verdict for
   both isolation runs plus this caption-diagnosis finding (currently only
   the line-domain LoRA and the original 4-epoch rough-domain LoRA are
   recorded there).
4. SDEdit-style translation follow-up remains blocked on the rough domain
   being judged sufficiently "rough-like" -- unchanged from the prior
   entry, now additionally blocked on the content-specification fix above.

## 2026-08-06/07 (later): Line-Domain Style Fidelity -- Fidelity-Budget Policy, Base/Capacity/Caption/Scale/Trigger-Token Isolation Chain, Best Config Adopted

Starting observation (user, after reviewing rough-domain isolation results):
even where content became recognizable, output still reads as "the base
model's own way of drawing images with a faint style nudge from our
tiles" -- not genuinely close to our tiles, for both domains, and
specifically for line-domain: neither the art style nor the line quality
resembled the training tiles.

### Fidelity-Budget Policy

New standing policy, `doc/diffusion_fidelity_budget_policy.md`: treat up
to ~30% of generated output as attributable to the base checkpoint's own
habits as "under control," modeled on the observed real-world deviation
rate between this project's own rough/line pairs (rough ignored, redrawn,
or partially followed during inking -- a normal, non-degenerate part of
the production process, not a data-quality problem). Explicitly
provisional and axis-by-axis, not a single scalar gate; see the policy
file for the full rationale and the running per-axis exception list
(`deep_black_ratio` excluded from gating, `line_width_p50` accepted up to
an observed ceiling, `components_per_1k_ink_px` loosened to a 50% band,
and `blank_cell_fraction`/`grid_ink_cv`/`width_consistency` later
explicitly deprioritized by the user as not the axes that matter most for
this judgment).

### Isolation Chain (each cheap, smoke-tested, one variable at a time)

1. **Base checkpoint** (AOM3A1B vs. vanilla SD1.5 `v1-5-pruned-emaonly`,
   both rank16 attn-only, same 1489-image `line_combined_koma_20260729`
   pool/caption/10-epoch budget): SD1.5 won 7/8 stroke-level profile axes
   over AOM3A1B, but visually settled into a *different* strong prior --
   bold manga/comic finished-inking with heavy solid-black fill blocks
   (`deep_black_ratio` ~55x real) instead of the tiles' thin uniform
   single-pass linework. Progress on structure, not resolution.
2. **Capacity** (SD1.5 + rank16->32, attention-only -> +conv1/conv2/
   conv_shortcut/proj_in/proj_out, ~7.8x trainable params, `--lora-
   target-modules` added to `train_domain_lora.py` for this): regressed
   on every macro-structure axis (`blank_cell_fraction` 84%->100% dev,
   `grid_ink_cv` 56%->77% dev, `deep_black_ratio` 5450%->17650% dev) while
   marginally helping axes that were already passing. Visually collapsed
   into a third, unrelated strong prior -- high-contrast shonen-manga
   battle scenes, screentone-like patterns, dense action content bearing
   no resemblance to the training tiles. Conclusion: more LoRA capacity on
   a ~1500-image pool does not buy fidelity, it buys easier access to
   whatever strong attractor is nearest in the base model's broader
   pretraining. Rank16 attn-only SD1.5 stayed the reference config.
3. **Per-image captions** (`scripts/tag_wd14.py`, already built for the
   ControlNet Direction 4 work, reused here: WD14 tagger -> per-tile
   danbooru tags + style suffix, `train_domain_lora.py --caption-csv`):
   trained on per-image captions instead of one fixed caption for all
   1489 images. Sampling with the same old *generic* caption as always
   produced virtually no change (checkpoint hash differed, output
   statistics were within noise of the fixed-caption run) -- confirms
   training-time content captions alone do nothing without a matching
   content-specific prompt at sampling time. Re-sampled with an explicit
   motif prompt matching the trained tag vocabulary
   (`"1girl, solo, close-up, white_background, simple_background, ..."`):
   best line-domain result to that point -- `background_ratio` moved into
   the 30% budget for the first time (29.4% dev), `faint_near_ink_ratio`
   nearly matched real (1.5% dev). Traded off against `line_width_p50` and
   `deep_black_ratio` getting worse (attributed to the close-up framing
   itself), which is where the per-axis exceptions above came from.
4. **LoRA inference-time scale** (`--lora-scale` added to
   `sample_domain_lora.py`, `cross_attention_kwargs={"scale": X}`, no
   retrain needed): swept 1.0/1.3/1.4/1.5/1.6 on the per-image-caption
   checkpoint. Found a real Pareto trade-off, not a free improvement --
   every macro-structure axis moved closer to real as scale increased
   (at 1.6, `blank_cell_fraction`/`grid_ink_cv`/`components_per_1k_ink_px`
   all moved inside the 30% band), but visual content coherence degraded
   in lockstep -- faces/figures became unrecognizable abstract scribbles
   above ~scale1.4-1.5. A clean, textbook instance of this project's
   standing metric-vs-visual divergence warning: the structural axes
   reward "sparse, uneven ink," which an incoherent scribble satisfies
   just as well as a genuine sparse composition. User visual verdict:
   scale1.6 line *taste* is right but shape is unusable; scale1.3 still
   visually distant in style. Scale alone could not resolve the
   trade-off -- pointed at the LoRA weights themselves needing work.
5. **Rare trigger token, attempt 1** (`"sks style, monochrome, black and
   white"` replacing the *entire* style suffix, same per-image WD14 tags
   otherwise): regressed badly. At scale1.0 the weak LoRA influence let
   SD1.5's own strong prior for "1girl ... monochrome, black and white"
   dominate -- which turned out to mean photorealistic B&W portrait
   *photography*, not line art. At scale1.3/1.4, photographic and
   manga-linework elements collided incoherently within the same image,
   worse than every prior variant. Confirms `"manga panel, monochrome
   line art"` were not just baggage carrying the base model's unwanted
   habits -- they were necessary *domain*-framing keeping generation in
   line-art territory at all. Failed outputs/checkpoint deleted (all
   gitignored: `results/**`, `checkpoints/*`, `logs/`; no git impact).
6. **Rare trigger token, attempt 2** (`"sks style, monochrome line art,
   manga panel, black and white"` -- domain-framing words kept, only the
   specific style-execution phrase `"lineartstyle, clean linework"`
   swapped for the rare token): success. No photo-bleed at any scale
   tested (1.0/1.3/1.4). At scale1.4: `line_width_p50` 5.73px (50.0% dev,
   *under* the 93.4%-dev ceiling set by the per-image-caption run, i.e.
   thinner than before) while every stroke-shape axis stayed inside the
   30% budget (`faint_near_ink_ratio` 13.0%, `long_component_ratio` 7.8%,
   `long_line_ratio` 11.4%, `orientation_entropy` 6.3%) and -- critically,
   confirmed visually -- shape/content coherence did not collapse the way
   it did on the ordinary-caption checkpoint at the same scale. Reads as:
   swapping just the style-descriptor phrase for a token with no
   pre-existing meaning pushed the quality-vs-coherence trade-off frontier
   itself further out, rather than just moving along the old one.

### Decision

Adopted as current line-domain reference config: checkpoint
`checkpoints/domain_lora_line_sd15base_sksv2_20260807/final` (SD1.5 base,
rank16 attn-only, per-image WD14 tags + `"sks style, monochrome line art,
manga panel, black and white"` suffix), sampled with an explicit motif
prompt (`"1girl, solo, close-up, white_background, simple_background,
..."`) at `--lora-scale 1.4`. User confirmed the line quality at scale1.4
is sufficient to stop here for now.

### Next Actions

1. Update `doc/diffusion_fidelity_budget_policy.md`'s comparison table
   with the sksv2 scale1.4 numbers as the new reference (done in the same
   session as this entry).
2. Update `doc/architecture_decisions.md` with the resolved line-domain
   verdict -- still outstanding from the prior entry, now with a much
   longer isolation chain to summarize.
3. Apply the same isolation chain (base checkpoint already resolved;
   per-image captions + motif prompt + LoRA scale + rare trigger token
   with domain framing kept) to the **rough domain**, which stalled
   earlier at the caption-motif stage with the canvas-filling/background-
   hatch habit still baked into the weights and unresolved.
4. SDEdit-style translation follow-up remains blocked on both domains
   being judged sufficiently domain-like -- line domain is now much
   closer; rough domain still needs the rough-side equivalent of this
   chain (see #3).

## 2026-08-07/08: Rough-Domain Isolation Chain -- Same Recipe Applied Directly, Larger Improvement Than Line Domain

Applied the line-domain isolation chain's outcome to the rough domain in
one combined run rather than re-isolating each variable -- base checkpoint
and capacity were resolved generically on the line domain (SD1.5 rank16
attn-only beat AOM3A1B and rank32+conv, not domain-specific findings), so
this run combined the remaining four techniques directly: per-image WD14
captions, motif-prompt sampling, LoRA inference-scale sweep, and a
domain-word-preserving rare trigger token.

**Setup**: data = the 2,014-image clean rough pool (`dataset/
unpaired_rough_candidates/*/rough`, already established as the better
data composition in the 2026-08-06 isolation runs). Tagged with
`scripts/tag_wd14.py` (works fine on rough pencil sketches too --
`sketch`, `1girl`, `close-up`, `white_background` etc. picked up
correctly, not just on finished line art). Caption = per-image WD14 tags +
`"sks style, pencil rough sketch, monochrome"` -- domain-framing words
(`"pencil rough sketch, monochrome"`) kept per the line-domain lesson,
only the style-execution descriptor (`"roughsketchstyle, messy sketchy
construction lines"`) replaced by the rare token. Trained rank16
attn-only on SD1.5 base, 10 epochs (2,510 steps), then sampled at
`--lora-scale` 1.0/1.3/1.4 with an explicit motif prompt.

**Result**: the parallel-hatch texture collapse (the dominant, unresolved
failure from the 2026-08-05/06 entries) is gone. All three scales produce
recognizable faces with construction-line-style layered hair strokes,
matching the real rough pool's actual character (tight face/head crops,
graphite-like layered strokes) far better than anything in the earlier
isolation runs.

Quantified against `rough_ref` and the original collapsed baseline
(`domain_lora_rough_20260804`), `results/lineart_profile_rough_sksv2_
compare_20260808.csv`, medians:

| metric | rough_ref | orig_rough_e4 (collapsed) | sksv2 scale1.4 |
|---|---:|---:|---:|
| `width_consistency` | 2.70 | 7.3% dev | **3.4% dev** |
| `orientation_entropy` | 0.79 | 8.4% dev | **0.7% dev (near-exact)** |
| `long_component_ratio` | 0.44 | 65.1% dev | **11.3% dev (in budget)** |
| `background_ratio` | 0.90 | 44.2% dev | **32.2% dev (near budget)** |
| `components_per_1k_ink_px` | 37.2 | 59.3% dev | **32.7% dev (in 50% band)** |
| `grid_ink_cv` | 2.03 | 80.6% dev | 60.7% dev (improved) |
| `blank_cell_fraction` | 0.72 | 100% dev (zero blank cells) | 86.9% dev (improved, still far) |
| `faint_near_ink_ratio` | 0.44 | 83.7% dev | 45.1% dev (improved) |

Every axis moved closer to `rough_ref`, several into or near the 30%/50%
budget for the first time on this domain. The improvement margin is
larger than what the same recipe produced on the line domain, consistent
with the rough domain's starting point being further from real (total
texture collapse vs. "recognizable but stylistically distant").
`blank_cell_fraction`/`grid_ink_cv` remain the largest gaps, matching the
line-domain pattern (and already deprioritized there by user judgment).

User visual confirmation: "碓かに下絵の雰囲気はだいぶでてる" (the rough-sketch
atmosphere is definitely coming through now).

### Decision

Adopted as current rough-domain reference config: checkpoint
`checkpoints/domain_lora_rough_sd15base_sksv2_20260807/final` (SD1.5 base,
rank16 attn-only, per-image WD14 tags + `"sks style, pencil rough sketch,
monochrome"` suffix, trained on the 2,014-image clean pool), sampled with
an explicit motif prompt (`"1girl, solo, close-up, sketch, ..."`) at
`--lora-scale 1.4`.

### Next Actions

1. Update `doc/diffusion_fidelity_budget_policy.md` with this rough-domain
   result (done in the same session as this entry).
2. Update `doc/architecture_decisions.md` with both domains' resolved
   verdicts -- still outstanding.
3. Both domains now have a much-improved reference LoRA. Revisit whether
   the SDEdit-style translation follow-up (the original motivation for
   this whole domain-only-generation detour) is ready to resume.

### Cleanup

With both domains' review decisions now recorded above, deleted the
isolation chain's superseded intermediate outputs (per `doc/RESULTS.md`
cleanup policy: delete per-sample image dirs once review decisions are
recorded; the adopted configs and permanent baseline comparisons stay
referenced by exact path in `doc/diffusion_fidelity_budget_policy.md`).

- `results/`: 14 unreferenced `domain_lora_*` directories deleted (~86MB)
  -- every line-domain intermediate (base-checkpoint/capacity/caption
  isolation steps, scale-sweep variants at scale 1.0/1.3/1.5/1.6, the
  superseded sks-v1 photo-bleed failure already deleted earlier) and the
  rough-domain data-composition/epoch-count isolation runs. Kept: the
  original 2026-08-04 baselines, the 2026-08-06 rough motif-test chain,
  both domains' adopted scale1.4 montages, both domains' WD14 caption
  CSVs, and all `lineart_profile_*.csv` metrics (kept unconditionally per
  policy).
- `checkpoints/`: 5 superseded checkpoints deleted (~17GB) --
  `domain_lora_line_sd15base_20260806`,
  `domain_lora_line_sd15base_captiontags_20260807`,
  `domain_lora_line_sd15base_hicap_20260807`,
  `domain_lora_roughclean_20260805`, `domain_lora_roughfull_e10_20260805`.
  Kept: the two 2026-08-04 original baselines and the two adopted sksv2
  checkpoints. Disk free space: 25GB -> 42GB.

## 2026-08-08: ControlNet Hallucination Re-Diagnosed -- Not Data-Scale Alone, Our Own LoRA Was Actively Hurting It

With both domains' style fidelity resolved (prior entry), attempted to
reconnect to conditional rough->line conversion. Two cheap diagnostics on
our own custom-trained ControlNet
(`checkpoints/controlnet_koma_direction4_longrun_20260803`) came back
negative:

1. **Pair alignment quality** (`dataset/pairs_480/pair_metadata.csv`'s
   `alignment_quality` column: `same_coordinate`/`uncertain`/
   `rough_shifted` buckets, 20 tiles each, `locally_refined` bucket
   dropped -- its 123 rows' `rough_path`/`line_path` no longer exist on
   disk, a stale `kurip`-source manifest entry, not investigated further).
   Chamfer-to-GT (`edge_map`+`chamfer` from
   `tile_region_manifest_480.py`, truncate=20px) was flat across buckets
   (8.79 / 9.49 / 9.09 px median) -- hallucination severity does not
   correlate with alignment quality, confirmed visually too (even
   best-aligned tiles produced content unrelated to input).
2. **`--controlnet-conditioning-scale` sweep** (1.0/1.3/2.0, `same_
   coordinate` bucket, same 20 tiles re-verified after an initial sample-
   list mismatch bug): flat chamfer (9.30 / 9.44 / 9.29 px median);
   visually, scale2.0 broke generation into noise-like crosshatch chaos
   rather than improving structure-following.

This looked like a structural ControlNet limitation. **User then reported
confirming ControlNet works fine in Stable Diffusion Forge, using a
public `lineart`-type ControlNet model on the same kind of rough input**
-- directly overturning that read. Re-diagnosed with three isolated
tests, using the public `lllyasviel/control_v11p_sd15s2_lineart_anime`
checkpoint (downloaded via diffusers, no local training) on the same
10-tile validation subset:

1. **Public ControlNet + our SD1.5 base + our line-domain LoRA
   (scale1.4), raw rough input**: chamfer only marginally better (8.81 vs
   our own ControlNet's 9.30 median) but visually a *different* failure
   mode -- output collapsed into dense crosshatch/shading texture
   covering the whole frame (`results/public_controlnet_lineart_
   anime_20260808/`), resembling the earlier (since-fixed) rough-domain
   "parallel-hatch collapse."
2. **Add the standard `lineart_anime` preprocessor** (installed
   `controlnet_aux`, `LineartAnimeDetector.from_pretrained("lllyasviel/
   Annotators")` -- converts the raw noisy pencil scan into a clean
   white-line-on-black edge map, the format public ControlNets actually
   expect instead of a raw grayscale scan): chamfer barely moved (8.87
   median) and the same dense-hatch texture persisted with our LoRA still
   attached (`results/public_controlnet_lineart_anime_preprocessed_
   20260808/`).
3. **Drop our line-domain LoRA entirely** (plain SD1.5 base + public
   ControlNet + preprocessed conditioning, generic caption): the fix.
   Visually the best result this whole branch has produced -- confident
   white-background linework with real structural correspondence to the
   input on multiple tiles (one tile: a dagger/sword shape in the rough
   reproduced recognizably in the output, the first clear input-specific
   correspondence seen in any ControlNet variant tried). Chamfer 8.75
   median, kept at
   `results/public_controlnet_noLora_full_20260808/` (10 tiles) for
   reference.

**Conclusion**: three separate factors were conflated. Public-ControlNet
pretraining scale (hundreds of thousands-plus images vs. our 1,489-2,510
tiles) is real and matters -- "10x more training didn't fix it"
(2026-08-04/05) meant 10x our own tiny baseline, not an absolute
sufficient amount. Preprocessing format plausibly matters (raw noisy
scan vs. clean edge map) but the chamfer numbers didn't show a large
isolated effect here. The one clearly load-bearing factor in this test:
**our own unconditionally-trained domain LoRA, stacked at scale1.4 on
top of the public ControlNet, actively overrides its conditioning
signal** -- the LoRA's own strong learned pull (toward its training
domain's texture) wins out over the structural information the
ControlNet is trying to inject, consistent with the same "base/adapter
prior wins over conditioning" dynamic suspected from the start, just
located in a different component (our LoRA) than originally assumed (the
base checkpoint or the ControlNet itself).

### Decision

Do not train a new ControlNet from scratch. Once the in-progress paired
rough/line dataset is ready (see `doc/work_log.md`'s dataset-extraction
notes and `[[project_panel_layer_extraction]]` memory), **fine-tune the
public `control_v11p_sd15s2_lineart_anime` checkpoint** on our own pairs
instead -- transfer learning from its massive existing pretraining
rather than starting from an empty `ControlNetModel.from_unet(unet)`
copy. Our house style should be acquired through that fine-tune directly
(the paired loss naturally pulls output toward our specific line-art
style), not bolted on afterward as a separately-trained, unconditionally-
learned LoRA -- today's result shows that composition actively fights
the conditioning signal rather than complementing it.

### Next Actions

1. Once paired data arrives, fine-tune `control_v11p_sd15s2_lineart_
   anime` (not `AOM3A1B`-based `ControlNetModel.from_unet`) on the new
   pairs, with per-tile WD14 captions from the start (the domain-LoRA
   chain's lesson: a single fixed caption starves the model of content-
   disambiguation signal) and the `lineart_anime` preprocessor applied to
   rough tiles before conditioning.
2. Do not stack the current unconditionally-trained line-domain LoRA on
   top of this pipeline -- confirmed actively harmful in this
   configuration. Revisit whether *any* form of style adapter is safe to
   add only after the paired fine-tune itself is working.
3. `dataset/raw_zips/dataset_psd_line_v2.zip` (275 line-only tiles,
   2026-08-08 arrival) is not yet paired data and cannot feed this fine-
   tune directly -- usable for the line-domain LoRA's training pool if
   that direction is revisited, but the paired-conversion work above is
   blocked on the user's separately-planned paired extraction batch.

## 2026-08-08 (later): Pseudo-Pair Bootstrap Attempt -- LoRA Confirmed Safer Than Full Fine-Tune, Result Still Inconclusive; Roughify Tuned Against Real Metrics

While waiting for real paired data (user: "next week" earliest, possibly
later), attempted a pseudo-pair bootstrap of the public ControlNet.

### Pseudo-Pair Generation

Built `tools/pair_extraction/roughify_line.py`: deterministic degradation
of real clean line art (multi-pass jittered duplication + elastic
distortion + graphite tone/noise) into synthetic "rough" images, so
correspondence to the source is guaranteed by construction -- unlike
SDEdit-based generation, which the same day's testing showed has
unreliable structure preservation. Applied to all 1,489
`line_combined_koma_20260729` tiles -> `dataset/pairs_480/train/rough_
pseudo_roughified_20260808/`.

### Full Fine-Tune Bootstrap: Destabilized The Pretrained ControlNet

Added `--controlnet-init` to `scripts/train_controlnet.py` (fine-tune
from a pretrained `ControlNetModel.from_pretrained(...)` instead of
`ControlNetModel.from_unet(unet)` from-scratch init). Fine-tuned the
public `control_v11p_sd15s2_lineart_anime` on the pseudo-pairs + per-tile
WD14 captions (reused `results/domain_lora_line_captiontags_20260807/
tags.csv`), LR=2e-6. Two runs (1860 steps, then a 300-step retry to test
an overtraining hypothesis): both destabilized the pretrained ControlNet's
clean output into new failure modes (dense hatch texture with raw input/
generic caption; soft painterly gray rendering with preprocessed input/
matched per-tile caption) despite retaining partial structural
correspondence (a recurring "sword tile" stayed recognizable across every
variant). 300 steps degraded about as much as 1860 -- not simply an
overtraining/step-count issue.

**Root cause, found via web research**: lllyasviel's own ControlNet
training docs state small-dataset training is safe because zero-
initialized output convolutions start as a no-op and grow gently --- but
that guarantee is explicit about training from a *fresh* `ControlNetModel.
from_unet()`-style cold start. It does not cover continuing to fine-tune
an *already-trained* checkpoint: those "zero convolutions" are no longer
zero once trained, so a warm-started full fine-tune has no equivalent
safety net, and a small/noisy dataset's gradient updates can push already-
fully-active weights arbitrarily. This resolved the apparent contradiction
between the official "small data is safe" claim and what was observed.

### LoRA Variant: Confirmed Safer, Not Yet A Clear Win

Added `--controlnet-lora-rank` to `train_controlnet.py`: freezes
`--controlnet-init` entirely and trains only a peft LoRA adapter on its
attention layers (`ControlNetModel.add_adapter`/`save_lora_adapter`/
`load_lora_adapter` -- note `load_lora_adapter` needs `prefix=None`
explicitly, its default `prefix='transformer'` silently matches nothing
against a ControlNetModel and loads no weights). Same pseudo-pairs, rank
16, LR raised to 1e-4 (standard LoRA practice, vs. 2e-6 for full
fine-tune). Result: visibly less destabilized than either full-fine-tune
variant (the sword tile rendered with cleaner linework, less hatch/
rendering-texture drift), but chamfer-to-GT was statistically
indistinguishable across all three (LoRA 8.71 / full-FT 8.74 / untouched
public ControlNet+preprocessor 8.75 px median on the same 10-tile set) --
confirms the LoRA-is-safer hypothesis but does not show a clear quality
win over just using the untouched public checkpoint.
Montage: `results/controlnet_bootstrap_lora_pseudo_pairs_20260808_eval/_montage.png`.

**Conclusion**: today's pseudo-pair bootstrap did not produce a checkpoint
worth adopting over the untouched public ControlNet. Its value was
methodological: confirmed LoRA (not full fine-tune) is the correct
mechanism for adapting an already-trained ControlNet on small/synthetic
data, and pinned down why (zero-conv safety net doesn't cover warm-start).
Apply this once real paired data arrives.

### Roughify Tuning Against `measure_lineart_profile.py`

Separately (not yet re-tested against a new bootstrap run), tuned
`roughify_line.py` by comparing its output's profile directly against
real `rough_ref` via `tools/evaluation/measure_lineart_profile.py`. v1's
biggest gaps: `long_line_ratio` (62% dev from real -- jittered whole-image
affine copies stayed too parallel/continuous), `faint_mean_dist_to_ink`
(66% dev -- faint pixels hugged the ink too closely, unlike real stray
construction lines). Revised to per-pass independent local elastic
distortion (instead of one shared whole-image warp after accumulation),
added a stroke-fragmentation step (random small gaps punched into the ink
mask) and a separate low-opacity/high-jitter "stray mark" pass. First
revision (v2) overshot fragmentation in the other direction; reduced
`fragment_gaps_per_1k_px` 6.0->2.5 and increased the stray pass's jitter/
opacity (v3): `long_component_ratio` and `components_per_1k_ink_px` both
now within ~1-2.5% of real (were 17-29% off), `long_line_ratio` improved
to 44% dev (from 62%), `line_width_p50`/`faint_mean_dist_to_ink` still
~40-54% off -- diminishing returns reached for this session, not fully
converged. Regenerated the full 1,489-tile pseudo-rough pool with v3
before further tuning continued elsewhere.

### Next Actions

1. Not yet done: re-run the LoRA ControlNet bootstrap against the v3
   (improved) pseudo-rough pool, to test whether the earlier "no clear
   win" result was specifically because v1's pseudo-rough statistics were
   still too far from real rough.
2. Continue roughify tuning if revisited: `line_width_p50` and
   `faint_mean_dist_to_ink` remain the largest gaps.
3. Once real paired data arrives, use the LoRA fine-tuning mechanism
   (confirmed safer) on the public ControlNet checkpoint, per the prior
   entry's decision.

## 2026-08-09: v3-Data LoRA Bootstrap Retest -- Worse, Not Better; Pseudo-Pair Bootstrap Shelved

Re-ran the LoRA ControlNet bootstrap (same recipe as the v1-data run:
rank16, LR1e-4, 1860 steps) on the v3-tuned pseudo-rough pool (`dataset/
pairs_480/train/rough_pseudo_roughified_20260808/`, regenerated with
`roughify_line.py` v3 -- per-pass local elastic distortion, stroke
fragmentation, separate stray-mark pass). Checkpoint: `checkpoints/
controlnet_bootstrap_lora_pseudo_pairs_v3data_20260808/final`.

4-way montage (rough / public ControlNet no-LoRA / v1-data LoRA / v3-data
LoRA / GT) on the standard 10-tile diagnostic set: `results/
controlnet_bootstrap_lora_pseudo_pairs_v3data_20260808_eval/
_compare_montage.png`. Chamfer-to-GT (`edge_map`+`chamfer`, truncate=20px,
same 10 tiles): public_noLora median 8.75 (mean 8.55), lora_v1data median
8.71 (mean 9.13), **lora_v3data median 9.51 (mean 10.12) -- worse than
both**, confirmed visually: v3data introduces a new, recurring
plaid/grid/crosshatch texture-collapse artifact overlaid across most
tiles regardless of input content (rows 1/2/3/4/6 of the montage all show
a similar diagonal-grid or window-pane pattern), a failure mode not
present in v1data or the untouched public checkpoint. Best guess: the v3
roughify revision's structural regularities (per-pass local elastic
warp's fixed sigma, or the stray-mark pass's fixed jitter geometry)
introduced a subtle but consistent statistical pattern across the 1,489
synthetic-rough tiles that the LoRA picked up as a spurious shortcut
instead of genuine structure-following. Not investigated further, since
this is pseudo-data by construction and the effort is better spent once
real paired data arrives.

**Decision**: shelve the pseudo-pair bootstrap approach entirely. Neither
v1 nor v3 pseudo-rough data produced a ControlNet worth adopting over the
untouched public checkpoint; v3 was actively worse. Do not pursue further
roughify tuning iterations for this purpose. The LoRA-vs-full-fine-tune
mechanism finding (2026-08-08 entry) remains valid and will be applied
once real paired data arrives (~2026-08-31 per current estimate); until
then, no further ControlNet training is planned.

## 2026-08-09: Panel-Detection Prototype (`dataset_psd_line_v2.zip`) Reviewed -- Real Gaps On Both Ends

Background Agent A's panel-detection prototype (`tools/pair_extraction/
extract_psd_line_koma_regions.py`, salvaged from its worktree after its
own background-resume failed) reviewed on its 30-page/129-tile sample
output (`results/psd_line_koma_extraction_20260808/`).

**Panel detection**: `pages.csv` panel_count distribution across 30
sampled pages: `{1: 25, 2: 2, 3: 2, 7: 1}`, only 5/30 flagged
`is_multi_panel=True`. Visual check against `panel_detection_overlay_qc.
png` found at least one confirmed false negative: `0032_05_line`, which
direct earlier inspection of the source PSD showed has ~3 visible panels
separated by thick baked-in black borders, was tagged `n=1`. Several
other pages that look multi-panel by eye were also tagged `n=1` --
real recall gap in the detector, not yet root-caused.

**Tiling**: `tile_qc.png` (129-tile grid with per-tile ink-ratio
annotations) shows a large fraction of near-blank/low-content tiles --
crops containing only a stray line or two, or isolated sound-effect/
dialogue text, no real linework. The existing koma pipeline avoids this
via a deliberate content-density-based sub-region split step (`doc/
raw_dataset_extraction_knowledge.md`, housei section); this prototype
does not yet have an equivalent step.

**Status**: usable as a first draft, not production-ready. Needs (1) a
second pass on the panel-detection recall gap and (2) a content-density
gate before tiling, mirroring the existing koma pipeline's sub-region
split. Not yet decided whether to invest further here vs. moving on to
workstream C (eval-metric improvement, next in the user's stated
priority order after B).

## 2026-08-09 (later): Cleanup Before SSD Swap

User is doing a physical SSD replacement in preparation for the large
paired-dataset arrival; other work paused, used the downtime to clean up
`results/`/`checkpoints/`/`dataset/` clutter. Root FS was at 25G free
(229G total, 89% used).

Deleted (all confirmed superseded/shelved earlier the same day, findings
already recorded in this log and in `results/` eval montages which were
kept):
- `checkpoints/controlnet_bootstrap_pseudo_pairs_20260808/` (5.4G, full
  fine-tune, rejected -- destabilized the pretrained ControlNet)
- `checkpoints/controlnet_bootstrap_pseudo_pairs_short_20260808/` (5.4G,
  300-step retry, also rejected)
- `checkpoints/controlnet_bootstrap_lora_pseudo_pairs_20260808/` (1.4G,
  LoRA on v1 pseudo-data, superseded by the shelve decision)
- `checkpoints/controlnet_bootstrap_lora_pseudo_pairs_v3data_20260808/`
  (1.4G, LoRA on v3 pseudo-data, confirmed worse than v1 -- see prior
  entry)
- `dataset/pairs_480/train/rough_pseudo_roughified_20260808/` (134M,
  synthetic pseudo-rough pool feeding the now-shelved bootstrap approach)

Not touched (explicitly kept per policy/still-referenced): domain LoRA
checkpoints (`domain_lora_{line,rough}_20260804` baselines +
`domain_lora_{line,rough}_sd15base_sksv2_20260807` adopted configs),
`results/unpaired_skima_*` (~1GB, tied to
`[[project_unpaired_data_pools]]`'s still-open continuity-regularization
follow-up, not confirmed dead), all `results/*_eval*` diagnostic montages
from today (small, kept as the recorded evidence for the shelve
decisions), the salvaged `dataset_psd_line_v2.zip` panel-detection
prototype output (2.3M+1.2M, still-open workstream A).

Freed ~14GB (root FS 25G -> 39G free). Also removed Agent A's leftover
worktree (`.claude/worktrees/agent-aca7212d7faf25055/`) and its throwaway
branch after confirming its one untracked file was byte-identical to the
already-salvaged copy in the main tree.
