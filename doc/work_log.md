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

## Next Actions

1. **Blocked on external work**: panel-border-layer extraction (another
   machine, user-side). Resume the panel-boundary-first segmentation plan
   once available.
2. Model-side experiment design using the broadened pool (ako5ver2 native
   strict 422, fitness 271, housei 65, fighting 40) remains open, but
   attribute the soft/density-map ceiling partly to unresolved alignment/scale
   mismatch now, not recipe-only; see `doc/CURRENT.md`.
3. Decide whether to rename the remaining 5 `kurip`-named infra scripts.
4. Do not mix fitness/housei/fighting/ako5ver2-native sources into one list
   without a deliberate experiment design.
5. If more housei volume is wanted later, `soft_ink_ratio` still has
   documented headroom (0.60/0.70 yield more tiles); use
   `diagnose_gate_funnel.py` on any new source before assuming which gate to
   loosen.
