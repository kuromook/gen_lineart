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

## 2026-08-09 (later): Panel-Detection Recall Fixed -- Recursive X-Y Cut Replaces Enclosed-Island Detection

Root-caused the panel-detection recall gap flagged in the prior entry.
`0195_15_line`: a panel's top border is undrawn (bleeds to the page trim)
and its side dividers start ~10px short of the physical top edge at detect
resolution -- enough of a gap for the free-space flood-fill to leak around
and merge the panel into the page's outer background component.
`0032_05_line` (previously-known false negative, ~3 visible panels): the
internal cross-shaped divider between panels is detected fine, but no
outer frame line encloses the whole grid, so all four quadrants leak into
one shared margin component. Confirmed both cases by rendering the
detect-resolution border-candidate mask directly and inspecting labeled
free-space components pixel-by-pixel -- the underlying enclosed-free-
space-island algorithm requires a fully closed loop on all four sides,
which this source frequently does not draw.

Bumping `--border-close-px` (4->8) alone fixed the small-gap case but not
the missing-outer-frame case (`0032_05` stayed 0 panels even at
close_px=25). Replaced the detector's core with **recursive X-Y cut**
(`recursive_xy_split` in `extract_psd_line_koma_regions.py`): starting
from the whole page, find the strongest row/column whose border-candidate
coverage exceeds `--min-divider-span-frac` (default 0.7) of the current
region's width/height, split there, and recurse into each half
independently. This only requires a single divider that spans most of the
*current region*, not a closed loop -- so it handles edge-bleed panels and
missing outer frames for free (each half's own edges become the new
implicit boundary for the next split), and handles staircase/irregular
grids correctly since each half is split independently rather than
assuming one global grid.

Threshold sweep on ~20 known-problem pages: 0.7 was the sweet spot
(0.8 lost the `0032_05` case again, 0.9 collapsed nearly everything back
to n=1). Visual check of the resulting overlay QC
(`results/psd_line_koma_extraction_20260809_recut_sample/
panel_detection_overlay_qc.png`, 30-page random sample) found no false-
positive over-splits -- single-panel illustrations with busy backgrounds
(`0357_01`, `0435_12`, sidewalk/scenery linework) stayed correctly
undivided.

Also fixed a latent crash: 15/290 pages in `manifest.json` reference an
`outputs.line` filename that does not actually exist in
`dataset_psd_line_v2.zip` (e.g. `0002_完成_line.png`) -- now skipped with
a warning instead of raising `KeyError` and aborting the whole run.

**Full 290-page before/after** (both dry runs, `--overlay-qc-count 0
--tile-qc-count 0`; 275 pages actually resolve after the 15 missing-file
skips):
| | multi-panel pages | panels (incl. fallbacks) | tiles |
|---|---|---|---|
| old (enclosed-island, close_px=4) | 86/275 (31%) | 463 | 1113 |
| new (recursive X-Y cut, close_px=8, span_frac=0.7) | 198/275 (72%) | 933 | 2022 |

Results: `results/psd_line_koma_extraction_20260809_recheck/` (old
baseline) vs `results/psd_line_koma_extraction_20260809_recut_full/`
(new). Content-density tiling gap flagged in the prior entry turned out to
be mostly a downstream symptom of the recall gap -- with panels now
correctly and tightly bounded, the existing (already-reused-from-koma-
pipeline) subregion ink-density split + tile ink-ratio gate produces
mostly content-dense tiles without any separate content-density mechanism
needed; spot-checked via `results/
psd_line_koma_extraction_20260809_recut_sample/tile_qc.png`.

**Status**: panel detection is now in much better shape (72% multi-panel
recall on a real-world sample, no observed false-positive over-splits).
Not yet run with `--save` to materialize tiles to disk, and no rough
counterpart exists for this source yet (line-only, per prior entries) so
this still cannot feed ControlNet training directly -- only the
domain-LoRA-relevant line-only pool benefits for now.

## 2026-08-09 (later still): Workstream C -- Eval-Metric Investigation Found A Bad Reference Pair, Not (Only) A Bad Metric

Picked up workstream C (chamfer-to-GT has repeatedly failed to
discriminate visually-different ControlNet outputs, per multiple 2026-08-08
entries). Used the clearest documented case as a calibration pair:
`results/public_controlnet_lineart_anime_preprocessed_20260808`
("bad_hallucination", our line-domain LoRA stacked on public ControlNet,
produces a content-independent plaid/crosshatch texture) vs `results/
public_controlnet_noLora_full_20260808` ("good_structural", plain base +
public ControlNet, praised in the 2026-08-08 entry for "the first clear
input-specific correspondence seen in any ControlNet variant tried").

Tried three independent metric designs on the standard 10-tile diagnostic
set, all against this pair:
1. Chamfer-to-GT with truncate reduced 20px->6px (cheap parameter fix).
2. A new conditioning-roundtrip fidelity metric
   (`tools/evaluation/condition_roundtrip_fidelity.py`): re-run the
   `lineart_anime` ControlNet preprocessor on the model's own output and
   compare the recovered conditioning map to the original conditioning
   map derived from the rough tile, sidestepping GT entirely.
3. Gaussian-blurred density-field normalized cross-correlation (robust to
   exact-point misalignment by construction).

All three -- despite being methodologically unrelated -- agreed with each
other and **disagreed with the "good_structural wins" label**, scoring
bad_hallucination equal or better on average across the 10 tiles.
Suspecting a harness bug, did a manual tile-by-tile visual re-review of
all 10 rows in the comparison montage instead of trusting the prior
session's holistic impression. Found the "good_structural" label does not
hold up per-tile: e.g. the dagger tile the 2026-08-08 entry specifically
cited as the success case actually shows bad_hallucination reproducing the
dagger silhouette *more* cleanly (good_structural's version is partly
obscured by its own crosshatch overlay); the housei_011 tile shows
good_structural replacing a simple rough curve with a fully unrelated
horned-creature illustration, while bad_hallucination at least preserves
the curve's rough direction under its texture overlay.

**Conclusion**: the 2026-08-08 "good_structural is the best result this
branch has produced" judgment was a holistic/best-case impression (likely
anchored on one standout tile plus general rendering polish), not a
claim that held at the per-tile average this 10-tile set was being used
to test. The three new metrics may not have been wrong -- they were being
validated against a reference label that does not actually hold up.
**Do not use this pair as a calibration reference for any future metric
work** without a proper from-scratch per-tile human fidelity ranking
first; the existing holistic write-ups in this log are not a substitute.

**Status**: paused here pending user direction -- next step, if
continued, is building a small human-labeled per-tile fidelity ranking
(not a holistic "which is better overall" judgment) before trying to
calibrate any metric against it again.

## 2026-08-09 (later still): Workstream C Resolved -- Conditioning-Roundtrip Metric Validated Against A Hand-Built Fidelity Ranking

Followed up on the prior entry (calibration pair turned out unreliable). Built
the missing piece: a manual tile-by-tile fidelity ranking (`results/
eval_metric_calibration_20260809/tile_fidelity_ranking.csv`) over the same
10-tile diagnostic set, judged strictly on "does the output's line structure
correspond to the rough's structure," independent of rendering polish.
Verdict: bad_hallucination (LoRA-stacked) more faithful on 6/10 tiles, tied on
4/10, good_structural (adopted 2026-08-08 variant) more faithful on 0/10.

Checked all three candidate metrics from the prior entry against this ranking
on the 6 decisive tiles: chamfer-to-GT (truncate=6px) agreed on only 1/6 --
confirming it answers a different question ("does this resemble finished
GT-style line art in general") than conditioning adherence. The
conditioning-roundtrip metric agreed on 5/6; blurred density-field
correlation agreed on 4/6. Full table: `results/
eval_metric_calibration_20260809/metric_agreement_summary.md`.

**Decision**: adopt `tools/evaluation/condition_roundtrip_fidelity.py` as the
primary diagnostic whenever the question is conditioning adherence /
hallucination detection (this is exactly the question that started workstream
C). Keep chamfer-to-GT for what it actually measures well -- general
resemblance to finished line art -- stop treating it as a conditioning-
fidelity proxy.

**Open implication, not acted on**: under this criterion, the LoRA-stacked
variant was structurally more faithful to input than the variant the
2026-08-08 entry adopted, on this 10-tile set. Not a simple "the LoRA variant
is better" call -- it still has the content-independent crosshatch artifact
that motivated dropping it -- so this is a genuine fidelity-vs-cleanliness
trade-off, not one variant dominating. Flagged for the user; the adoption
decision itself was not revisited this session.

## 2026-08-09 (later still): Eval Metric Inventory

User: with real paired data still ~3 weeks out, now is a good time for a
full metric stocktake, not just the ControlNet-specific fix from the prior
two entries. Read through every script in `tools/evaluation/` plus the
`chamfer()` usages in `tools/pair_extraction/` and classified each by what
comparison it actually performs, not just its filename. Written up as
`doc/eval_metric_inventory.md`.

Key groupings that came out of this:
- **Point-distance-to-GT family** (`evaluate_fixed_outputs.py`): the one
  implicated by today's finding. Also found a second, independent problem
  while inventorying it: its ink extraction uses a raw grayscale
  threshold<128, which on diffusion outputs (soft/antialiased, unlike clean
  scans) inflated ink_ratio to 60-165x GT in one measured case -- a real bug
  on top of the truncate-radius issue, not yet fixed.
- **Point-distance-to-conditioning family** (`condition_roundtrip_fidelity.py`):
  today's validated fix, the right tool for "did the model follow input"
  questions.
- **No-reference structural-profile family** (`measure_lineart_profile.py`,
  `evaluate_stroke_stability.py`): mechanically different -- no nearest-
  point search at all, just per-image statistics compared as distributions.
  Not exposed to the density-saturation failure mode by construction. This
  is already the correct, unaffected basis for the domain-LoRA fidelity-
  budget work ([[project_diffusion_fidelity_budget]]).
- **Distance-banded intensity family** (`evaluate_halo_outputs.py`): also
  mechanically different (ring-averaged density, not nearest-point), built
  for a GAN-era artifact not currently active on the diffusion branch. Not
  re-verified against a hand ranking this session.
- **Real-vs-real alignment-gating family** (chamfer inside
  `tools/pair_extraction/*.py`, `score_pair_agreement.py`): different
  question (are two real scans aligned, no hallucination risk since neither
  side is generated) and already has an extensive independent calibration
  history in this log. Not re-examined this session; flagged as an
  assumption (lower hallucination-confound risk) rather than a verified one.

No code changes made beyond the inventory doc itself; the ink-extraction bug
in `evaluate_fixed_outputs.py` is logged as an open follow-up, not fixed.

## 2026-08-09 (later still): Fixed `evaluate_fixed_outputs.py`'s Ink-Extraction Bug

First item off the eval-metric-inventory follow-up list. Switched ink
extraction from a raw grayscale threshold (`gray < 128`) to the same
Canny-edge extraction (`edge_map`) already validated for
`condition_roundtrip_fidelity.py`, and dropped the default chamfer truncate
radius 20px->8px (both CLI-overridable: `--extraction threshold|edge`,
`--truncate-px`). `--extraction threshold` reproduces the exact old numbers
for anyone needing historical comparability -- verified byte-for-byte
against the pre-fix run (ink_ratio 60.66/165.19, chamfer 11.30/9.32,
matching the inventory doc's measured values exactly).

With the fix (edge extraction, truncate=8), ink_ratio on the same
bad_hallucination/good_structural pair dropped to a sane 9.2/21.8 (was
60.7/165.2) and chamfer to 4.77/4.23 (was 11.30/9.32). Self-comparison smoke
test (target vs. itself) gives chamfer=0, F1=1.0 as expected.

**Important negative result, worth being explicit about**: re-checked the
fixed metric against the hand-built fidelity ranking
(`results/eval_metric_calibration_20260809/tile_fidelity_ranking.csv|
metric_agreement_summary.md`) -- still only 1/6 agreement on the decisive
tiles, same as before the fix. This confirms the ink-extraction bug and the
conditioning-fidelity blind spot are two *separate* problems: fixing ink
extraction makes `evaluate_fixed_outputs.py` correct for the question it's
actually built to answer ("how close does this land to GT"), but does not
and cannot turn it into a conditioning-fidelity metric -- that job stays
with `condition_roundtrip_fidelity.py`. Updated `doc/eval_metric_inventory.md`
to mark this follow-up done and record the negative result so nobody
mistakes "ink extraction fixed" for "discrimination problem fixed."

## 2026-08-09 (later still): Purged Unreferenced `results/` Entries (Full History)

Followed up on the session-close cleanup habit noted earlier today. Cross-
referenced every top-level entry in `results/` (259 total) against every
`doc/**/*.md` file plus `README.md` by exact filename string match. 127
entries (1.78GB) had no exact-match reference anywhere in the docs.

Manually spot-checked a handful of the "unreferenced" candidates first
(`router_feature_probe_current_oracle_choices_with_features.csv`, the
per-source `*_koma_tiles_480_dense_20260801*` QC bundles) and found some of
them ARE topically covered by doc prose just not by exact filename -- so
this match is a blunt instrument, not a precise "genuinely unused" signal.
User's call after hearing this caveat: audit metrics are about to become an
active, ongoing workstream (today's eval-metric inventory), old artifacts
from months-old, already-superseded experiment lines are unlikely to be
worth manually re-reviewing one by one, and disk space is no longer tight
(new SSD, 1.6TB free) so this wasn't about reclaiming space -- decided to
delete the full unreferenced set rather than hand-review each one.

Deleted all 127 entries. `results/` went from 259 items / 2.6GB to 132
items / 726MB. Full pre-deletion candidate list is not preserved separately
-- it was exactly "every top-level `results/` entry as of 2026-08-09 whose
filename does not appear verbatim in any `doc/**/*.md` or `README.md`",
reproducible by the same grep-based cross-reference if ever needed again.
Everything currently referenced by exact path in the docs (132 remaining
entries, including all of today's own new outputs -- see the surrounding
2026-08-09 entries in this log) was left untouched.

## 2026-08-09 (later still): Eval-Metric Literature Survey -- Our Finding Is Known, Not Novel

User asked whether an established literature exists for illustration/
line-art comparison metrics that might reveal a flaw in our approach or
framing. Ran a background research agent against this exact question.
Full writeup: `doc/eval_metric_literature_survey_20260809.md`.

Headline: the chamfer/tolerance-F1 density-saturation problem found earlier
today is a known, decades-old phenomenon in boundary detection -- Pratt's
Figure of Merit (1978, Chamfer-like) was replaced by BSDS's one-to-one
bipartite-matched precision-recall protocol for exactly this reason, and
it's still active research territory in 2026 (MatchED, CVPR 2026, for
learned edge detectors specifically). Also found that today's
`condition_roundtrip_fidelity.py` framework independently converged on the
same design as ControlNet's own segmentation-IoU eval and ControlNet++'s
formalized "controllability" consistency reward -- but ControlNet++'s
established comparison function for edge/line-art conditions specifically
is SSIM, not Chamfer, meaning our roundtrip metric likely fixed *what* gets
compared but not the underlying comparison mechanism's weakness.

Also surfaced a documentation gap: `evaluate_fixed_outputs.py`'s
`f1_2px`/`precision_2px`/`recall_2px` are named like BSDS-family
edge-detection F-scores but use a materially weaker many-to-one distance-
threshold match rather than BSDS's bipartite one-to-one match -- worth a
caveat so nobody imports BSDS-equivalent intuition from these numbers.

New prioritized prototype list (detail in the survey doc): BSDS-style
bipartite matching via `scipy.optimize.linear_sum_assignment`; swap/add
SSIM inside the roundtrip metric; a GF-HOG-inspired gradient-orientation
descriptor targeting repeating-texture hallucination specifically;
re-validate the roundtrip metric at higher ink density (today's 5/6 result
may reflect comparing similar-density content, not a fixed mechanism).
Confirmed no action needed on `measure_lineart_profile.py` -- its
no-reference hand-designed-statistics approach is validated as the more
defensible choice for domain/style-fidelity questions, given documented
FID/LPIPS domain-mismatch problems on non-photographic content.

Not yet implemented -- this is a literature-grounded backlog for the next
metric-audit pass, not code changes.

## 2026-08-09 (later still): Prototyped and Validated the 3 Literature-Suggested Metrics -- BSDS Matching Reaches 6/6

Implemented and validated all three candidates from the literature survey
against the same 10-tile hand fidelity ranking used throughout today.

1. **BSDS-style one-to-one bipartite-matched F1**
   (`tools/pair_extraction/tile_region_manifest_480.bipartite_match_f1`):
   candidate pairs within tolerance found via KD-tree, then maximum-
   cardinality bipartite matching (`scipy.sparse.csgraph.
   maximum_bipartite_matching`) -- a documented simplification of BSDS's
   exact min-cost LP formulation, not a faithful reimplementation. Smoke
   test: self-match gives F1=1.0; dense unrelated random noise (5x GT
   density) drops recall from the old many-to-one metric's inflated 0.90 to
   a more honest 0.65 under one-to-one matching, precision stays
   correctly low (~0.13) either way. Added as `bsds_f1`/`bsds_precision`/
   `bsds_recall` in `evaluate_fixed_outputs.py`, alongside (not replacing)
   the existing `f1_2px` family. **Result: 6/6 agreement with the hand
   ranking on the 6 decisive tiles -- the best of any metric tried today,
   and the cheapest (no roundtrip/preprocessor model needed, just the
   corrected matching rule applied directly to output-vs-GT).**

2. **SSIM inside the roundtrip metric**
   (`tools/evaluation/condition_roundtrip_fidelity.py`, new
   `roundtrip_ssim` column, `skimage.metrics.structural_similarity` on the
   raw grayscale conditioning maps before binarization -- SSIM wants
   continuous tone, not a binary mask). Matches ControlNet++'s established
   per-condition-type controllability metric for edge/line-art conditions.
   **6/6.** Also added `roundtrip_bsds_f1` (same bipartite matcher, on the
   binarized conditioning maps) alongside -- **also 6/6.** Original
   `roundtrip_chamfer` kept for continuity, still 5/6.

3. **GF-HOG-inspired orientation-histogram similarity**
   (`tile_region_manifest_480.orientation_similarity`): per-cell (8x8 grid)
   gradient-orientation histograms, compared via histogram intersection,
   averaged over cells with gradient energy in both images. Computed
   directly between rough input and model output (no GT, no preprocessor
   needed). **5/6.** Not yet wired into a CLI script -- available as a
   function for now, meant as a mechanistically-different secondary check
   (orientation-distribution regularity) for repeating-texture
   hallucination specifically, not a primary metric.

Updated the visual-check montage (`tools/compare/
make_multi_model_eval_compare.py --annotate-metrics`) to also print
`bsds_f1` per tile (font tripled per user request earlier today, band
height increased 150->196px for the extra line). Regenerated: `results/
eval_metric_calibration_20260809/metric_fix_visual_check_montage.png`.

Full updated agreement table across all 8 metric variants tried today:
`results/eval_metric_calibration_20260809/metric_agreement_summary.md`.
Updated recommendation: use `bsds_f1` as the default "did the output
structurally correspond" check when GT exists (cheapest, best result);
use `roundtrip_ssim`/`roundtrip_bsds_f1` when it doesn't (e.g. judging a
ControlNet against its own conditioning before real paired data arrives).

## 2026-08-09 (later still): `dataset_psd_line_v2.zip` Materialized -- 2022 Tiles Saved

`--save` run completed (`results/psd_line_koma_extraction_20260809_save/`,
`dataset/psd_line_koma_extraction_20260809/line/`). 2022 tiles, matching
the prior dry-run exactly (same params). Visual QC (`panel_detection_
overlay_qc.png`, `tile_qc.png`): panel splits track real gutters/borders
cleanly across a wide sample, no obvious false-split artifacts.

**Minor caveat found during QC, not blocking**: a handful of source pages
(seen: 0001, 0003, 0020, 0021) have a faint gray calibration/reference
strip near one edge (tick-mark scale + "NAN060"-style label text, likely a
scanning color/greyscale reference chart) baked into the flattened raster.
It's faint enough to barely clear the tile ink-ratio floor
(`--tile-ink-min 0.010`), so it mainly affects tiles that would otherwise
be near-blank -- these are already low-value tiles, so impact is small, but
worth knowing about if this source ever needs a stricter content-density
filter or the strip needs explicit masking. Not acted on this session.

This source is still line-only (no rough counterpart), so per prior
entries it feeds domain-LoRA-relevant line pools only -- not usable for
ControlNet training until a paired rough extraction exists for it.

## 2026-08-09 (later still): Retroactive Audit -- GAN-Era Adoption Ranking Unaffected

User asked whether today's metric findings should change any past
training/inference decision. Reasoned through it first: the failure mode
found today (dense output masking lack of structural correspondence) is
architecturally tied to conditional generation where an adapter/LoRA can
override its conditioning signal (ControlNet+LoRA); the GAN-era direct-
regression models (`cleanup-refiner` branch) map input to output
deterministically, a different and much less hallucination-prone failure
surface. Also, `doc/architecture_decisions.md` and `doc/model_results_
summary.md` already establish (and were created specifically because of
past cases of) "never trust F1/chamfer alone, montage review is the real
adoption gate" -- so GAN-era decisions were never purely metric-driven to
begin with.

Ran one concrete spot check to confirm rather than just reason about it:
re-scored the 8 surviving `combined_koma_*_20260731`/`_20260730` family
output sets (the standard 8-tile clean eval set, `eval_clean_lineart004_8.
txt`) with the new `bsds_f1` alongside the original `f1_2px`. **Ranking
order was identical across all 8 models for both metrics** (lucy_mild_msgan
> dualhead_v2 > cleanupdark > lucy_mild_noadv > attn > hed_v3 > dualhead >
hed). ink_ratio for this family also stayed in a bounded, sane range
(0.07-2.1x), unlike the 60-165x blowup found for diffusion-model output --
consistent with GAN-era outputs already being closer to cleanly binarized,
not soft/antialiased.

**Conclusion**: no GAN-era adoption decision needs revisiting based on
today's metric findings. The one item that *does* need revisiting when
real paired data arrives is already flagged (2026-08-08 ControlNet
LoRA-drop decision, see the "Workstream C Resolved" entry) -- that is
specifically a conditional-diffusion/ControlNet-adapter case, the exact
architecture class where today's failure mode applies.

## 2026-08-19/20: Real Paired Data Arrived -- `dataset_clip_pairs.zip`, Plus Two Unpaired Line-Only Batches

Four new raw archives appeared at the repo root (not yet placed). Identified
and sorted each by inspecting its internal manifest/structure before moving
anything, per this project's standing storage policy
(`doc/preprocess/raw_dataset_storage_policy.md`).

- **`dataset_clip_pairs.zip` (8.0GB, 3590 entries) -- real paired rough/line
  data**, the item this project has been blocked on since the 2026-08-08
  ControlNet-LoRA-drop decision (estimated ~2026-08-31, arrived ~2 weeks
  early). Auto-extracted by the user's own tooling from CLIP STUDIO `.clip`
  files across 74 project folders (2015-2024), each with a `manifest.json`
  giving per-page `line`/`sketch` filenames plus built-in alignment
  diagnostics (bbox/centroid/`normalized_centroid_distance`/
  `block_density_correlation`/`aligned`). Several subproject names overlap
  known sources (`ako5`/`ako6`/`ako7`/`ako3B`/`ako4`/`housei`/`fitness`/
  `hamlabi`) -- **not yet confirmed whether these duplicate the existing
  koma-pipeline extracts or are a distinct scan**; needs a dedup/overlap
  check before use. Moved to `dataset/raw_zips/dataset_clip_pairs.zip`.
- **`dataset_comicstudio_line.zip` (4.9GB, 3214 entries) -- unpaired,
  line-only** (88 project folders, 2008-2014+pending; manifests confirm
  `"aligned_pair": false` throughout). Same extraction-tool family as
  `clip_pairs` but for CLIP STUDIO sources where only a line layer was
  classified. Moved to `dataset/raw_zips/dataset_comicstudio_line.zip`.
- **`dataset_psd_line.zip` (280MB, root) -- a new/larger PSD-source line-only
  batch**, not a re-encoding of the existing `_v2`: 735 documents scanned
  (was 731) yielding 317 line-only outputs (was 275), confirmed by hash and
  content diff. Renamed on arrival to `dataset_psd_line_v3.zip` per the
  storage policy's versioning rule (never overwrite an existing archive
  name) and moved to `dataset/raw_zips/`.
- **`dataset_4th.zip` (165MB, root) -- confirmed duplicate**, not new data
  (matching the user's own suspicion going in). Per-file CRC check against
  `dataset/raw_zips/dataset_4th_koma.zip`: 99/100 members byte-identical
  (only `manifest.json` differed) -- this source was already archived and
  already fully processed through the koma pipeline on 2026-08-01
  (`dataset/pairs_480/valid_train_4th_koma_20260801.txt`). Deleted from the
  repo root after user confirmation; no data lost, the real archive is
  `dataset/raw_zips/dataset_4th_koma.zip`.

Full per-file detail and manifest schema notes:
`doc/preprocess/raw_dataset_storage_policy.md`.

### Next Actions

1. `dataset_clip_pairs.zip` is the priority: inventory per-project page
   counts, check its `ako5`/`ako6`/`ako7`/`ako3B`/`ako4`/`housei`/`fitness`/
   `hamlabi`-named subprojects for overlap against the already-extracted
   koma-pipeline sources before assuming any of it is net-new content.
2. Spot-check the extraction tool's own built-in alignment diagnostics
   (`normalized_centroid_distance`/`block_density_correlation`/`aligned`)
   against a manual sample, to decide whether they can be trusted directly
   or still need this project's own chamfer-based gate
   (`ALIGNMENT_*` in `tile_region_manifest_480.py`) before promoting any
   pairs to training.
3. Once paired data is confirmed usable, this is the trigger for the
   2026-08-08 decision: fine-tune the public `control_v11p_sd15s2_
   lineart_anime` checkpoint via LoRA (confirmed safer than full fine-tune)
   on real pairs, with per-tile WD14 captions from the start.
4. `dataset_comicstudio_line.zip` / `dataset_psd_line_v3.zip` are lower
   priority -- both are line-only, useful only as more line-domain-LoRA
   training material, not for the paired ControlNet fine-tune.

## 2026-08-20 (later): clip_pairs Name-Collision Investigation -- ako5 Is New, housei Is Degraded, fitness/hamlabi Are Fine

Followed up on Next Action #1/#2 above for the four `clip_pairs` folders
whose names matched known sources, per explicit user hypotheses going in
("ako5 は同名フォルダで別の中身かもしれない、housei/fitness/hamlabi は同じ内容
で機械抽出なぶん精度が低い可能性がある"). Ran two independent forked
investigations in parallel (per [[feedback_pair_extraction_parallel_agents]])
rather than assuming either hypothesis was correct — both were checked
against actual page content and byte-level comparison, not just folder
names. Results, condensed (full detail:
`doc/preprocess/raw_dataset_storage_policy.md`):

- **`011_2017_ako5`**: user's suspicion confirmed -- a different manuscript
  (bikini pin-up illustration set, 6543x7016 canvas) from the known
  `ako5ver2` source (4961x7016, standard multi-panel manga), despite the
  name match. 66 pages, new material, catalogue separately.
- **`066_2024_housei\kazenagare`**: user's suspicion confirmed and worse than
  expected -- not just lower machine-extraction confidence, actually
  missing 8 of 18 known pages and picking the wrong/an incomplete layer on
  at least one checked page (near-blank output vs. the known archive's full
  page). **Do not use as a training source without full manual review; the
  existing `housei_koma_subregion_refined` extraction stays the production
  source.** Also found clip_pairs' page numbering does not line up 1:1 with
  the known archive's index for this source -- any future per-page diff
  needs re-indexing first.
- **`069_2024b_fitness\kurip`**: user's suspicion *not* confirmed -- same
  manuscript, quality comparable to the existing extraction (several
  `sketch` files byte-identical to the known archive). No evidence of
  degradation.
- **`070_2024b_hamlabi\works`**: same manuscript, one checked page
  byte-identical (MD5) to the known archive -- effectively a re-export, not
  degraded, but also not new content.
- **`063_2023b_hamlabi\works`** (a second hamlabi-named folder the user had
  not explicitly flagged): turned out to be byte-identical to
  `070_2024b_hamlabi\works` itself -- the same content duplicated under two
  year-tags *inside* `clip_pairs`, not a second distinct hamlabi work.

All four folders confirmed to carry genuine per-page `line`+`sketch`
pairing with alignment diagnostics, unlike `psd_line`/`comicstudio_line`.

Also surfaced during this pass: `clip_pairs` contains ~15 more `ako*`-family
folders (`ako6` through `ako10`, `akogoods`, `akokate`, `akocult`, `ako3`/
`ako3B`/`ako4`, several with multiple year-tagged entries) with no name
match against any existing source -- per the `ako5` finding above, these are
very likely genuinely new, previously-uncatalogued manuscripts from the same
recurring-character naming convention, not overlap candidates. Not
inventoried yet.

### Next Actions

1. `ako5` (`011_2017_ako5`) and `fitness` (`069_2024b_fitness`) are
   confirmed usable/new -- both are candidates for this project's own
   review/tiling pipeline whenever paired-data work resumes.
2. `housei` and `hamlabi` from `clip_pairs` add no new usable content over
   what's already extracted -- do not route either through the pipeline as
   a primary source; `hamlabi` could still serve as a redundant integrity
   cross-check if ever useful, `housei` should not be used without a full
   manual per-page review given the confirmed layer-selection failure.
3. The ~15 unmatched `ako*`-family folders are the largest unexplored
   opportunity in `clip_pairs` -- a full per-project page-count/content
   inventory pass is the natural next step, not yet started.
4. Original Next Action #3 (LoRA fine-tune of the public ControlNet
   checkpoint on real pairs) remains the eventual goal once a first usable
   paired subset (starting with `ako5`/`fitness` above, or a broader
   `clip_pairs` inventory) is assembled and reviewed.

## 2026-08-21: clip_pairs `ako*`-Family Inventory -- 21 Folders Collapse To 12 Distinct Works, `aligned:true` Is A Reliable Quality Filter

Continued the `clip_pairs` investigation onto the ~15-21 previously-unmatched
`ako*`-family folders flagged in the prior entry. Three passes, first two
mechanical/direct (no forking needed), third split into two parallel forks
(per [[feedback_pair_extraction_parallel_agents]]) for the visual
spot-checking:

1. **CRC-based dedup pass** (cheap, exact-byte, done directly): 21 raw
   folders collapse to ~12 distinct works. Several are wholesale or
   near-wholesale duplicates re-exported under different year tags inside
   `clip_pairs` itself -- same pattern as the `hamlabi` 2023/2024 duplicate
   found in the prior entry (`ako7` x2 identical, `ako10` x2 identical,
   `akogoods` x2 identical +1 near-dup, `akocult`'s smaller entry is a full
   subset of its larger one, `ako3`/`ako3B`/`ako4` each have a smaller
   subset entry superseded by a fuller one). Full table with representative
   folder/subproject per work: `doc/preprocess/raw_dataset_storage_policy.md`.
2. **Manifest summary pass** (direct): confirmed every subproject has
   `pairs == n` (true rough+line pairing throughout), but `aligned_true`
   fraction varies wildly per work (35/38 for `ako7` down to 3/12 for
   `ako10`).
3. **Visual spot-check** (2 parallel forks, 6 works each): confirmed all 12
   are genuine sequential multi-panel manga, not single illustrations like
   `011_2017_ako5` turned out to be (`akokate` samples even carry real
   published volume/page annotations). But found the low-`aligned_true`
   works have real, severe extraction failures, not just conservative
   alignment scoring: `ako9`/`ako10` sampled `line` outputs were nearly
   blank against fully-drawn sketches; `akogoods` (despite the
   merchandise-sounding name -- content is normal sequential manga)
   extracted solid-black-fill/shadow fragments with no actual linework;
   `ako4` had sketch/line reversed in production stage (blank sketch,
   finished screentone-heavy line). `akokate`'s own `aligned:false` sample
   showed a genuine partial-extraction failure (only a background panel
   captured, not the page's real content) -- confirming the manifest's
   `alignment.aligned` field is a meaningful, checkable quality signal, not
   noise.

**Standing rule for any future `clip_pairs` use**: filter to
`alignment.aligned == true` at minimum, then still spot-check per-source
before bulk training use (per both `housei`'s page-numbering mismatch and
`akokate`'s own `aligned:false` failure case above -- `aligned:true` is
necessary but not sufficient).

**Usable-as-is** (aligned:true pairs, confirmed clean on samples): `ako5`
(new, prior entry), `ako7`, `ako8`, `akokate`, `akocult`, `akocultB`,
`ako3`, `ako3B`, `fitness` (prior entry).
**Needs manual review before use, higher failure rate than `aligned:true`
filtering alone would catch**: `ako9`, `ako10`, `akogoods`, `ako4`,
`housei` (prior entry).
**No new content, skip**: `hamlabi` (both entries, prior entry), and every
superseded/duplicate entry in the dedup table above.
**Not yet inventoried**: `ako6` (confirmed usable content type, but not
deeply spot-checked beyond one page that happened to be a failure case --
treat as unverified pending a proper look, not yet in either bucket above).

### Next Actions

1. When paired-data work resumes, start from the confirmed "usable-as-is"
   list above (`ako5`/`ako7`/`ako8`/`akokate`/`akocult`/`akocultB`/`ako3`/
   `ako3B`/`fitness`), filtered to `alignment.aligned == true`, run through
   this project's own review/tiling pipeline (do not skip this project's
   own alignment gate just because `clip_pairs` provides one -- treat its
   diagnostic as a pre-filter, not a replacement, per the housei/akokate
   findings).
2. `ako9`/`ako10`/`akogoods`/`ako4` need a root-cause look at *why* the
   auto-extractor is failing so often on these specific works before
   deciding whether they're worth manually recovering or should be
   written off -- not started.
3. `ako6` needs one more proper spot-check pass (only one, unrepresentative
   failing sample seen so far).
4. This closes out the `clip_pairs` name-collision/inventory investigation
   for now. The full remaining ~62 non-`ako`/`housei`/`fitness`/`hamlabi`
   `clip_pairs` project folders (other artists' works / other series names
   entirely, never cross-checked at all) are still uninventoried --
   lower priority since none had a name collision to investigate, but
   represent the largest remaining unknown quantity in this archive.

## 2026-08-21 (later): External Feedback Report For The `clip_pairs` Extraction Tool

Per explicit user direction: this investigation's failure-pattern findings
were never expected to beat this project's own already-reviewed extractions
(mechanical auto-extraction, as expected) -- but sharing the specific
failure patterns with whoever runs the `extract_line_and_sketch`/`clip_pairs`
extraction tool could plausibly improve *its* output quality going forward.
Wrote `doc/preprocess/clip_pairs_extraction_feedback_20260821.md`, an
external-facing report (Japanese, matching the user's own working language)
distinct from this log's internal record: catalogs the 5 concrete failure
patterns found across both investigation passes (near-blank line layer
despite full sketch -- `ako9`/`ako10`; solid-fill/shadow layer picked
instead of linework -- `akogoods`; sketch/line production-stage mismatch --
`ako4`; partial single-panel extraction -- `akokate`'s `aligned:false`
sample; cross-export page-number/content mismatch -- `housei`), each with
exact zip paths as reproducible examples, plus concrete tool-side
improvement suggestions (ink-ratio-based blank-layer detection, line-vs-fill
layer discrimination, sketch/line ink-ratio-divergence sanity check,
page-coverage-area validation, stable cross-export page identifiers) and a
short list of what worked well as a positive baseline. Added to
`doc/README.md`'s preprocess index, flagged there as external-facing (not
just internal reference like the rest of that section).

**Delivered 2026-08-21 (later still)**: the user opened a Claude Code
session on the Windows extraction-tool machine specifically to receive
this. Checked cross-machine Remote Control messaging first (`ListAgents`,
then a `claude-code-guide` agent lookup of the exact setup steps) --
confirmed unusable here: Claude Code's cross-session messaging does not
work on native Windows (only WSL2), and the user's Windows session is
native. Fell back to SCP in the pull direction instead (Windows machine
fetching from this machine, avoiding any need for an SSH server or
credentials on the Windows side): confirmed this machine's `sshd` active
on port 22, LAN IP `192.168.1.34`, gave the user the exact `scp
sh1@192.168.1.34:/home/sh1/deepl/lineart/doc/preprocess/
clip_pairs_extraction_feedback_20260821.md <dest>` command to run from
Windows. User confirmed receipt. This closes out the delivery step; no
further action needed on this specific report unless the extraction tool
maintainer responds with follow-up questions.

## 2026-08-21 (later still): clip_pairs Full Inventory Completed -- 48 Remaining Folders, `aligned:true` Downgraded From "Reliable" To "Useful But Insufficient"

User asked to continue with the uninvestigated remainder of `clip_pairs`
(the 48 folders with no name collision against a known source). Same
methodology as the `ako*`-family pass: mechanical CRC dedup first, then
manifest pair/aligned-fraction stats, then parallelized visual spot-checks
(3 forks this time, ~10 works each, lighter-weight one-sample-per-work
given the generally healthier aligned-fraction stats than the `ako*`
family showed).

**`4th` name-collision resolved** (3 folders: `047_2022_4th`/
`051_2022b_4th`/`059_2023b_4th`): same manuscript as the already-processed
`dataset_4th_koma.zip`, not new content. Per-page CRC check against the
known 36-page archive found several pages (21/22/26/27/31/32 for the two
larger folders) have non-matching bytes at the same page number; direct
visual comparison of `page0021`'s `line` output confirmed the
`housei`-style severe-degradation failure -- the known archive's page0021
is a full hospital-room dialogue page, `clip_pairs`' version is almost
entirely blank (only a bed-frame outline and a prosthetic-leg object
survive). Do not use these; existing `dataset_4th_koma.zip` stays
production.

**`skima` name-collision resolved** (`022_2018_skima10`/
`035_2019b_skima2`, 10 pages combined): unrelated to the known 626-page
`unpaired_rough/skima` pool (rough-only by design, no line ever produced)
-- these two are a much smaller, different source that happens to share
the nickname and does have real pairs. Negligible size, not pursued
further.

**Remaining 43 folders inventoried, collapsing to ~29 distinct works**
after CRC dedup (`gakusai3`+`gakusai1-3DL`, `comics4`'s 3-work internal
compilation with one sub-work duplicating standalone `underworld`,
`toramusume1`+`toramusume2`, `UNI`+`UNI_ml` -- partial-overlap siblings,
not pure duplicates, confirmed same series with at least one duplicated
title/cover page across both exports -- and a 3-way `nurse` export
cluster). Visual spot-check across all ~29 (3 parallel forks): confirmed
varied genuine content, ~40% sequential multi-panel manga and the rest
single illustrations/reference sheets/design sheets -- no `ako5`-style
content-type surprise.

**Important revision to the `aligned:true` reliability finding from the
`ako*`-family pass**: this round found the first confirmed
counterexamples. `014_2017_kacho`'s `aligned:true` sample has both
sketch and line showing unfinished white-void patches (wrong production
stage on both sides). `029_2019_fringe3`'s `aligned:true` sample pairs a
hospital/dialogue manga page (`sketch`) with an entirely unrelated *color
instruction sheet* (`line`) -- a new failure pattern beyond the ones
already catalogued (blank layer / wrong-layer-type / partial-page):
**completely unrelated documents mispaired**, undetected by the alignment
diagnostic. `043_2021_hero`'s `aligned:true` `page0006` has a full sketch
but an almost-blank line (the familiar near-blank pattern, this time on an
`aligned:true` entry); its `aligned:false` `page0002` was worse still --
sketch and line show *completely unrelated scenes*, looking like a
page-index mixup rather than a layer-selection failure. `073_2024c_mayer`
also showed a debatable/loosely-stylized `aligned:true` correspondence.
**Revised rule**: `alignment.aligned == true` is a useful coarse filter
(removes most broken pairs) but confirmed **not sufficient on its own** --
false positives are real and recur across multiple unrelated works, not
isolated noise. Any bulk `clip_pairs` use still needs per-page or at least
per-source-sampled verification beyond the manifest's own flag.

This closes the full 74-folder `clip_pairs` inventory. Full per-work
disposition table (usable / needs-review / no-new-content) recorded in
`doc/preprocess/raw_dataset_storage_policy.md`.

### Next Actions

1. Update `doc/preprocess/clip_pairs_extraction_feedback_20260821.md` with
   the new "unrelated document mispaired" failure pattern and the revised
   `aligned:true`-insufficiency finding before next sending it (or a
   follow-up) to the extraction tool's maintainer -- done, see the
   following entry.
2. When paired-data pipeline work actually resumes, the practical starting
   set is the confirmed-clean-sample works across both the `ako*` pass and
   this pass; still budget for a per-source review pass, not a blind bulk
   ingest, given the `aligned:true` insufficiency finding above.
3. No further `clip_pairs` inventory work is planned -- the archive is now
   fully catalogued at the folder level.

## 2026-08-21 (later still): Feedback Report Updated -- Requested The Extraction Tool Add A Koma-Layer Export

Discussed the pair-creation outlook from the current `clip_pairs` +
existing-sources dataset: rough order-of-magnitude estimate of ~600-900
usable pages across the confirmed-decent works (undercounted for a few
multi-subproject folders not fully tallied, e.g. `gakusai1-3DL`'s other 2
subprojects, `ako6`'s other 9 subfolders), versus the existing
1489-tile `combined_koma_20260729` pool -- plausibly a 1.5-2x tile-count
uplift if pipeline yield tracks `housei`'s ~4.7 tiles/page ratio, but
tempered by the `aligned:true`-insufficiency finding (real usable rate
likely lower) and by a real pipeline gap: unlike the 5 existing
koma-pipeline sources, `clip_pairs` pages carry no dedicated koma
(panel-border) layer, so `match_koma_panels.py`'s per-panel
translation+scale alignment search can't be applied directly -- would need
either adapting `psd_line`'s flattened-raster `recursive_xy_split` panel
detector (a strictly weaker fallback, dependent on panel borders actually
being drawn in the flattened image) or falling back to whole-page matching
(already known from this project's own history to underperform per-panel
matching).

User's read: the koma-layer gap is significant given how central it was to
fixing this project's own "residual misalignment" root cause on the 5
existing sources (2026-07-26/27-29 entries above) -- and improving the
extraction tool to provide it could meaningfully raise `clip_pairs`' (and
any future export's) realized quality, not just this project's own
downstream handling of it. Updated
`doc/preprocess/clip_pairs_extraction_feedback_20260821.md` with a new
top-priority section (above the existing 6 failure-pattern-driven
suggestions) explaining the causal link established in this project's own
history (per-panel scale/translation alignment fixed a real production-
stage geometric transform that whole-page matching couldn't correct;
naive flattened-raster panel-border detection failed on a real page,
misidentifying character hair as a border) and formally requesting the
`extract_line_and_sketch` tool also export a koma/panel-border layer
alongside sketch/line, using the same auto-selection-by-name-pattern
convention already used for sketch/line layer selection.

### Next Actions

1. Not yet done: re-sending the updated report to the Windows-side session
   -- same SCP pull command as before (`doc/work_log.md`'s 2026-08-21
   delivery entry) would work if the user wants to resend now that it's
   been revised twice since first delivery.
2. If the koma-layer export request is implemented on the tool side, this
   project's next step would be adapting `match_koma_panels.py` (or a
   generalized version of it) to consume clip_pairs-style koma layers
   directly, following the same pattern already proven on the 5 existing
   sources -- not started, blocked on the tool-side change.

## 2026-08-21 (later still): Line-Domain LoRA Pool Expansion With `psd_line` Tiles -- Rejected, Severe Texture Collapse

With paired-data work deferred to re-extraction (pending the koma-layer
request above), picked up the "unpaired data" follow-up discussed
separately this session: the already-extracted but never-used 2022
`dataset/psd_line_koma_extraction_20260809/line` tiles (saved 2026-08-09,
sitting idle since) were merged into the line-domain LoRA's training pool
alongside the existing 1489-tile `line_combined_koma_20260729` (3511 total,
2.36x the original pool), to test whether more/varied line-domain data
improves fidelity to the real reference distribution.

**Pipeline**: `experiments/run_domain_lora_line_sd15base_sksv2_expanded_psdline_20260821.sh`
(new) -- WD14-tagged the 2022 untagged psd_line tiles with the same
`sks style, monochrome line art, manga panel, black and white` caption
suffix as the adopted config (`--caption-suffix` flag on `tag_wd14.py`, no
new tooling needed), merged with the existing `tags_sksv2.csv`, smoke-
tested (6 steps), then trained the exact same recipe as the adopted
`domain_lora_line_sd15base_sksv2_20260807` (rank16 attn-only, SD1.5 base,
10 epochs) with only the data pool changed -- isolating "more/varied data"
as the one variable, per this project's isolation-experiment methodology.
Ran detached (`nohup ... & disown`, verified `PPID=1`), ~7h total (4380
steps at ~3.87s/step for training alone).

**Result: rejected, both by metric and by direct visual comparison.** Ran
`measure_lineart_profile.py` with all three sources in one invocation for a
rigorous same-run comparison (`koma_ref` / `adopted_sksv2_original`
[samples from the still-on-disk original checkpoint] /
`expanded_psdline`), not just comparing against remembered numbers from a
different session:

| metric (median) | koma_ref | adopted (original) | expanded (+psd_line) |
|---|---:|---:|---:|
| background_ratio | 0.945 | 0.786 (17% dev) | 0.548 (**42% dev, worse**) |
| long_component_ratio | 0.819 | 0.751 (8% dev) | 0.447 (**45% dev, much worse**) |
| components_per_1k_ink_px | 10.91 | 8.33 (24% dev) | 22.64 (**108% dev, much worse**) |
| grid_ink_cv | 1.878 | 1.285 (32% dev) | 0.670 (**64% dev, worse**) |
| blank_cell_fraction | 0.688 | 0.281 (59% dev) | 0.063 (**91% dev, worse**) |
| faint_near_ink_ratio | 0.903 | 0.782 (13% dev) | 0.964 (7% dev, better) |
| line_width_p50 | 3.82 | 5.73 (50% dev) | 5.73 (50% dev, identical) |

Most macro-structure axes got substantially worse, not better; only 2 of
14 axes improved. `line_width_p50`'s median landing on the exact same
value (5.7300) in both independent runs was checked for a caching/reuse
bug (different training runs, different checkpoint dirs, different
samples, same script otherwise) -- looks like a real coincidence /
recipe-level attractor rather than a bug, not investigated further since
it doesn't change the overall verdict.

Direct visual comparison of both scale1.4 contact sheets (per this
project's standing rule to never trust metrics alone) confirmed the
regression is severe, not borderline: the original adopted checkpoint
still produces recognizable anime faces/figures with clean linework; the
expanded checkpoint has **collapsed into a dense vertical-hatching/striping
texture** across nearly every one of the 16 tiles, with little to no
recognizable face/figure content remaining -- the same qualitative failure
mode as the original (pre-isolation-chain) rough-domain LoRA's
"parallel-hatch collapse" (2026-08-05/06 entries above), just now
appearing on the line domain instead.

**Root-cause spot check**: sampled raw `psd_line` tiles directly (not LoRA
output) and found several do contain long near-vertical parallel-line
content -- hair strands, crosshatch shading -- plus at least one confirmed
instance of the faint gray calibration-strip artifact already flagged as a
known caveat for this source (2026-08-09 entry above, "0023"-prefixed
page). Plausible mechanism: this content pattern, still a minority of the
pool, was enough to shift the LoRA's learned distribution toward a
repeating-stripe attractor when trained from scratch on the pooled data
(not incrementally fine-tuned), similar in kind to the earlier
data-composition-skew story on the rough domain (2026-08-06 entry,
`ako*`/`housei*`/`komadense` prefixes dominating 74% of that pool).

**Decision**: do not adopt. `domain_lora_line_sd15base_sksv2_20260807`
remains the production line-domain checkpoint, unchanged. The expanded
checkpoint/samples are kept on disk as a negative-result reference (not
deleted), per this project's policy of preserving failure-mode evidence.

### Next Actions

1. ~~Not pursued this session: filtering `psd_line` tiles~~ -- done, see
   the following entry.
2. `dataset_comicstudio_line.zip` (the larger, ~3104-page unpaired line
   pool discussed earlier this session) should not be added to the
   line-domain pool naively given this result -- if pursued, needs the
   same kind of composition/content-density scrutiny that fixed the rough
   domain's earlier collapse, not a blind merge.
3. `domain_lora_line_sd15base_sksv2_20260807` stays the reference for any
   downstream use (e.g. a future ControlNet-adjacent style adapter) --
   unaffected by this rejected experiment.

## 2026-08-21 (later still): Filtered-Pool Retry -- Partial Improvement, Still Not Adopted

User's hypothesis after seeing the rejected expansion's visual collapse:
the vertical-hatch texture looked like the LoRA was "over-optimizing some
metric." Clarified the actual mechanism (no adversarial/reward loss exists
in `train_domain_lora.py` -- plain diffusion denoising MSE on real images,
this project's evaluation metrics are never fed back into the training
objective) but verified the substance of the intuition directly: computed
per-tile `measure_lineart_profile.py` stats for the raw `psd_line` pool vs
the original `line_combined_koma_20260729` pool (500-sample comparison,
`results/lineart_profile_orig_vs_psdline_raw_pools_20260821.csv`). Found a
real, large structural skew -- `psd_line`'s `components_per_1k_ink_px`
median (34.2) is over 3x the original pool's (10.9), `long_line_ratio`
median 0.89 vs 0.78, `faint_of_drawn_ratio` median 0.000 vs 0.383 (psd_line
is almost purely binary ink, no soft/faint pixels at all). Mechanism:
plain MSE denoising loss is inherently easier/lower-loss to fit on
repetitive, low-entropy content (parallel hatching) than complex unique
content (faces/figures) -- pooling a source skewed toward the former can
pull a capacity-limited LoRA's learned distribution toward that "cheap"
mode, the same class of failure as the original rough-domain "parallel-
hatch collapse" (2026-08-05/06 entries), different specific trigger
(content-composition skew here vs. missing-caption-signal there, since
this run already had per-image WD14 captions).

**Filtered retry**: computed full-pool per-tile stats
(`results/lineart_profile_psdline_full_20260821.csv`, all 2022 tiles) and
kept only tiles falling within the original pool's own p90/p10 bands on
`long_line_ratio`/`long_component_ratio`/`components_per_1k_ink_px` --
**510 of 2022 (25.2%) passed**, revealing this isn't a minority-outlier
problem but a whole-source structural skew (median of the full psd_line
pool already exceeds the original pool's own p75-p90 range on the
fragmentation axis). Symlinked the passing 510 into
`dataset/psd_line_koma_extraction_20260809/line_filtered_20260821/`,
built a merged caption CSV (1489 + 510 = 1999 rows,
`results/domain_lora_line_captiontags_20260807/tags_sksv2_filtered_psdline_20260821.csv`),
and reran the exact same recipe via
`experiments/run_domain_lora_line_sd15base_sksv2_filtered_psdline_20260821.sh`
(+34% pool size instead of the rejected attempt's +136%).

**Result: real but partial improvement, still not adopted.** Metrics vs
`koma_ref` (median deviation, filtered vs. the two prior variants):

| metric | adopted (original) | rejected (full expansion) | this filtered retry |
|---|---:|---:|---:|
| long_component_ratio | 8.4% | 45.4% | **1.8% (fixed)** |
| components_per_1k_ink_px | 23.6% | 107.6% | **14.9% (fixed)** |
| line_width_p50 | 50.0% | 50.0% | **21.7% (fixed)** |
| faint_near_ink_ratio | 13.4% | 6.9% | **3.4% (fixed)** |
| background_ratio | 16.9% | 42.0% | 26.6% (still worse than original) |
| grid_ink_cv | 31.6% | 64.3% | 64.0% (unfixed) |
| blank_cell_fraction | 59.1% | 90.9% | 90.9% (unfixed) |

The two axes most implicated in the original collapse
(`long_component_ratio`, `components_per_1k_ink_px`) came back to
near-parity with the adopted checkpoint or better -- confirms the
composition-skew diagnosis was the right mechanism. But
`grid_ink_cv`/`blank_cell_fraction` stayed exactly as bad as the rejected
full-expansion attempt.

**Visual check** (`results/domain_lora_line_sd15base_sksv2_filtered_psdline_20260821_scale14/`)
confirmed the mixed numeric picture directly: roughly 5-6 of 16 tiles
still show the vertical-stripe/barcode collapse (plus one grid/lattice
artifact), the rest show recognizable faces/hair/shoulders -- a real
reduction from the rejected attempt's near-total collapse, but not a fix.
This matches the `grid_ink_cv`/`blank_cell_fraction` axes staying poor:
those measure exactly "is the canvas filled edge-to-edge with a repeating
texture," which a persisting minority of striped tiles keeps failing.

**Decision**: do not adopt. Diminishing-returns judgment call: a stricter
filter threshold would likely reduce the collapse rate further but shrink
the usable psd_line subset well below 510 tiles, at which point the
addition stops being a meaningful pool expansion. `domain_lora_line_sd15base_sksv2_20260807`
(1489-tile, unmodified) remains production. Not pursuing a third,
stricter-filter iteration this session.

### Next Actions

1. Not pursued: a stricter filter pass (e.g. p75 bands instead of p90) to
   see if the collapse rate keeps falling roughly linearly with pool
   purity, or whether it plateaus -- would clarify whether this is purely
   a composition-dose effect or hits a floor.
2. `dataset_comicstudio_line.zip` still needs the same
   composition-analysis step (`measure_lineart_profile.py` on its raw
   tiles vs `koma_ref`) before any pooling attempt, now with a concrete
   precedent for what a bad structural skew looks like and how partial the
   fix can be even after filtering.
3. `domain_lora_line_sd15base_sksv2_20260807` stays the reference line-
   domain checkpoint; both rejected/partial expansion attempts (checkpoints,
   samples, profile CSVs) kept on disk as negative/partial-result evidence,
   per this project's evidence-preservation policy.

## 2026-08-21 (later still): Extraction Tool Reply -- Koma Layers Delivered, Deep QC Pass, 18.6% Cross-Source Duplication Found

The extraction-tool maintainer replied to `doc/preprocess/clip_pairs_extraction_feedback_20260821.md`
via a new file-drop convention: `inbox/reply_clip_pairs_20260821.md`
(24KB, very thorough) plus an updated `dataset_clip_pairs.zip` at the repo
root. Archived the zip as `dataset/raw_zips/dataset_clip_pairs_v2.zip`
(11.4GB, 5360 entries, confirmed 1644 koma files present via direct
inspection). This is a substantial, high-quality response -- summarizing
the key points:

### 1. Koma-layer request: fulfilled, 94.4% coverage

The **top-priority ask from the prior feedback report was implemented**:
`{prefix}_{page}_koma.jpg` now ships alongside line/sketch, same
folder/naming/canvas-size convention, for 1644 of 1741 pairs (94.4%).
Turns out the tool already had koma extraction (used previously for
`kazenagare`/`ako5`/`kurip`/`hamlabi`/`gakuen`/`4th` -- i.e. exactly this
project's 6 already-known sources), but applying it to the much larger
`clip_pairs` set surfaced **two real bugs never caught by the smaller
prior usage**: (a) folder-type koma layers sometimes carry a baked-in
composited mip-map that, when rasterized, returns actual panel artwork
instead of border lines (concrete example: `aljanne2 page0003` measured
32.7% ink before the fix, 1.2% after -- matching the ~1.3-2.0% range
measured on other sources); (b) full-bleed panels (edges touching the
page boundary) were dropped entirely because vertex-clipping to canvas
bounds destroyed their polygons. Both fixed; re-checked against `4th`'s
existing koma output with no regressions (32/33 pages identical, 1
improved by recovering a previously-dropped bleed panel).

Of the 97 pages without koma: 79 genuinely have no koma layer in the
source file (artist didn't use one), 15 are illustration/single-cut works
excluded as out-of-scope by design (`fujiko2`/`hitoduma`/`x2`/`udonge`/
`circlecut08`/`bakugi1H1H4`/`bakugi2H1H4`/`tama`/`toramusume1`, plus
`skima10`/`UNI_ml` which look like manga but lack a koma layer, still
under investigation on their side), and only 3 are true gaps (`running`
x2, `succor` x1 -- a known, understood, deliberately-unfixed edge case:
a panel vertex sitting at y=0.001 falls through the coordinate filter;
judged not worth the refactor for 0.17% of pages, will fix if we need it).
Per-page koma presence is recorded in each folder's `koma_manifest.json`
(`koma: null` = absent).

### 2. QC block added to every manifest entry (proposal 1, corrected)

Corrected a factual error in the prior feedback report: `manifest.json`
entries never had an `ink_ratio` feature field -- that was misremembered
from a different source (`dataset_psd_line`'s `index.tsv`). Built QC
measurement from scratch instead, appended (not replacing) a `qc` block to
every entry: `pair_quality` (ok/line_fragment/sketch_fragment),
`line_ink_ratio`/`sketch_ink_ratio`, `sketch_cell_coverage`/
`line_cell_coverage` (grid-cell coverage fractions, the primary axis),
`line_bbox_fraction`/`sketch_bbox_fraction`, `grid_correlation`,
`content_fingerprint`, `measured_at_max_side` (1024px -- all 1742 pairs
measured at reduced resolution for tractability; ink ratios read higher
than full-res measurements would, not directly comparable to this
project's own full-resolution numbers, but valid for relative
page/layer comparison). Also exports `config/clip_pairs_qc.csv`, one row
per pair, with `review_rank` (ascending `grid_correlation`, most
suspicious first) and dedup fields (below).

**Calibration honesty worth noting**: their first threshold attempt
(tuned against our 8-good/3-bad work-level labels) failed badly (18% false
positive on good works, only 54% recall on bad ones) -- but traced this to
*our* labels being too coarse, not their detector: spot-checking
low-coverage pages from works we called "clean" (`ako3B`, `hamlabi`) found
the exact same page-level `line_fragment` failure as our own flagged
`ako9 page0003` example, with an essentially identical measurement
signature. **This failure mode is scattered across individual pages
within otherwise-good works, invisible at work-level granularity** -- 39
of 69 folders have zero flagged pages, the rest have some. Confirms/refines
this project's own 2026-08-21 (later) inventory finding that `aligned:true`
being insufficient wasn't isolated noise -- here it's the same story one
level down (page-level QC needed, not work-level).

Cross-check against our specific reported examples:

| page | our pattern | their `qc` verdict |
|---|---|---|
| `ako9 page0003` | 1 (blank line) | `line_fragment` (caught) |
| `ako10 page0001` | 1 (blank line) | `line_fragment` (caught) |
| `housei page0001` | 5 (fragment) | `line_fragment` (caught) |
| `ako4 page0002` | 3 (sketch blank) | `sketch_fragment` (caught) |
| `ako9 page0010` | 4 (partial page) | `ok` (**missed** -- background furniture covers 17.7% of sketch cells, statistically continuous with normal sparse pages) |
| `hero page0002` | 6 (unrelated doc) | `ok` (**missed**, expected -- see below) |

### 3. Decision: flag, don't delete -- and why `aligned:true` alone is the wrong gate

Explicitly did **not** delete or quarantine flagged pairs -- left the
cutoff decision to us. Counts: `ok` 1646 (94.5%), `line_fragment` 60
(3.4%), `sketch_fragment` 35 (2.0%). Cross-tabulated against `alignment.
aligned`: of 1400 `aligned:true` pairs, QC additionally catches 29; of 341
`aligned:false` pairs, **275 are QC-clean** -- i.e. `aligned` runs too
strict and would discard a lot of good pages if used as a hard gate on its
own, consistent with (and now quantified beyond) this project's own
2026-08-21 finding that `aligned:true` is necessary-but-not-sufficient.
Their concrete recommendation: **filter on `qc=ok` + dedup + koma-presence
(not `aligned`) as the starting pool** if we're going to do our own
koma-based alignment refinement anyway.

Filtering funnel (before/after the 289-page re-extraction fix in section
5 below):

| condition | before fix | after fix |
|---|---:|---:|
| total | 1742 | 1741 |
| `qc=ok` | 1459 | 1646 |
| `qc=ok` + unique (deduped) | 1215 | 1343 |
| `qc=ok` + unique + has koma | 1151 | **1272** |
| `qc=ok` + unique + `aligned=true` | 1039 | 1149 |

**Note**: their own prose recommendation quotes "1,151ペア" and "1,039"
as the koma-available/aligned-true starting points, which are the
*before-fix* column values, not the *after-fix* ones (1272/1149) shown in
their own table -- likely a stale reference left in from before they
finalized the re-extraction numbers. Flagged this back to them in the
outbox reply; treat **1272** (qc=ok + unique + has koma, post-fix) as the
correct current starting-pool size for this project's own use unless they
say otherwise.

### 4. Pattern 6 (unrelated-document pairing): confirmed genuinely undetectable automatically

Tested our own suggested fix (coarse low-frequency/perceptual-hash
correlation) and got the **opposite** of the expected result: bad pairs
(`kacho` 0.48, `fringe3` 0.44) scored *higher* correlation than several
good ones (`ako7` 0.44, `akocultB` 0.42, `fitness` 0.29, `hamlabi` 0.14,
`ako3` 0.09). Root cause: this isn't "line/sketch are different pages" --
it's "the wrong layer was selected from *within the same `.clip` file*"
(e.g. a color-instruction sheet or character reference sheet that's a
separate artwork living inside the same file as the real manga page).
Same artist, same style, same canvas size, similarly-distributed coarse
ink density -- nothing a low-frequency/pixel-statistics check can
separate; it needs semantic content understanding. **Correctly declined to
oversell a fix that doesn't work**, providing `review_rank` (grid_correlation
ascending) for manual-review prioritization instead. Confirmed our
flagged `hero page0002` ranks 240/1742 and `ako9 page0010` ranks 391/1742
by this ordering -- both within the top ~15% most-suspicious, i.e. the
ranking is still a real, useful triage signal even though it can't be a
hard automated filter.

### 5. Root-caused and fixed the `line_fragment` mechanism (not just detected it)

Went beyond the original ask (make existing data usable) into actually
fixing 3 stacked bugs in the extraction logic itself, then re-ran 289
pages: `ok` rate 83.8% -> 94.5% (1459 -> 1646), `line_fragment` 14.4% ->
3.4% (251 -> 60), 3D-render-as-sketch cases 8 -> 0. **Zero regressions**
(0 pages went `ok` -> flagged).

- **Mechanism 1**: the extractor's layer-compositing logic let a single
  name-matched layer (literally named "線画") suppress every
  statistically-detected candidate layer, even a tiny fragment against a
  much larger correct unnamed layer (`akogoods page0012`: a 0.0072-ink
  named fragment blocked a 0.1597-ink real layer, 22x larger). Fixed by
  compositing both name-matched and statistically-matched layers together
  (with a role-based exclusion list to prevent contamination -- see below).
- **Mechanism 2**: `black_among_drawn` upper bound (0.88) was excluding
  genuine 1-bit line art with zero anti-aliasing; relaxed to 0.98 (not
  1.00 -- fully-1.00 layers are confirmed solid-fill/speech-bubble
  layers, not line art; empirically 1.00 would have wrongly pulled in 519
  extra layers across 74% of otherwise-good pages). Also relaxed
  `white_ratio` floor 0.84 -> 0.80 (the `akogoods` fragment above missed
  the old floor by 0.001).
- **Mechanism 3**: 3D-render exclusion previously matched only exact names
  (`lineart`/`shadow`/`ao`/`base`); missed an entire naming family
  (`rendering_shadow_back0001` etc.) across 64 pages in 22 works (17 of
  which had zero exact-name matches, so detection silently did nothing).
  Real impact found: 8 pairs had a 3D shadow pass delivered as "sketch" --
  7 corrected to genuine sketch content, 1 pair dropped (no non-3D sketch
  candidate existed for that page).
- **Side-effect caught and fixed**: naively compositing statistically-
  matched layers (mechanism 1's fix) started pulling in shadow/color/AO/
  koma-border/rough layers on 33 pages -- and this looked like an
  *improvement* on the QC coverage metric (more ink -> higher coverage),
  a metric-blind-spot the tool's own author caught by actually reading
  layer names, not trusting the number. Fixed by excluding role-named
  layers (`mask`/`color`/`tone`/`shadow`/`bg`/`コマ`/`rough`/`sketch`/`ao`/
  `base`/`用紙`/`フキダシ`, Japanese included) from the statistical
  compositing candidate pool while still admitting genuinely-named
  "線画" layers unconditionally.

Remaining unfixed: 36 `line_fragment` + 35 `sketch_fragment` cases where
the content genuinely doesn't exist in the source page (not a selection
bug); pattern 6 (declined, see above); and partial-improvement cases like
`housei page0012` (coverage 0.004 -> 0.318 but visually only the top half
of the page has real linework) -- consistent with this project's own
"housei's clip_pairs version stays worse than our existing archive"
finding (below).

### 6. `housei` page-numbering "mismatch": explained as a real numbering-scheme difference, not a bug

Our flagged finding (`clip_pairs` `page0001` sketch showing unrelated
content to the existing `housei_NNN` archive's `page0001`) is expected
behavior, not corruption: `clip_pairs` page numbers are the literal
original `.clip` filename; the existing `dataset_housei` archive's
`housei_NNN` numbers come from a **different, previously-reconciled
numbering scheme (this project's own prior block-density-correlation
brute-force matching, which found the old archive's numbers were off by
+1 from the manuscript's own layer numbering)** -- both are internally
consistent, just on different axes. Added `source_path`/`source_size`
(absolute path + byte size of the original `.clip` file) to every entry
per our proposal 5, so future matching can go through the source file
identity instead of page-number-guessing. Declined full SHA-256 hashing by
default (1742 files, ~90GB read cost) -- available on request.

**Confirms our original recommendation stands**: `066_2024_housei` in
`clip_pairs` is 10 pairs, **all 10 flagged `line_fragment` (100%)** --
worse than initially estimated. The existing manually-reviewed
`dataset_housei.zip` (18 pairs + 18 koma, fully visually verified, with
several manual per-page selection overrides this source specifically
needs) remains the correct source for housei; do not use `clip_pairs`'s
housei entry.

Comparison table for the other name-collision sources (clip_pairs is
*not* uniformly worse -- only housei is):

| our existing archive | pairs | `clip_pairs` equivalent | pairs |
|---|---:|---|---:|
| `dataset_housei` | 18 | `066_2024_housei` | **10 (worse)** |
| `dataset_ako5` | 48 | `011_2017_ako5` | 66 |
| `dataset_hamlabi` | 13 | `063_2023b`/`070_2024b` | 18 each |
| `dataset_kurip_v4` (fitness) | 38 | `069_2024b_fitness` | 80 |
| `dataset_4th` | 33 | `047`/`051`/`059` | 10/27/26 |
| `dataset_gakuen` | 16 | (not in the 74-folder set) | -- |

Offered to send any of these manually-reviewed archives if useful --
not needed, we already have all 6 (they're this project's own existing
koma-pipeline sources).

### 7. New finding not in our original report: 18.6% cross-work-id content duplication

Found while building the dedup/QC pipeline, not something we'd flagged:
**266 duplicate groups covering 324 of 1742 pairs (18.6%)** are
byte-identical content filed under different `work_id`s -- the same
source `.clip` files physically duplicated across multiple year-tagged
folders on their end (F: drive), with the extraction tool assigning a
fresh `work_id` per folder path rather than per physical file. Affects 9
work families: `ako7`/`akogoods`/`4th`/`nurse`/`akocult`/`hamlabi`/`ako3`/
`ako3B`/`ako10` -- directly explaining several of our own earlier
mechanical CRC-based dedup findings (`070_2024b_hamlabi` matching our
archive byte-for-byte, the `akogoods`/`ako10`/`ako7` 100%-overlap clusters
from the 2026-08-21 (earlier) `ako*`-family inventory). `content_fingerprint`/
`duplicate_group`/`duplicate_count`/`is_primary` added to the QC CSV;
taking `is_primary`-only collapses to ~1418-1422 unique pairs (exact
number shifts slightly with the post-fix pair count). Not deduped on
their end by design -- left the choice of which duplicate to keep to us.

### Decision / Next Actions

1. **Requested (outbox reply, `outbox/reply_clip_pairs_qc_request_20260821.md`)**:
   send the fully QC-enhanced 1741-pair local version (currently only on
   the tool maintainer's machine) -- the just-received `_v2` zip has koma
   layers but not yet the `qc`/`source_path`/`content_fingerprint` fields.
2. Flagged the 1151/1039 vs. 1272/1149 (before-fix vs. after-fix column)
   discrepancy in their own reply back to them for confirmation --
   treating **1272** (qc=ok + unique + has koma, post-fix) as this
   project's working number for now.
3. Once the QC-enhanced zip arrives: this is now the concrete unblock for
   resuming paired-data pipeline work (deferred earlier this session
   pending exactly this koma-layer delivery) -- next step is adapting
   `match_koma_panels.py` (or a thin wrapper) to consume `clip_pairs`-style
   koma layers directly, filtering to `qc=ok` + `is_primary` + has-koma
   (not `aligned=true`, per their recommendation) as the starting pool
   (~1272 pairs pre-panel-splitting), following the same per-panel
   translation+scale alignment approach already proven on the 5 existing
   koma-pipeline sources.
4. Confirmed we already hold all 6 of the manually-reviewed archives they
   offered to resend (ako5ver2/hamlabi/fitness/housei/4th/gakuen) -- no
   transfer needed there.
5. `066_2024_housei` in `clip_pairs` should still not be used (100%
   line_fragment even after their fix) -- keep `dataset_housei.zip` as the
   production housei source, unchanged from the prior entry's conclusion.

## 2026-08-21/22: `clip_pairs` Koma Panel-Alignment Search -- 1272 Pairs, 4848 Panels, Complete

Built `tools/pair_extraction/match_clip_pairs_koma_panels.py`, a new driver
that reuses `match_koma_panels.py`'s panel-detection/alignment-search
functions unchanged (`detect_panels`, `search_panel_alignment`,
`extract_scaled_roi`, etc., imported directly) but replaces its
single-`--zip-root` iteration with one driven off the extraction tool's own
`clip_pairs/clip_pairs_qc.csv` (work_id/slug/page/line/sketch/koma
filenames already resolved there, no per-folder manifest.json parsing
needed). Filtered to `pair_quality=ok` + `is_primary=True` + `koma`
present -- 1272 of 1741 pairs, verified this matches the extraction tool's
own stated count exactly.

Smoke-tested first (10 pairs, 6m30s, 42 panels, chamfer median
15.51->15.17, no errors) before committing to the full run. At ~39s/pair
the full 1272-pair run was estimated ~14h, so chunked into 100-pair
slices with `--append` (`experiments/run_clip_pairs_koma_panels_20260821.sh`,
matching this project's established chunking convention for long
extraction runs). Launched detached (`nohup ... & disown`, verified
`PPID=1`).

**Result**: completed cleanly in ~16.5h (21:32 2026-08-21 -> 14:01
2026-08-22), all 13 chunks, zero errors/tracebacks. 4848 panels across 57
work_ids (of the ~69 in the filtered pool -- some works evidently
contributed 0 qualifying panels, not investigated further). chamfer
median 20.04 -> 18.44, mean 23.79 -> 21.18; 32.2% of panels picked a
non-1.0 scale (confirms the per-panel scale search is doing real work,
not just translation).

**Script bug found during QC review**: `--qc-out`/`make_qc()` writes to a
static path every chunk invocation, so only the *last* chunk's first 60
panels survived in `results/clip_pairs_koma_panels_20260821_qc.png` --
every earlier chunk's QC montage was silently overwritten. Not a
correctness bug (the CSV/JSON accumulate correctly via `--append`), but it
meant the only surviving visual QC sample was from the tail of the list
(`071_2024b_kids`), not a representative spread. Worth fixing (per-chunk
QC output path, or a QC-off flag for chunked runs) if this driver pattern
is reused again.

**Fixed the sampling gap directly** rather than re-running: wrote
`tools/pair_extraction/sample_clip_pairs_koma_qc.py`, an ad-hoc script
that re-crops 24 evenly-spread rows from the final accumulated CSV
directly from the zip (joining back to `clip_pairs_qc.csv` for exact
filenames, then reusing `crop_page`/`extract_scaled_roi`/`overlay_edges`
with each row's own recorded scale/dx/dy) --
`results/clip_pairs_koma_panels_20260821_qc_spread.png`. Visual review
(project's standing rule: never trust chamfer/F1 alone) confirmed the
alignment is working correctly across a genuine cross-section of works
(`gakusai3`/`gal`/`pc`/`suc`/`ako5`/`gakusai1-3DL`/`pink`/`stoptime`/
`ako6`/`comics4`/`ako7`/`fringe3`/`succor`/`ako8`/`akogoods`/`akokate`/
`4th`/`hyoui_color`/`akocultB`/`ako3B`/`fitness`/`ako4`/`mayer`) -- most
rows show tight rough/line edge overlap after alignment. A handful of
weak cases (`akocultB page0048`, `ako3B page0025`: F1 stuck at 0.01-0.07)
turned out to be near-blank/text-only panels, not alignment failures; one
(`ako4 page0009`: chamfer improved 52.6->38.2 but F1 still only 0.33)
reconfirms the already-known `ako4` sketch/line production-stage mismatch
persisting even after the extraction tool's QC filtering -- expected,
matches the earlier finding, not a new problem. Low-content/low-F1 panels
like these are expected to be caught by the downstream tile-level gates
(same `ALIGNMENT_*` constants used on the 5 existing sources), not a
reason to distrust the panel-alignment stage itself.

### Next Actions

1. ~~Sub-region split~~ -- in progress, see the following entry.
2. Tiling (`tile_region_manifest_480.py`, unchanged gates) is the
   remaining pipeline stage after sub-region split finishes.
3. Consider a chamfer/F1 cutoff before sub-region splitting to drop the
   near-blank/low-content panels seen in the QC spread sample (e.g.
   `akocultB`/`ako3B`'s F1<0.1 cases) -- not decided yet, may just let the
   existing downstream tile-level gates handle it as they do for the 5
   existing sources.

## 2026-08-22 (later still): Materialize + Sub-Region Split Launched

**Materialize** (`tools/pair_extraction/materialize_clip_pairs_koma_panels.py`,
new -- `materialize_koma_panels.py`'s per-panel crop/align logic reused
unchanged, adapted for clip_pairs' multi-zip-root manifest by joining back
to `clip_pairs_qc.csv` for exact filenames per row, same pattern as the
match-stage driver): dry-run confirmed 4750 of 4848 panels pass the
existing `--max-chamfer 45` generous pre-filter (98%, matches the 5
existing sources' typical acceptance rate at this stage). Visual QC on the
first rows found mostly good matches with a few visibly poor ones in the
chamfer 27-31 range (`001_2015_gakusai3` page0035's panels) -- expected,
this stage is a cheap pre-filter, the real gate is `tile_region_manifest_480.py`'s
fixed `ALIGNMENT_*` constants downstream, unchanged from how the 5
existing sources work. Ran `--save` for real (~85 min, 9502 files written,
`dataset/regions_clip_pairs_koma_panels_20260822/manifest.csv`).

Added a `"housei": row["pid"]` alias field to the output manifest so the
unmodified `split_koma_panel_subregions.py`/`build_region_valid_masks.py`/
`tile_region_manifest_480.py` (which index `row["housei"]` directly as the
generic per-page id, a naming holdover from when housei was the only koma
source) work against this new manifest without any further code changes --
confirmed no other hardcoded `"housei"` column references exist in the
mask/tile scripts.

**Sub-region split**: smoke-tested `split_koma_panel_subregions.py`
directly (already generic, no `clip_pairs`-specific fork needed) on 20
panels -- 29 sub-regions, chamfer median 15.13 -> 13.97 after refinement,
5m18s (~16s/panel). Extrapolated to the full 4750 panels: ~21h. Launched
chunked (200-panel slices via the script's own native
`--offset-panels`/`--limit-panels`/`--append`,
`experiments/run_clip_pairs_koma_subregions_20260822.sh`) detached in the
background (verified `PPID=1`). `--max-chamfer 999` (effectively no
panel-level filter at this stage, matching the 5 existing sources'
convention of deferring the real gate to tile extraction).

### Next Actions

1. ~~Wait for the sub-region split~~ -- done, see following entries.
2. ~~build_region_valid_masks.py / tile_region_manifest_480.py~~ -- done,
   see following entries.
3. ~~Visual QC~~ -- done, see following entries.

## 2026-08-22/24: `clip_pairs` Pipeline Completed -- Masks, Tiling, Dedup Cleanup, 6978 Final Tiles

Sub-region split (`experiments/run_clip_pairs_koma_subregions_20260822.sh`,
launched 2026-08-22) completed cleanly after ~21h (offset chunking held up
across the full run, zero errors): 9210 sub-regions from the 4750
materialized panels.

**Masks**: `build_region_valid_masks.py` run unchanged (native settings,
`--support-px 20 --window 61 --expand-ignore 16 --close-ignore 16`,
matching the 5 existing sources) -- fast, ~0.36s/row from a 30-row smoke
test, full 9210-row run completed in ~31 min, all rows processed
successfully.

**Tiling**: `tile_region_manifest_480.py` run with the same native-strict
gate recipe as `run_koma_tile_pipeline.sh` uses for the 5 existing sources
(`--ink-min 0.012 --ink-max 0.08 --max-black-component-ratio 0.025
--max-thick-ink-ratio 0.015 --max-line-width-p50 6.0 --max-long-line-ratio
0.25 --max-soft-ink-ratio 0.40` -- the "ako5ver2-derived default" per
`doc/preprocess/dataset_status.md`, not housei's 0.50 relaxation, since
`clip_pairs` spans many different artists/styles unlike any single
existing source). **Process note**: `--limit` only truncates the *output*
tile count after the full candidate scan runs to completion -- it does
NOT limit input rows scanned, so an initial "smoke test" with `--limit
200` was actually running the full 9210-row scan the whole time (caught
mid-run via the per-50-row progress log, killed at row 900/9210, and
immediately relaunched for real with no `--limit` and correct production
output paths -- no time lost, since the expensive part is the scan, not
the truncation). Full run took ~5h (9210 regions, 119,216 raw candidate
tiles before dedup) plus the final dedup/save pass.

**Result**: 7383 tiles initially accepted --
`dataset/pairs_480/valid_train_clip_pairs_koma_20260823.txt`,
`dataset/pairs_480/train/line_clip_pairs_koma_20260823/`.

**Integrity audit found a new failure mode**: `audit_pair_dataset_integrity.py`
reported 705 findings (345 exact line-hash + 360 exact rough-hash
duplicates *within* the training list) -- every prior koma-pipeline source
reported 0 findings, so this was investigated rather than dismissed.
Traced example duplicate pairs back to their source pages via the tiles
CSV's `source_name` column: e.g. `018_2018_ako6/ako6/page0084` vs.
`018_2018_ako6/done5/ako6-29`, `018_2018_ako6/done/ako6-4` vs.
`018_2018_ako6/ako6/page0052` -- **different slugs within the same
work_id producing byte-identical tile crops**, not a pipeline bug. Root
cause: the extraction tool's `is_primary`/`content_fingerprint` dedup
(from the 2026-08-21 QC delivery) operates on *whole-page* content hashes,
so it only catches byte-identical full pages -- it does not catch the
partial-overlap case already known from the 2026-08-21 `ako*`-family
inventory (`019_2018_ako7` sharing ~59% of its pages with
`026_2019_ako7`/`031_2019b_ako7` without being byte-identical overall). An
unchanged background/margin region within two otherwise-different page
exports can still crop to an identical tile. Clustering the 705 findings
(union-find over the pairwise findings) gave 401 duplicate clusters / 806
tiles involved, spanning multiple works beyond just `ako6`
(`ako5`/`ako6`/`ako7`(x2 exports)/`kacho` all appeared) -- confirms this
is a general property of the partial-overlap pattern, not an `ako6`-only
quirk.

**Fix applied**: kept one representative per duplicate cluster, dropped
the other 405 tiles from the training list and deleted the corresponding
rough/line files (original pre-dedup list backed up as
`dataset/pairs_480/valid_train_clip_pairs_koma_20260823.txt.predup`).
Re-ran the integrity audit: **0 findings**. Final count: **6978 tiles**
(vs. the existing 1489-tile `combined_koma_20260729` pool -- roughly a
4.7x expansion).

**Visual QC** (project's standing rule: never trust an extraction result
without looking at it): reviewed evenly-spaced-sample and tail montages at
native resolution. Top/mid ranks (score 5.0-6.4) show tight, clean
rough/line correspondence (faces, hair, hands) with F1 0.6-0.99 -- as good
as the 5 existing sources' best tiles. Tail ranks (score ~2.5) show the
same "sparse/faint but not semantically mismatched" degradation pattern
already established as normal for this pipeline's low-score tail (housei,
fitness, etc.) -- no content mismatches or alignment failures observed.

### Next Actions

1. `clip_pairs` koma tiles (6978, `dataset/pairs_480/valid_train_clip_pairs_koma_20260823.txt`)
   are ready to use -- not yet incorporated into any training run
   (combined-pool mixing decision, per this project's standing rule of not
   casually concatenating sources without a stated rationale, is still
   open).
2. ~~Report the whole-page-vs-partial-overlap dedup gap~~ -- done.
   `outbox/note_partial_overlap_dedup_gap_20260824.md` pushed via SCP to
   the correct destination
   (`C:\Users\sh1\code\extract_line_and_sketch\inbox\`, confirmed via
   `dir` -- also confirmed the earlier 2026-08-22 misdelivered note is now
   present there too, so the user's manual fix-up landed correctly).
   Framed as FYI/no-action-required (this project's own dedup pass already
   handles it cleanly), with a tentative suggestion (reuse the existing
   `grid_correlation` block-based metric for block-level dedup, not just
   whole-page) offered but not requested.
3. `housei`/`ako9`/`ako10`/`akogoods`/`ako4` remain excluded from this
   pool via the upstream `qc=ok` filter (per the 2026-08-21/22 entries) --
   unaffected by this pipeline run.
4. Consider whether to also process the ~62 clip_pairs project folders that
   were part of the 1272-pair filtered pool but not individually
   deep-inspected in the earlier `ako*`-family-focused visual review --
   the tiling pipeline already ran against the full filtered pool
   regardless, so this is really just an open question about how much
   more manual spot-review is warranted before trusting the pool at scale,
   not a blocked pipeline step.

## 2026-08-24/25: The Deferred ControlNet LoRA Fine-Tune, Run For Real -- Completed Cleanly, Quality Still Poor

Per user direction, executed the 2026-08-08 decision now that real paired
data exists: LoRA fine-tune of the public `control_v11p_sd15s2_lineart_anime`
checkpoint (not a from-scratch `ControlNetModel.from_unet` copy) on real
pairs, with per-tile WD14 captions and the `lineart_anime` preprocessor
applied to rough tiles before conditioning -- all three elements of that
decision, assembled for the first time.

**Data assembly**: combined `combined_koma_20260729` (1489, existing) +
`clip_pairs_koma_20260823` (6978, this session's new pipeline run) = 8467
pairs (`dataset/pairs_480/valid_train_combined_all_20260824.txt`, line
tiles symlinked into `dataset/pairs_480/train/line_combined_all_20260824/`).
WD14-tagged the 6978 new tiles (`scripts/tag_wd14.py`, same caption suffix
as the existing captions file for consistency) -- **took ~8.8h**, much
slower than the earlier ~1.2-1.3s/img clean-machine benchmark (4.55s/img
observed), consistent with the CPU-thermal-throttling explanation already
recorded for a similar slow run on 2026-08-01. Merged into
`dataset/pairs_480/captions_combined_all_20260824_wd14.csv` (8467 rows).

**New preprocessing tool**: `tools/pair_extraction/preprocess_lineart_anime_condition.py`
-- no such batch tool existed before (the `lineart_anime` detector had only
ever been invoked ad hoc, e.g. inside `condition_roundtrip_fidelity.py`'s
eval code and one-off 2026-08-08 diagnostic samples). Reuses the exact
`LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")` invocation
already validated there. GPU-accelerated, fast (~0.04s/img once warm; full
8467-tile pool took 6.3 min) -- confirms the earlier WD14 tagging slowness
was CPU-specific (onnxruntime CPUExecutionProvider), not a general
machine-load problem at the time. Output:
`dataset/pairs_480/train/rough_lineart_anime_20260824/`.

**Training**: `experiments/run_controlnet_lora_realpairs_20260824.sh`
(new, modeled on the 2026-08-08 pseudo-pair bootstrap script) -- smoke
test (6 steps) passed, then the full run: rank16 LoRA on
`control_v11p_sd15s2_lineart_anime`, lr=1e-4, 10 epochs = 10,580 steps
(1058 steps/epoch, 8467 tiles), ~3.1-3.15s/step, **completed cleanly in
~9.2h with zero errors**. Final checkpoint:
`checkpoints/controlnet_lora_realpairs_20260824/final`.

**Eval bug found and fixed same-day**: the script's own eval step invoked
`infer_controlnet.py` with `--rough-dir dataset/pairs_480/train/rough`
(raw pencil scans) -- but the model was *trained* on `lineart_anime`-
preprocessed conditioning, so this eval was testing the wrong input
distribution entirely. Confirmed `infer_controlnet.py` has no built-in
preprocessing step (checked its args directly). Fixed by preprocessing the
10-tile diagnostic set
(`dataset/pairs_480/diag_controlnet_same_coordinate_10.txt`) with the new
preprocessing tool and re-running inference
(`results/controlnet_lora_realpairs_20260824_eval_preprocessed/`) for a
correctly-matched comparison against the original mismatched-conditioning
run (`results/controlnet_lora_realpairs_20260824_eval/`).

**Result: quantitatively confirmed the preprocessing fix helps, but the
underlying quality is still poor.**

`evaluate_fixed_outputs.py` (bsds_f1, this project's best-validated GT
metric per Workstream C):

| variant | bsds_F1 | ink_ratio |
|---|---:|---:|
| correct (`lineart_anime`) conditioning | 0.140 | 7.5x GT |
| raw (mismatched) conditioning | 0.132 | 20.3x GT |

`condition_roundtrip_fidelity.py` (conditioning-adherence metric):

| variant | roundtrip_ssim | roundtrip_bsds_f1 |
|---|---:|---:|
| correct conditioning | 0.408 | 0.074 |
| raw conditioning | 0.260 | 0.063 |

Correct conditioning wins on every axis in both metric families -- the
preprocessing bug was real and worth fixing -- but both variants' absolute
scores are low. **Visual review of the 10-tile comparison montage**
(rough / `lineart_anime` condition / correct-cond output / mismatched-cond
output / GT) confirmed the numbers: most tiles show dense, content-
independent crosshatch/parallel-hatch hallucination overwhelming the
actual input structure -- the same qualitative failure mode this branch has
hit repeatedly (rough-domain LoRA's original parallel-hatch collapse
2026-08-05/06, the line-domain psd_line-expansion collapse 2026-08-21).
One tile (a recurring dagger/sword shape used as a diagnostic landmark
since 2026-08-08) is a partial exception -- shape recognizably preserved
under the correct-conditioning variant, though still textured/noisy.

**Not yet root-caused.** Candidate explanations not yet distinguished:
possible content-composition skew in the newly-added `clip_pairs` tiles
(real manga does contain genuine dense-hatching panels, unlike the
synthetic pseudo-rough case); LoRA rank16 may be too constrained to learn
nuanced per-content behavior on top of an already-opinionated public
checkpoint at this data scale; 10 epochs over 8467 tiles may still be
under-trained for a *fine-tune* (as opposed to the from-scratch case where
this same step count was already shown insufficient back in Direction 4's
original longrun test) in a different way. No conclusion drawn -- this
needs the same kind of isolation-experiment treatment already applied
successfully to the domain-LoRA collapses, not a single-shot verdict.

Comparison montage (rough / `lineart_anime` condition / correct-cond output
/ mismatched-cond output / GT, all 10 diagnostic tiles):
`results/controlnet_lora_realpairs_20260824_compare_montage.png`.

### Next Actions

1. Not decided: whether to pursue root-causing this (composition-skew
   check on `clip_pairs` tiles via `measure_lineart_profile.py`, same
   method as the psd_line investigation; a stricter/lower LoRA rank or
   scale sweep; more/fewer epochs) or pause ControlNet work again --
   open, needs user discussion given the multi-hour-per-experiment cost.
2. Checkpoint and both eval variants kept on disk as reference/evidence,
   per this project's standing policy, regardless of what's decided next.
3. `infer_controlnet.py` still has no built-in `lineart_anime`
   preprocessing option -- any future eval run must preprocess the input
   externally first (via the new `preprocess_lineart_anime_condition.py`)
   or risk repeating today's mismatched-conditioning bug.

## 2026-08-26: Diagnostic Sample List Found Half-Broken -- Corrected Metrics, Verdict Unchanged

User spotted a real problem by eye: in the compare montage's "lineart_anime
cond" column, every row except the dagger tile looked essentially solid
black -- i.e. the conditioning image the model actually saw looked empty
for most tiles. Investigated rather than dismissed.

**Confirmed and root-caused, not a preprocessing bug**: measured raw pixel
stats on the *original* (pre-`lineart_anime`) rough tiles referenced by
`dataset/pairs_480/diag_controlnet_same_coordinate_10.txt`. Sharp bimodal
split -- 5 of 10 tiles (`housei_002_25_11`/`housei_011_13_03`/
`housei_018_18_15`/`housei_002_23_15`/`lineart_003_011`) are themselves
essentially blank source scans (std 4.5-7.7, `dark_ratio(<200)` 0.03-0.5%
-- e.g. `housei_011_13_03.jpg`: mean=254.1, std=4.5, virtually no pixel
below 200). The other 5 (including the dagger tile) have genuine content
(std 20-48, dark_ratio 4.8-9.3%). This diagnostic list predates this
session (created in the original Direction 4 exploration, 2026-08-08 era)
and evidently references stale/broken files for half its `housei`-named
entries -- not a new bug introduced today, but never previously noticed
because this list hadn't been used with an ink-content sanity check
before.

**Checked whether this also affects the training pool** (500-tile random
sample from the full 8467-tile `rough_lineart_anime_20260824` conditioning
set): only 2.0% near-blank overall, and the `houseikoma`-prefixed tiles
specifically (the current koma-pipeline's own housei tiles, a *different*
naming convention from the broken diagnostic list's `housei_NNN_NN_NN`
files) showed 0% blank in-sample, median ink ratio 0.019 -- normal. **The
training data itself is not implicated** -- this was specifically an
eval-diagnostic-list problem, not a training-data problem.

**Recomputed both metric families on just the 5 valid tiles**
(`dataset/pairs_480/diag_controlnet_same_coordinate_10_valid5.txt`, new,
saved as a lasting fix so future eval runs on this diagnostic set don't
repeat the mistake):

| metric | 10 tiles (5 broken) | 5 valid tiles only |
|---|---:|---:|
| `bsds_F1` | 0.140 | 0.143 (~unchanged) |
| `ink_ratio` | 7.5x GT | 4.3x GT (improved) |
| `roundtrip_ssim` | 0.408 | 0.418 (~unchanged) |
| `roundtrip_bsds_f1` | 0.074 | 0.140 (~doubled) |

**Verdict**: the broken tiles were a real confound -- they measurably
inflated over-inking and roundtrip-fidelity failure (a blank conditioning
input gives the model nothing to follow, so of course it invents content)
-- but `bsds_F1` against GT barely moved, and visual review of the 5 valid
tiles still shows the same crosshatch-hallucination pattern on all but
the dagger tile. **The core "not yet good enough" conclusion from the
prior entry stands**, now on cleaner evidence rather than confounded by
half-broken test inputs.

### Next Actions

1. Use `dataset/pairs_480/diag_controlnet_same_coordinate_10_valid5.txt`
   (not the original 10-tile list) for any future eval on this checkpoint
   or successors, until/unless the original list's 5 broken entries are
   individually fixed or replaced.
2. Original Next Actions (root-cause investigation vs. pause) from the
   prior entry are unchanged by this correction -- still open, still
   needs user discussion.

## 2026-08-26 (later): Project Housekeeping -- `results/` Cleanup Policy Reversed, 166→43 Items

User paused the ControlNet root-cause work (explicitly not GPU-cost-averse
-- "electricity is the only cost on a local machine" -- this was a
deliberate project-management priority call, not a budget constraint) to
address two standing friction points directly.

### `results/` cleanup

User: `results/` had grown large enough that finding a needed file cost
real time (~1 min per search by their estimate), and the overwhelming
majority of it was "garbage" from settled/rejected experiments kept under
the old "might have residual value" theory. **Explicitly reversed that
theory** -- see [[feedback_session_close_cleanup_habit]] (memory,
rewritten this session) and `doc/RESULTS.md` (rewritten this session,
full policy statement there).

Executed in tiers: (1) clear-cut settled-failure/debug artifacts deleted
without discussion (old ControlNet-from-scratch Direction 4 montages,
rejected psd_line pool-expansion attempts, GAN-era Direction 5/6/8/9
survey montages+per-sample dirs, old badrough/halo/router/linefield-era
metric CSVs, a `koma_memtest` scratch dir, stale `region_search_loop`);
(2) per-source raw-dataset folders
(`ako5ver2`/`fitness`/`hamlabi`/`gakuen`/`housei`/`fighting`) trimmed to
just their manifest/tile CSVs (data provenance), all QC/overlay images
deleted -- the review they supported concluded long ago;
(3) `results/lessons/` trimmed to CSVs only, montage images dropped (the
finding is fully written up in `doc/architecture_decisions.md`).
Explicitly kept: the domain-LoRA adopted-config reference samples
(`domain_lora_{line,rough}_sd15base_sksv2_20260807_scale14/`), the
`bsds_f1` calibration basis (`eval_metric_calibration_20260809/`),
`domain_lora_line_captiontags_20260807/` (a real functional dependency --
an adopted training script's `--caption-csv` points at it), all
`lineart_profile_*.csv` (per the project's own long-standing "keep
unconditionally" rule for these), and everything from today's active
`clip_pairs`/ControlNet-real-pairs work.

**Result: 166 items / 1.6GB -> 43 items / 79MB.**

Also deleted `config/results_manifest.json` (a hand-maintained lightweight
index that had drifted out of sync with `results/`'s actual contents for
some time -- keeping a second manifest in sync was itself part of the
clutter problem) rather than regenerating it. No tooling depended on it,
only doc references (now updated). `doc/RESULTS.md` fully rewritten with
the new policy, the exact category-by-category deletion list, and an
explicit "delete at verdict time, not later" rule going forward.

### Session/context hygiene

User separately flagged: the accumulated volume of logs/memory/rule
markdown across sessions in this project is degrading response quality,
and wants a reset toward leaner, more purpose-scoped sessions going
forward (not one long-running thread covering many unrelated large
tasks). Saved as [[feedback_session_context_hygiene]] (new memory).
Concretely connects to `doc/work_log.md` itself, which had reached 4595
lines (approaching the existing 5000-line maintenance-policy threshold in
`doc/documentation_maintenance_policy.md`) -- largely from this single
session's own very large `clip_pairs` pipeline + ControlNet fine-tune
work. Not yet compacted as of this entry; see Next Actions.

### `lineart-controlnet-realpairs` Track Spun Off (2026-08-26)

Per the session-hygiene concern above, the user introduced a new
"track" pattern: a sibling project folder
(`/home/sh1/deepl/lineart-controlnet-realpairs/`, plain directory, not a
git worktree) for continuing the ControlNet real-pairs hallucination
investigation, briefed via a single minimal
`inbox/initial_notice.md` (mirrors the extraction-tool inbox/outbox
convention, applied locally). This `lineart` checkout is now the shared
foundation the track reads code/data from (`../lineart/venv`,
`../lineart/scripts`, `../lineart/dataset`); the track writes its own new
experiment output into its own `results/`/`logs/`/`checkpoints/`, not
back into this folder's `results/` (just cleaned up, see above -- not to
be re-bloated). See [[feedback_track_folder_pattern]] memory for the
general pattern.

**The ControlNet hallucination root-cause work itself now continues in
that track, not here.** This `work_log.md` entry is the handoff marker;
do not duplicate further ControlNet-realpairs experiment logs in this
file going forward -- check the track folder's own docs/logs instead.

### Next Actions

1. `doc/work_log.md` is a natural candidate for a compaction pass now
   (matching `doc/documentation_maintenance_policy.md`'s existing
   procedure) given both the size threshold and the user's stated
   session-hygiene concern -- not yet done, open.
2. Going forward: apply the revised `results/` policy at the point each
   experiment's verdict is reached (per `doc/RESULTS.md`'s new "Going
   Forward" section) rather than deferring to periodic sweeps.
3. ControlNet-hallucination root-cause work continues in
   `lineart-controlnet-realpairs/` (see above), not in this checkout --
   the Next Actions listed in the 2026-08-24/25 and earlier 2026-08-26
   entries are superseded by that track's own `inbox/initial_notice.md`.

## 2026-08-22 (later): SSH-Push Delivery Failure -- Wrong Destination Directory

First SSH-push to the extraction tool side (`note_panel_level_qc_20260822.md`,
sent to a newly-created `C:\Users\sh1\inbox_from_lineart\`) went
undelivered -- not a network/auth problem (the file really did land, `dir`
confirmed it), but the receiving Claude Code session never noticed it
because that path is outside its own project working directory. Checked
the Windows filesystem directly over SSH and found the tool's real project
root: `C:\Users\sh1\code\extract_line_and_sketch\`, which already has its
own `inbox\`/`outbox\` folders mirroring this project's own convention.
**Corrected destination for all future pushes**:
`C:\Users\sh1\code\extract_line_and_sketch\inbox\`, recorded in
[[feedback_extraction_tool_ssh_push]] memory. User manually surfaced the
misplaced file to the tool session this time; no resend needed. The
orphaned `C:\Users\sh1\inbox_from_lineart\` folder is still there,
unused going forward -- left as-is, cleanup optional.

## 2026-08-21 (later still): QC Version Confirmed Correct -- Our "Missing QC" Read Was A Same-Folder File Mixup

Sent the outbox reply requesting the QC-enhanced version; the tool
maintainer replied within the hour (`inbox/reply_qc_clarification_20260821.md`)
with a direct correction, **and it was our mistake, not theirs**: the
`qc`/`source_path`/`content_fingerprint` fields we checked for were absent
from `koma_manifest.json` (the per-page koma-only file) because we
inspected the wrong file in the folder -- the real pair manifest,
`manifest.json`, had them all along. Verified directly: SHA-256 of our
already-downloaded `dataset_clip_pairs_v2.zip` matches their quoted
checksum exactly
(`ef087dff29d0fa86f75bc2b4b18ecde8e788a60f11a62385d521b4cd250885bc`), and
`unzip -p ... manifest.json | python3 -c '...'` on the same
`041_2021_akogoods` example they used confirms the full `qc` block is
present. **No re-transfer was needed** -- `dataset_clip_pairs_v2.zip` is
already the final version. Corrected `doc/preprocess/raw_dataset_storage_policy.md`'s
entry accordingly (removed the incorrect "partial/intermediate version"
characterization).

Also resolved the numeric discrepancy from the prior entry: confirmed
**1272** (not 1151) is the correct `qc=ok` + unique + has-koma pair count
-- the sender's prior reply had left a stale pre-fix number in the prose
paragraph even after updating the table; both are now corrected on their
end too. Also traced their own "サーバ上のdataset_clip_pairs.zipは1,697組版
のまま" line (the thing that made us doubt the zip's completeness in the
first place) to a literal copy-paste artifact: that sentence was written
before finalizing the zip, then accidentally shipped bundled inside the
very zip it was describing as outdated.

**New operationally-relevant fact volunteered in this reply**: the
delivered zip is a **mixed-vintage extraction** -- only 289 of 2644 total
pages used the latest (bug-fixed) extraction code; the remaining 1451
pages are still on the older code. A full 2644-page re-extraction with the
fixed code is planned for their next free machine window (~Monday from
2026-08-21); even some already-`ok`-flagged pages may improve further
(their own example: `066_2024_housei page0007`, coverage 0.208 -> 0.561
under the newer code despite already being `ok`). Explicitly confirmed:
**safe to start this project's own panel-pipeline work on the current
1272-pair pool now**, not blocked on Monday -- `content_fingerprint` will
let us detect which specific pairs change content in the next delivery.
They also asked for any panel-alignment-quality or `pair_quality`-accuracy
feedback from our own pipeline run, to fold into Monday's re-extraction.

### Next Actions

1. `dataset_clip_pairs_v2.zip` is confirmed final and ready to use --
   extract `clip_pairs/clip_pairs_qc.csv` for filtering
   (`pair_quality=ok` + `is_primary` + koma present -> ~1272 starting
   pairs), then begin adapting `match_koma_panels.py` to consume
   `clip_pairs`-style koma layers, per the prior entry's plan. Not started
   yet.
2. After a first pipeline pass, send back any concrete panel-detection or
   `pair_quality`-accuracy findings (false positives/negatives) so they
   can fold it into Monday's full re-extraction -- open, depends on step 1
   actually running first.
3. Once Monday's re-extraction lands, diff `content_fingerprint` values
   against this version to identify which of the 1272 pairs changed and
   need reprocessing -- not yet relevant, no new delivery yet.
