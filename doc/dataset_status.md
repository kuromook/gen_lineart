# Dataset Status

Updated: 2026-07-25 JST

This file summarizes current usable datasets, review targets, held-out rows,
and recommended next dataset actions. Chronological extraction details remain
in `doc/work_log.md`; dataset-specific quirks live in
`doc/raw_dataset_extraction_knowledge.md`.

## Current Priority

Current dataset work is focused on region-matched raw manuscript expansion.

Primary active source:

- `ako5ver2`

Current immediate objective:

- continue reviewing the 281 kept ako5ver2 variable-region rows
- produce a small, reviewed training manifest before the next controlled model
  run

## ako5ver2

Raw archive:

- `dataset/raw_zips/dataset_ako5ver2.zip`
- zip root: `dataset_ako5`

Current review target:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/`

Current QC to inspect:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/valid_mask_qc.png`

Current filtered manifest:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv`

Review status:

| category | rows | file |
|---|---:|---|
| source variable-region rows | 291 | `manifest.csv` |
| user-excluded mismatches | 7 | `removed_user_mismatch.csv` |
| held umbrella/mask-insufficient layer differences | 3 | `held_user_mask_insufficient_umbrella.csv` |
| all flagged review rows | 10 | `flagged_user_review_20260725.csv` |
| kept after current review flags | 281 | `manifest_user_review_keep281.csv` |

Excluded mismatch review indices:

- 42
- 58
- 59
- 81
- 90
- 98
- 112

Held umbrella / layer-difference review indices:

- 69
- 79
- 86

Interpretation:

- ako5ver2 has usable region-matched variable-aspect candidates.
- Some rows contain production-layer differences, such as rough-only umbrella
  content absent from line art.
- Current valid masks are not enough to safely train those umbrella rows.

Recommended next actions:

1. Continue visual review from `valid_mask_qc.png`.
2. Add new mismatch / hold / accept decisions to the review CSV set.
3. Consider filtering by high mask validity and low unsupported line ratio.
4. Train only after the reviewed manifest is stable.

Suggested experiment order:

1. `ako5ver2_varregion_keep281_masked_unet`
   - source: `manifest_user_review_keep281.csv`
   - loader: region manifest
   - fit mode: `square_pad`
   - image size: 480 first
   - mask: `valid_mask_path`
   - purpose: main current ako5 signal
   - status: from-scratch e20 completed, technically valid but visually too
     density-map / black-fill oriented
   - current artifacts:
     - `checkpoints/ako5ver2_varregion_keep281_masked_unet480_e20/best.pth`
     - `results/ako5ver2_varregion_keep281_masked_unet480_e20_compare.png`
   - warmstart artifacts:
     - `checkpoints/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10/best.pth`
     - `results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_compare.png`
     - `results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_threshold52_compare.png`
   - interpretation: warmstart improves loss but still looks too soft /
     density-map oriented; thresholding produces black islands more than clean
     line art
2. `ako5ver2_strict88_480_unet`
   - source: `valid_train_ako5ver2_region_20260725_strict.txt`
   - loader: pair list
   - line-dir override:
     `dataset/pairs_480/train/line_ako5ver2_region_20260725_strict`
   - purpose: fixed-tile compatibility/control signal
   - current artifacts:
     - `checkpoints/ako5ver2_strict88_480_warm_clean_bce_e10/best.pth`
     - `results/ako5ver2_strict88_480_warm_clean_bce_e10_threshold52_compare.png`
     - `results/ako5ver2_strict88_480_warm_clean_bce_e10_threshold_sweep_compare.png`
   - interpretation: much more line-like than keep281 masked U-Net, but still
     thickens lines and grows black-fill regions; keep separate for now
3. `ako5ver2_keep281_tiles_nobeta_top300_480`
   - source: reviewed keep281 masked manifest, cropped as true 480x480 windows
     from 768x768 masked region images
   - list:
     `dataset/pairs_480/valid_train_ako5ver2_keep281_tiles_nobeta_top300_20260725.txt`
   - line-dir override:
     `dataset/pairs_480/train/line_ako5ver2_keep281_tiles_nobeta_top300_20260725`
   - QC:
     - `results/ako5ver2_keep281_tiles_480_nobeta_top300_qc.png`
     - `results/ako5ver2_keep281_tiles_480_nobeta_top300_qc_tail.png`
   - filters:
     - line ink <= 0.08
     - largest black component ratio <= 0.025
     - thick ink ratio <= 0.015
   - artifacts:
     - `checkpoints/ako5ver2_keep281_tiles_nobeta_top300_480_warm_clean_bce_e10/best.pth`
     - `results/ako5ver2_keep281_tiles_nobeta_top300_480_warm_clean_bce_e10_threshold_sweep_compare.png`
   - interpretation: black-fill exclusion improves loss and reduces blob mass,
     but output remains sparse ink/edge fragments rather than clean line art
4. `ako5ver2_native_tiles_strokescale` (current preferred direction)
   - source: keep281 re-materialized near native source resolution instead of
     the 768 px long-side normalization
   - materializer: `tools/pair_extraction/materialize_region_manifest_native.py`
   - materialized target: `dataset/regions_ako5ver2_native_20260725/`
     (165 of 281 regions kept; 116 dropped for being below 480 px native, which
     were only viable through upscaling)
   - masks: `dataset/regions_ako5ver2_native_20260725_masked_line_conservative/`
     (`--support-px 20`, native-scale recalibrated from the 768 default of 5)
   - tile CSV: `results/ako5ver2_native_tiles_480_strict_cut25.csv`
   - QC: `results/ako5ver2_native_tiles_480_strict_cut25_qc*.png`
   - filters: same stroke-scale gates as the 768-based strict subset, plus
     `--min-tile-score 2.5` to drop a tail of tiles that passed every individual
     gate but showed unrelated rough/line content on full-resolution review
   - counts: 422 tiles from 112 regions (498 tiles / 119 regions before the
     score cutoff; 90 tiles / 45 regions in the earlier 768-normalized subset)
   - status: reviewed, saved, and integrity-audited (0 findings)
   - list: `dataset/pairs_480/valid_train_ako5ver2_native_strict_cut25_20260725.txt`
   - line dir: `dataset/pairs_480/train/line_ako5ver2_native_strict_cut25_20260725`
   - trained: `ako5ver2_native_strict_cut25_480_warm_clean_bce_e10`
     (checkpoint: `checkpoints/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10/best.pth`)
   - result: loss decreased monotonically (0.3048 to 0.2617), but visual output
     is the same soft/density-map failure mode seen across every earlier
     keep281-derived recipe; see `doc/model_results_summary.md` and the
     2026-07-25 native strict cut25 training entry in `doc/work_log.md`
   - conclusion: the native re-materialization and stroke-scale filter design
     is considered data-side validated; the remaining gap is model-side (loss
     design / training length), not a data quality problem
   - interpretation: same filter design as the 768-normalized strict subset,
     applied to native-resolution material; see
     `doc/raw_dataset_extraction_knowledge.md` for the native re-materialization
     rationale and `doc/region_dataset_extraction_policy.md` for the tile-score
     tail-quality note
5. `ako5ver2_keep281_tiles_strokescale` (768-normalized, superseded by native)
   - source: reviewed keep281 masked manifest, filtered to the usable source
     scale band instead of by edge correspondence alone
   - CSV: `results/ako5ver2_keep281_tiles_480_strokescale.csv`
   - QC:
     - `results/ako5ver2_keep281_tiles_480_strokescale_review.png`
     - `results/ako5ver2_keep281_tiles_480_strokescale_qc.png`
     - `results/ako5ver2_keep281_tiles_480_strokescale_qc_tail.png`
   - filters:
     - src_per_out band 1.2 to 4.0
     - line ink 0.012 to 0.08
     - largest black component <= 0.025, thick ink <= 0.015
     - soft ink ratio <= 0.45, line width p50 <= 6.0
     - strict 3 px edge recall >= 0.40, strict 3 px precision >= 0.15
     - long straight line ratio <= 0.25
   - counts: 90 tiles from 45 regions, 117 of 281 regions passed the scale gate
   - status: dry-run only, awaiting user QC review before `--save`
   - interpretation: visually the cleanest keep281-derived tile set so far,
     character/drapery scale with real correspondence, but small
5. Only after these are inspected, decide whether to:
   - keep them separate
   - train a staged/curriculum run
   - materialize one source into the other's format for controlled mixing

## ako5ver2 Fixed-Tile Baseline

Fixed 480px strict region-guided extraction:

- `dataset/pairs_480/valid_train_ako5ver2_region_20260725_strict.txt`

Rows:

- 88

Line directory override:

- `ako5r=dataset/pairs_480/train/line_ako5ver2_region_20260725_strict`

Integrity audit:

- `results/pair_dataset_integrity_summary_ako5ver2_region_20260725_strict.csv`
- `results/pair_dataset_integrity_findings_ako5ver2_region_20260725_strict.csv`

Audit result:

- findings: 0

Status:

- keep as a fixed-tile baseline only
- do not treat as the preferred ako5 workflow, because it does not preserve
  variable-aspect parent/child regions

Recommended use:

- use as a small compatibility/control dataset, not as the main ako5 source
- useful for checking whether the old 480px pair-list training path can learn
  anything from the cleaner ako5ver2 extraction
- useful as a later supplemental source only after the variable-region manifest
  behavior is understood
- do not mix into the first variable-region masked experiment, because the
  supervision/mask semantics differ

Suggested first role:

- short fixed-tile baseline run with explicit line-dir override
- compare against the variable-region masked run and hamlabi filtered398

## fighting

Formerly uploaded and referred to as `lineart`; renamed because that name was
too generic and collided with an unrelated legacy `lineart_`-prefixed source
category already used by `lineart/dataset_gate.py`'s `DEFAULT_LABELS` (168
pre-existing `lineart_NNN_NNN.jpg` files in `dataset/pairs_480/train/rough`
predate this session and are unrelated).

Raw archive:

- `dataset/raw_zips/dataset_fighting.zip`
- layout: `rough/rough_<page>-<index>.jpg` paired with `2/line_<page>-<index>.jpg`
- 8 pages, 24 tiles/page, 192 pairs total, already fixed native 480x480 tiles

Unlike ako5ver2/fitness, no region matching or alignment search was needed:
pairs are already 1:1 by filename suffix and visually well aligned throughout
(top and tail full-resolution review both showed correct rough/line content
correspondence, e.g. face, hand, cheek, thighs, cape/hair).

Filter tool:

- `tools/pair_extraction/filter_fighting_pairs.py`
- reuses the same strict stroke-scale filter (`analyze_tile` from
  `tile_region_manifest_480.py`) and native-scale mask calibration
  (`build_mask` from `build_region_valid_masks.py`, `support_px=20`)
  validated on ako5ver2 native tiles

Result:

- source pairs: 192
- accepted: 40 (21%)
- rejected: 152
- list: `dataset/pairs_480/valid_train_fighting_native_strict_20260725.txt`
- line dir: `dataset/pairs_480/train/line_fighting_native_strict_20260725`
- mask dir: `dataset/regions_fighting_native_20260725_masks`
- QC: `results/fighting_native_tiles_480_strict_qc.png`,
  `results/fighting_native_tiles_480_strict_qc_tail.png`
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_fighting_native_strict.csv`)

Status:

- reviewed, saved, integrity-audited; ready as a training source
- not yet trained on; do not mix into another source's list without a
  deliberate experiment design

## fitness

Formerly uploaded and referred to as `kurip`; renamed because that name was
actually a person's username. All leak-era `kurip`-named data, checkpoints,
comparison scripts, and experiment runners predating this session were deleted
outright rather than renamed (per explicit user decision: leak-era material is
not needed and not worth the larger rename effort). Only this session's
reviewed native-strict extraction was kept and renamed.

Raw archive:

- `dataset/raw_zips/dataset_fitness_v4.zip`
- 38 pages, 4961x7016 native resolution (same scan setup as ako5ver2)

Route: `tools/pair_extraction/diagnose_pair_alignment.py` recommended
`needs_global_or_local_alignment` (lighter than ako5ver2's
`needs_region_matching`). Used `tools/pair_extraction/match_kurip_regions.py`
(name predates use beyond its original source; fully generic via
`--zip`/`--zip-root`) for line-anchored, rough-side local offset search on a
fixed 480x480 grid, then
`tools/pair_extraction/filter_matched_region_tiles.py` (renamed from
`filter_kurip_matched_tiles.py`) for the same strict stroke-scale filter
validated on ako5ver2. Candidate tiles are already native-resolution fixed
480x480 crops, so `src_per_out` is always 1.0 by construction; none of the
ako5ver2 source-scale normalization work applies here.

Full-resolution review across the score range (top and tail) found no
semantic mismatches; a first pass at the default `--min-tile-score 2.5`
(before review) needed raising to `2.9` after tail review, per the same
"tile score is not a content-match guarantee" caution from
`doc/region_dataset_extraction_policy.md`.

Result:

- raw region matches: 1,509
- accepted at `--min-tile-score 2.9`: 271 tiles from 37 regions
- list: `dataset/pairs_480/valid_train_fitness_native_strict_20260726.txt`
- line dir: `dataset/pairs_480/train/line_fitness_native_strict_20260726`
- mask dir: `dataset/regions_fitness_native_20260726_masks`
- QC: `results/fitness_native_tiles_480_strict_cut29_qc.png`,
  `_qc_tail.png`
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_fitness_native_strict.csv`)

Operational note: this environment was observed to silently kill long-running
background extraction jobs somewhere around 10-13 minutes with no traceback,
regardless of tool-level timeout settings (affected both a detached
`nohup`+`disown` process and a properly harness-tracked background task; two
overnight autonomous agents working on this and the `housei` source were also
lost this way, their transcripts unrecoverable). Worked around by adding
`--offset`/`--limit`/`--append` to `filter_matched_region_tiles.py` and running
the full pass in ~250-row chunks, each well under the failure window. See
`doc/raw_dataset_extraction_knowledge.md`.

Status:

- reviewed, saved, integrity-audited; ready as a training source
- not yet trained on; do not mix into another source's list without a
  deliberate experiment design
- `tools/pair_extraction/match_kurip_regions.py` and 4 other pre-existing,
  still-active infra scripts (`extract_kurip_matched_tiles.py`,
  `prepare_kurip_tiles.py`, `refine_kurip_vlm_tiles.py`,
  `vlm_review_kurip_matches.py`) still carry the old name; renaming them is a
  separate, larger decision (cross-referenced in docs, and
  `prepare_kurip_tiles.py` is shared with hamlabi) held open pending user
  input

## housei

Raw archive:

- `dataset/raw_zips/dataset_housei.zip`
- flat zip layout (no subfolder root; use `--zip-root ""`)
- manifest schema differs from ako5/fitness: `{"page": "page0010", "housei":
  "housei_001", "sketch": ..., "line": ..., ...}` (no `"file"` key)
- 18 pages, 4961x7016 native resolution (same scan setup as ako5ver2/fitness)

`tools/pair_extraction/diagnose_pair_alignment.py` needed a small fix to
support the empty `zip_root` (its `load_manifest`/`load_zip_pair` had no
bare-filename fallback; added the same `read_zip_member` fallback pattern
already used elsewhere). Route recommendation:
`needs_global_or_local_alignment`, same as fitness, so the same
`match_kurip_regions.py` + `filter_matched_region_tiles.py` route was used.

Yield is low relative to page count at the ako5ver2-default gates: a
full-permissive measurement pass across all 602 raw matches confirmed only
about 51 tiles clear the strict edge-recall/precision/soft-ink/width gates
regardless of `--min-tile-score` (1.5 through 2.9 all gave ~51), so the
bottleneck is those gates, not the score cutoff. Diagnosed further with the
new `tools/pair_extraction/diagnose_gate_funnel.py` (evaluates every gate
independently per tile instead of stopping at the first failure): across all
602 matches, `soft_ink_ratio` alone accounts for 168 of 214 rejections at the
point it is checked (78%), far above every other gate (next-worst individual
fail rate was `ink_range` at 31%). This is a rough-style property of this
source (more pencil/gray texture than ako5ver2 or fitness), not a matching or
alignment problem.

Relaxed `--max-soft-ink-ratio` from the ako5ver2-derived default `0.40` to
`0.50` after visually confirming (23-tile spot check, later the full set) that
the added tiles hold the same quality: correct content correspondence
throughout, no semantic mismatches, no visible gray/density-map degradation.
Adopted as this source's setting; other gates stayed at ako5ver2 defaults.

Result:

- raw region matches: 602
- accepted: 65 tiles from 14 regions (`--max-soft-ink-ratio 0.50`; was 25 tiles
  from 10 regions at the ako5ver2-default `0.40`)
- list: `dataset/pairs_480/valid_train_housei_native_strict_20260726.txt`
- line dir: `dataset/pairs_480/train/line_housei_native_strict_20260726`
- mask dir: `dataset/regions_housei_native_20260726_masks`
- QC: `results/housei_native_tiles_480_strict_qc.png`, `_qc_tail.png`
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_housei_native_strict.csv`)

Status:

- reviewed, saved, integrity-audited; ready as a training source
- not yet trained on individually; do not mix into another source's list
  without a deliberate experiment design
- further relaxation of `soft_ink_ratio` beyond 0.50 was judged to have more
  room (yield keeps climbing: ~97 at 0.60, ~134 at 0.70) but was intentionally
  not pursued further this session to move on to training/inference results;
  revisit if more housei volume is wanted
- superseded for the first combined training pass by a chamfer-based
  alignment filter; see `## Combined Training Runs (2026-07-26)` below

## Combined Training Runs (2026-07-26)

First attempts at training on more than one reviewed native-strict source
together, to get an "overall" read on model quality across the broadened
pool. Both use the same recipe as prior single-source runs (`unet`, warmstart
from `checkpoints/shape1_clean_split_bce/best.pth`, strict resume, 480px, 10
epochs, `lr 1e-5`, `pos-weight 3.0`), for direct comparability.

### combined_20260726 (798 tiles, no alignment filter)

All four reviewed sources concatenated as-is: ako5ver2 native strict cut25
(422) + fitness (271) + housei (65) + fighting (40).

- list: `dataset/pairs_480/valid_train_combined_20260726.txt`
- line dir: `dataset/pairs_480/train/line_combined_20260726` (symlinks into
  each source's own line dir; rough dir is already shared)
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_combined_20260726.csv`)
- runner: `experiments/run_combined_20260726_warm_clean_bce_e10.sh`
- final epoch: `Epoch 010/10: G=0.2459` (from 0.2793)
- per-source QC: `results/combined_20260726_perSource_{ako5nat,fitnessm,houseim,fighting}.png`

Visual result: output crispness varies sharply and consistently by source
(fighting best, ako5ver2-native/housei worst, fitness in between), tracking
each source's chamfer distance almost exactly. See
`doc/region_dataset_extraction_policy.md` ("Alignment Gate vs Style Gate")
and `doc/raw_dataset_extraction_knowledge.md` ("Residual Misalignment") for
the full investigation this triggered.

### alignfilt12_20260726 (217 tiles, chamfer<=12 re-filter)

Same four sources, each re-filtered from its already-saved/reviewed tiles to
keep only `chamfer <= 12`: ako5nat 53/422, fitness 108/271, housei 31/65,
fighting 25/40.

- list: `dataset/pairs_480/valid_train_alignfilt12_20260726.txt`
- line dir: `dataset/pairs_480/train/line_alignfilt12_20260726` (symlinks)
- per-source alignment-tightened lists:
  `dataset/pairs_480/valid_train_{ako5nat,fitness,housei,fighting}_alignfilt12_20260726.txt`
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_alignfilt12_20260726.csv`)
- runner: `experiments/run_alignfilt12_20260726_warm_clean_bce_e10.sh`
- final epoch: `Epoch 010/10: G=0.2016` (from 0.2483; lower absolute loss than
  the 798-tile run, expected since the training set itself is smaller/cleaner)
- comparison montage: `results/compare_798_vs_alignfilt12_20260726.png`

Visual result: no visible output-quality difference from the 798-tile run on
matched samples, despite chamfer correlating with quality across sources in
both runs. This is an open question, not a negative result on the alignment
hypothesis itself — see "Determining The Alignment Threshold" in
`doc/region_dataset_extraction_policy.md` for the live open questions
(threshold may not be tight enough yet; 10-epoch warmstart may not be
sensitive enough to show the difference; chamfer alone may not be a
sufficient alignment metric).

Status: both are exploratory/diagnostic runs, not adoption candidates. Do not
promote either checkpoint. The investigation this triggered (scale/deformation
as the likely root cause of residual misalignment, panel-boundary-first
segmentation as the planned fix) is the more important outcome; see
`doc/CURRENT.md` and `doc/raw_dataset_extraction_knowledge.md`.

## hamlabi

Current useful reviewed dataset base:

- `dataset/regions_hamlabi_loop_auto_review_postalign12_masked_line_conservative_filtered398/manifest.csv`

Rows:

- 398

Notes:

- Built through variable-aspect search, post-align, valid-mask workflow, and
  user mismatch removal.
- User visual review removed ranks 116 and 118 from the previous 400-row set.
- Plain 480px U-Net e40 training worked technically but did not produce a
  finished line-art expert.

Current result artifacts:

- checkpoint:
  - `checkpoints/hamlabi_filtered398_unet480_e40/best.pth`
  - `checkpoints/hamlabi_filtered398_unet480_e40/epoch040.pth`
- montage:
  - `results/hamlabi_filtered398_unet480_epoch040_compare.png`
- note:
  - per-sample output images under `results/` were removed during results image
    cleanup; keep the montage and metrics/checkpoints as the durable reference

Interpretation:

- hamlabi extraction workflow is useful.
- hamlabi is complex: roughs are dense, sketchy, and often black-fill-heavy.
- Model quality should not be used as the only judgment of extraction quality.

Recommended next action:

- use hamlabi as a working reference for the variable-region extraction
  pipeline while comparing other raw datasets such as ako5ver2.

## Archived / Historical Dataset Notes

Old leak-era results and assumptions should remain historical.

Do not read archive directories unless explicitly auditing:

- `doc/archive/`
- `results/archive/`

## Promotion Checklist

Before any reviewed raw dataset becomes a training source:

1. QC reviewed.
2. Mismatches excluded.
3. Layer/prop differences either held out, masked, or explicitly tagged.
4. Manifest path is stable.
5. Integrity audit passes when using fixed pair-list training.
6. Training command records manifest, mask key, image size, fit mode, and any
   line-dir overrides.
