# Raw Dataset Extraction Knowledge

Updated: 2026-07-25

Use this file for reusable knowledge discovered while extracting pairs from
specific raw manuscript datasets.

This is not a chronological work log. Record observations here when they affect
future extraction, review, masking, filtering, or training decisions for a raw
source dataset.

General extraction procedure and gates remain in `doc/EXTRACTION_RULES.md`.

## What Belongs Here

Record dataset-specific facts such as:

- known rough/line layer differences
- recurring missing objects or extra objects
- page or manifest quirks
- mask failure modes
- black-fill, screentone, effect-line, or 3D/reference-layer issues
- review decisions that should influence future filtering
- source-specific thresholds or parameters that worked or failed

Do not use this file for:

- full command transcripts
- ordinary experiment metrics
- one-off training conclusions
- historical leak-era assumptions

Keep those in `doc/work_log.md` or archive them when obsolete.

## Review Decision Vocabulary

Use consistent labels when possible:

- `exclude_mismatch`: rough and line do not represent the same usable content.
- `hold_mask_insufficient`: content may correspond, but current mask/target is
  not safe enough for training.
- `hold_layer_difference`: rough contains content that line art omits, or line
  art contains content absent from rough, likely due to separate layer, 3D
  asset, late edit, or compositing.
- `accept_reviewed`: visually reviewed and acceptable for the current training
  design.
- `accept_with_tag`: acceptable only if tagged for a known source-specific
  condition such as black fill, umbrella, prop, screentone, or panel fragment.

## ako5ver2

Raw archive:

- `dataset/raw_zips/dataset_ako5ver2.zip`
- zip root: `dataset_ako5`

Current variable-region review target:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/`

Important generated review files:

- `valid_mask_qc.png`
- `manifest.csv`
- `manifest_user_review_keep281.csv`
- `removed_user_mismatch.csv`
- `held_user_mask_insufficient_umbrella.csv`
- `flagged_user_review_20260725.csv`

### Layer / Prop Difference: Umbrella

Observed review indices:

- 69
- 79
- 86

Decision:

- hold out from the current training manifest
- label as `hold_mask_insufficient` / `hold_layer_difference`

Observation:

- The rough image contains an umbrella.
- The corresponding line art does not contain that umbrella.
- This likely means the umbrella was added later by 3D, a separate layer, or a
  compositing/editing step outside the rough-to-line manuscript pair.

Why this matters:

- If these rows are trained as normal rough-to-line pairs, the model is taught
  to remove a real rough object that is absent only because of the production
  layer pipeline.
- The current conservative valid mask is not sufficient for this case, because
  it does not fully protect the unmatched umbrella area.
- These rows are not simple geometric mismatches; they are source-pipeline
  differences.

Current handling:

- Excluded from `manifest_user_review_keep281.csv` together with hard
  mismatches.
- Recorded separately in:
  - `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/held_user_mask_insufficient_umbrella.csv`

Future handling options:

- improve valid-mask generation so rough-only prop/layer regions can be ignored
  instead of dropping the whole pair
- tag umbrella / prop-layer cases as a separate feature for later routing or
  source-specific modeling
- keep the pair only if manual mask correction removes the unmatched prop area
  from supervised loss

### Source Scale Spread In The 768-Normalized Manifest

Observed on `manifest_user_review_keep281.csv` (281 rows):

| quantity | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| `line_box` long side (source px) | 121 | 495 | 1288 | 2588 | 6947 |
| `src_per_out` (source px per 768-normalized px) | 0.16 | 0.64 | 1.68 | 3.37 | 9.05 |

Observation:

- Every region is normalized to a 768 px long side regardless of source size.
- A 480 px tile therefore covers between 76 and 4,342 source pixels.
- Visual check (`results/ako5ver2_keep281_scale_band_diagnostic.png`):
  - `src_per_out` below about 1.0: rough is a blurred smudge and the line target
    is nearly empty, which teaches the model to erase
  - `src_per_out` about 1.2 to 3.5: single character, face, or drapery scale
    with crisp strokes and genuine rough/line correspondence
  - `src_per_out` above about 4: multi-panel page composition, tiny figures,
    gray antialiased strokes, and heavy black fill

Why this matters:

- Measured over all 2,389 candidate tiles
  (`results/ako5ver2_keep281_tiles_480_measure_all.csv`), strict edge F1 at 3 px
  tolerance *rises* with downscaling: 0.169 below `src_per_out` 0.8 versus 0.511
  above 6.
- That is not better correspondence. Downscaling turns both rough and line into
  dense edge mush that overlaps everywhere.
- Any tile ranking driven by edge correspondence alone will therefore promote
  page-scale composition crops first. The earlier keep281-derived tile sets were
  ranked this way, which is consistent with the trained models behaving like
  density-map / ink-region predictors.
- Gray share of the drawn area confirms the degradation: the median region at
  `src_per_out` 3.37 has about half of its drawn area at gray levels rather than
  solid ink.

Current handling:

- `tools/pair_extraction/tile_region_manifest_480.py` records `src_per_out` and
  gates on a band with `--min-src-per-out` / `--max-src-per-out`.
- Score mode `strict` deliberately excludes `src_per_out` from the score, since
  the band gate handles scale and correspondence is scale-confounded.

Future handling options:

- re-materialize regions near native scale instead of forcing 768 px long side;
  at native scale the same 281 regions yield about 16,700 candidate 480 px tiles
  versus about 466 under the current normalization
- keep a separate downscaled composition track only if a panel-layout expert is
  ever wanted

### Native Re-Materialization

All 42 ako5ver2 source pages are 7016x4961. Scale spread in the 768-normalized
manifest therefore came entirely from proposed region size, not from page
resolution, so a single fixed source-to-output ratio normalizes the whole
dataset.

Divisor chosen by measuring output stroke width on matched source footprints:

| divisor | stroke-band tiles | median `line_width_p50` | median `soft_ink_ratio` |
|---:|---:|---:|---:|
| 1 (native) | 201 | 3.82 | 0.416 |
| 1.5 | 103 | 2.74 | 0.406 |
| 2 | 50 | 1.91 | 0.434 |
| 3 and above | <=15 | 1.91 (resolution floor) | 0.42 to 0.55 |

Native scale wins on every axis: strokes land in the healthy 2 to 4 px band for
480 px tiles, tile yield is highest, and gray share is lowest. Downscaling only
pushes strokes to the resolution floor while gray mush increases.

Result of `tools/pair_extraction/materialize_region_manifest_native.py`:

- source rows: 281
- kept: 165
- dropped below 480 px native: 116
- output: 1,224 Mpx, about 427 MB
- target: `dataset/regions_ako5ver2_native_20260725/`

The 116 dropped rows were only ever viable through upscaling, which produced the
blurred-rough / near-empty-line tiles described above.

### Pixel Thresholds Are Scale-Bound

Every pixel-unit threshold in the region pipeline was implicitly calibrated for
the 768 px normalization and is wrong at native resolution.

Alignment: reviewed `align_dx` / `align_dy` from the 768 space were badly
insufficient once scaled up. The native refine moved 137 of 165 regions, with a
median correction of 21 px and p90 of 74 px.

Valid masks: `--support-px 5` at native marks most real strokes as unsupported,
so the mask ignores exactly the content to be learned.

| `--support-px` | mean ignore ratio | mean unsupported line edge |
|---:|---:|---:|
| 5 (768 default) | 0.123 | 0.568 |
| 12 | 0.089 | 0.378 |
| 20 | 0.059 | 0.248 |
| 32 | 0.021 | 0.143 |

Adopted native mask settings:

- `--support-px 20 --window 61 --expand-ignore 16 --close-ignore 16`
- rationale: a line edge counts as supported when some rough edge lies within
  about 20 manuscript pixels, which matches observed sketch-to-line deviation

Tiler edge tolerances were rescaled the same way, from `--close-px 8` /
`--strict-close-px 3` at 768 to `--close-px 22` / `--strict-close-px 8` at
native.

### User-Reviewed Mismatches

Observed review indices:

- 42
- 58
- 59
- 81
- 90
- 98
- 112

Decision:

- exclude as `exclude_mismatch`

Recorded in:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/removed_user_mismatch.csv`

Current filtered manifest:

- `dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv`

Rows:

- source rows: 291
- excluded mismatches: 7
- held umbrella/mask-insufficient rows: 3
- kept rows: 281

## Residual Misalignment: Scale/Deformation, Not Just Translation

Observed 2026-07-26 while investigating why chamfer distance (alignment
quality) varies so much across sources and correlates with trained-model
output crispness (see `doc/region_dataset_extraction_policy.md`, "Alignment
Gate vs Style Gate"). User's production-process explanation, confirmed by a
direct test:

- Line art is inked from a printed rough, then scanned back in. Nothing in
  that workflow requires or rewards keeping rough and line pixel-aligned;
  they are stored together only so the files aren't lost, not for overlay
  precision. Any alignment has to be recovered after the fact.
- After inking, line art goes through a finishing pass that can rescale or
  reposition content per panel (koma), per character, or occasionally in a
  smaller partial region. This is a real geometric transform between rough
  and line, not just noise.
- Content redraws/omissions (rough has content the line omits or vice versa)
  are a separate problem already handled by the valid mask.

Test: added a joint translation+scale search (scales 0.85-1.15) on fitness's
5 worst-chamfer tiles (originally chamfer 20.8-21.8). Result: chamfer dropped
to 13.7-15.7 (25-38% reduction), and 3 of 5 tiles picked a non-1.0 scale
(0.85, 1.08, 1.15) as the best fit. Confirms real scale mismatch exists and a
translation-only search (which is all `match_kurip_regions.py` does, and all
`materialize_region_manifest_native.py`'s refine step does) cannot correct it.
`match_hamlabi_regions.py` does search a few discrete scale candidates, but
only a global scale per parent region, not per-panel or per-character.

Decision: local mesh-level (non-uniform) deformation is out of scope for now;
too open-ended to model generally, revisit only if a good general method
appears later. Uniform scale + translation, applied at the right region
granularity, is the current target.

### Planned Fix: Panel-Boundary-First Region Segmentation

Root cause of "one region contains 3 panels" / "region cuts across a panel
boundary" (both observed in `match_hamlabi_regions.py`'s output): its
`region_proposals()` groups regions purely by line-ink connected-component
proximity (dilate + connected components on inked pixels). It has no concept
of a panel border at all, so nearby panels with close ink content merge into
one region, and proposals can freely cross a real panel boundary.

A first attempt at panel-border detection (long-line morphology: open with a
long horizontal/vertical structuring element to isolate straight runs) failed
on a real ako5ver2 page: it flagged character hair strokes as false panel
borders instead of finding real ones. On inspection, that page's panel borders
are not clearly distinguishable from character line art by simple
darkness+straightness alone.

Key fact from the user: panel border lines are composited from a **separate
layer** in the original production file. The finished line art layer itself
generally does not contain panel border lines at all. The rough sometimes has
ruled panel lines, sometimes only a rough sketch of them. This means panel
borders cannot be reliably recovered from the flattened rough/line raster
images alone — the panel-border layer must be extracted separately from the
source layered file.

Plan (paused here, panel-layer extraction happens on another machine):

1. Extract the panel-border layer as its own dataset (separate PC, out of
   scope for this session).
2. Use that layer as the segmentation starting point: cut pages into clean
   single-panel regions first (full-bleed/"buchinuki" panels that cross the
   frame are expected to be hard and can be deferred to a later stage).
3. Stage 1 (panel-level): within one panel, expect translation + a single
   uniform scale only (no mesh deformation), so alignment search/verification
   should be more tractable than the current whole-page approach.
4. Stage 2 (character-level): if a panel has multiple characters, each may
   have an independent scale adjustment; split into character-level
   sub-regions and re-run alignment search/scoring per character. Only
   low-scoring cases need manual visual review at this stage.
5. Stage 3 (finer sub-character regions): deferred; deformation here may be
   irregular/"special" per the user, needs separate consideration once stages
   1-2 are working.

## Environment: Long Background Jobs Get Silently Killed

Observed 2026-07-26 while extracting `fitness` (formerly `kurip`) and
`housei`: this environment kills long-running extraction processes somewhere
around 10-13 minutes of wall-clock time, with no traceback, no OOM entry in
`dmesg`/`journalctl`, and no correlation to row count (different runs died at
different row counts processing the same file). Confirmed not a memory issue:
`free -h` showed ample headroom at the time of failure.

This affected three different execution methods equally:

- a detached `nohup ... & disown` process (untracked by the harness)
- a properly harness-tracked `run_in_background: true` Bash task
- two overnight autonomous background agents (their transcripts became
  unrecoverable; `SendMessage` to their agent IDs returned "No transcript
  found")

A command piped through `| tail` or `| tee` can also make this failure
invisible: those commands exit 0 even when the upstream process was killed
partway, so a "completed exit code 0" task notification is not proof the
underlying script finished. Always check that the expected output file
actually exists, not just the reported exit code, for any long extraction run
in this environment.

Workaround: split long extraction/filter passes into short chunks (~250 rows
took a few minutes each, safely under the failure window) using
`--offset`/`--limit`/`--append` on `filter_matched_region_tiles.py`, running
each chunk as its own foreground tool call rather than one long background
job. Verify each chunk's output file before starting the next.

## fitness (formerly kurip)

Raw archive:

- `dataset/raw_zips/dataset_fitness_v4.zip`

Renamed from `kurip` because that name was a person's username. All
pre-session leak-era `kurip`-named data (2,526 same-coordinate-extraction era
tiles under `dataset/pairs_480/train/rough`, 11 `line_kurip_*` directories, 29
training lists, ~40 `results/kurip_*` files, 9 checkpoints, 8
`make_kurip_*_compare.py` one-off comparison scripts, 4 dead experiment
runners) was deleted rather than renamed, per explicit user decision that
leak-era material did not need to be preserved. Only this session's reviewed
native-strict extraction (271 tiles) was kept and renamed to `fitness`.

`tools/pair_extraction/match_kurip_regions.py`,
`tools/pair_extraction/extract_kurip_matched_tiles.py`,
`tools/pair_extraction/prepare_kurip_tiles.py`,
`tools/pair_extraction/refine_kurip_vlm_tiles.py`, and
`tools/pair_extraction/vlm_review_kurip_matches.py` are still-active current
infra (not leak-era) and were deliberately left un-renamed this session;
`prepare_kurip_tiles.py` in particular is shared with hamlabi, so renaming it
is a separate, larger decision than the fitness data rename. Do not delete
these on sight because of the name.

Route and result: see `doc/dataset_status.md` (`## fitness` section).

## housei

Raw archive:

- `dataset/raw_zips/dataset_housei.zip`

Manifest schema differs from ako5/fitness/kurip: entries have an explicit
`"page"` key (e.g. `"page0010"`) and no `"file"` key. The generic
`page_id()`/`page_lookup()` helpers used across these tools fall back to
deriving an identifier from the `"sketch"`/`"line"` filename when `"file"` is
absent, which produces a working but inelegant page id (e.g.
`housei_016_sketch` instead of a clean page number) in this manifest CSV's
`page` column. Not a bug, just a naming quirk to expect when reading housei's
region/tile CSVs.

Also flat zip layout (no subfolder root): pass `--zip-root ""`.
`diagnose_pair_alignment.py` needed a small fix to support this (see
`doc/EXTRACTION_RULES.md` change log implicit in this session; the fix added a
`read_zip_member` fallback to bare filename, mirroring the pattern already
used in `materialize_region_manifest_native.py` and `match_hamlabi_regions.py`).

Route and result: see `doc/dataset_status.md` (`## housei` section).

### Panel-Border Layer Delivered (2026-07-26): Koma Panel Segmentation

`dataset/raw_zips/dataset_housei_v2.zip` (kept alongside the original as
`dataset_housei.zip`) adds a per-page panel-border-only layer,
`housei_NNN_koma.jpg`, plus `koma_manifest.json` (per-page `sources`: which
named `コマ` vector/raster layer(s) in the original production file the koma
image traces, and an `ink` ratio). This is the panel-border-layer dataset
"Planned Fix: Panel-Boundary-First Region Segmentation" (below) was waiting
on, delivered for `housei` only so far (not ako5ver2/hamlabi). Zip layout also
changed: files now live under a `dataset_housei/` subfolder, so tools need
`--zip-root dataset_housei` for the v2 archive, not `""`.

New tool: `tools/pair_extraction/match_koma_panels.py`. Two stages, both
dry-run/candidate-generation only:

1. `detect_panels()`: the koma layer contains only panel-border ink on an
   otherwise blank page. Panel interiors are the connected components of the
   *non-ink* area (after closing small border gaps with a morphological
   close), excluding whatever component(s) touch the page edge (outer
   margin/background). Verified against the koma reference image directly:
   detected boxes matched the visual panel layout exactly on every page
   checked, including a page with a nested inset panel (handled correctly as
   two separate components, the inset and the surrounding ring).
2. Per-panel alignment: reuses the fixed alignment-gate primitives
   (`edge_map`/`support_f1`/`chamfer`, `ALIGNMENT_*` constants) from
   `tile_region_manifest_480.py`, extended with a uniform-scale search
   (0.85-1.15), per the already-documented per-panel scale/deformation
   finding above.

Performance note (reusable beyond housei): an initial implementation scored
every translation candidate via full boolean-array indexing
(`line_support[rough_edge_window]`) over whole-panel-sized arrays, which cost
O(panel area) per candidate and made a 2-page test take 11m40s. Rewriting to
index sparse edge-pixel coordinates (`np.nonzero`) instead, so per-candidate
cost scales with ink density rather than panel area, cut the same test to
2m57s (~4x) with byte-identical results. Even so, a translation search wide
enough to avoid boundary-hit false optima (needed `--max-shift 160`, not the
initial `64`, since many pages' true offset was 60-160px) makes the full
18-page/81-panel batch too slow for one process given this environment's
background-job kill window (see "Environment: Long Background Jobs Get
Silently Killed" above); it was run in 6 chunks of 3 pages
each (~3.5-5.5 min/chunk) using `--start-page`/`--end-page`/`--append`.

Two distinct anomalies found on full-resolution review, illustrating why
review must happen before any production use:

- **`housei_004` (page0001): true page-level asset mismatch, not an alignment
  problem.** All 7 panels hit the `--max-shift 160` search boundary with
  chamfer staying high (30-40) even at the best found offset. The whole-page
  downsampled edge overlay showed the line art drawn at a substantially larger
  scale than the rough, spilling far outside the rough's corresponding
  content — not fixable by translation/scale search at any reasonable range.
  Root cause, confirmed by the user: this page's rough is actually a ラフ
  (layout-stage rough), not the 下絵 (shitae/underdrawing) the rest of the
  source pairs use — visible in the manifest as a differently-formatted
  `sketch_layer_name` (`"n.houseiA_001-0"` vs. the normal
  `"p.housei_NNN"` pattern; this is the *only* entry among all 18 pages with
  the `"n."` prefix, confirmed by listing every page's `sketch_layer_name`).
  Checked whether the previously-used (pre-koma) zip had a better rough for
  this page: `housei_004_sketch.jpg` is byte-identical (same md5) between
  `dataset_housei.zip` and `dataset_housei_v2.zip`, so this was already the
  asset in use before the koma layer was added, not a regression from the v2
  upload. No fix available from data already on hand; would need the actual
  shitae layer re-extracted from the source production file. Decision: hold
  `housei_004` out of this batch entirely (`hold_layer_difference`-style, but
  at the whole-page level rather than a sub-region).
- **`housei_010`/`housei_011`/`housei_012` (page0007/page0012/page0019): a
  distinct cluster of much higher residual misalignment, but real content
  correspondence.** Per-page median best-chamfer is 27.8-34.4 for these three
  vs. 11.2-16.5 for every other page in the batch; 10 of 74 non-housei_004
  panels landed within 20px of the `--max-shift 160` boundary, concentrated in
  these three pages. Unlike `housei_004`, full-resolution panel crops confirm
  correct rough/line content correspondence throughout (same characters, same
  poses, recognizable after alignment) — this is not a wrong-page or
  wrong-asset mismatch. Read as the already-documented non-uniform
  deformation/residual-misalignment finding (see "Residual Misalignment"
  above) showing up more severely here, plausibly because these panels have
  busier or more dynamic content (multiple overlapping characters, action
  poses) than the rest of the batch. Their `sketch_layer_name` is normal
  (`p.housei_NNN`), ruling out the `housei_004`-style asset-type explanation.
  Decision: keep these panels in the batch (do not exclude like `housei_004`,
  since the content match is real), but flag them as high-residual-alignment
  for a later chamfer-based quality gate to filter naturally at the tile
  level, rather than forcing a page-specific fix into the general pipeline.

Batch result (81 panels detected across 17 processed pages, `housei_004`
excluded from the count below):

- `results/housei_koma_panels_20260726.csv` / `.json`: 74 accepted-page rows
  (housei_004's 7 rows also present in the file but should be excluded on
  read)
- chamfer median: base 17.24 -> best 14.55 (mean 20.84 -> 17.02)
- QC: `results/housei_koma_panels_20260726_qc_chunk{1..6}.png` (per-chunk,
  not combined); page-level panel-box overlays:
  `results/housei_koma_panels_20260726_overlays/housei_NNN_panels.png`

### Panel-Level Quality Gate And Materialization (2026-07-26)

User direction: page-level mismatch (housei_004) is exclude-worthy, but a
page with some bad panels (housei_010/011/012) should still contribute its
good panels rather than being dropped wholesale — filter at panel
granularity, not page granularity.

Chamfer distribution across the 74 non-housei_004 panels showed a clean
natural gap: p80=20.02, p90=28.92 (stable count for cutoffs 20-25, so this is
a real gap, not a fitted elbow) and the gap boundary lines up exactly with
the housei_010/011/012 cluster identified earlier. Adopted `chamfer <= 20.0`
as the panel-level accept gate: 59/74 panels pass, rejecting all of
housei_010 (5) + housei_011 (4) + housei_012 (3) plus 3 individual weak
panels from otherwise-good pages (housei_001 panel1, housei_018 panel3, and
one more). Visually reviewed the full accepted set at native tile resolution
(`results/housei_koma_panels_20260726_materialized_qc.png`) before and after
the gate — clean throughout, including the tail near the cutoff.

Contrast-adjustment side investigation (before settling on the chamfer gate):
tested autocontrast cutoff=1/2, CLAHE, and percentile stretch against the
existing autocontrast cutoff=0 baseline on housei_010/011/012 panels plus a
clean reference (housei_002). All four alternatives gave negligible F1/chamfer
change (within ~0.01 F1 / ~1 chamfer point) on every page including the clean
one — contrast/faintness is not the bottleneck for these three pages; the gap
is a genuine structural stroke-position mismatch (loose rough drawing and/or
real non-uniform deformation), not an image-processing artifact. This
superseded an earlier, now-retracted hypothesis that faint rough contrast
explained the cluster.

New tool: `tools/pair_extraction/materialize_koma_panels.py`. For each
accepted panel, re-crops the rough page at its already-found best
`(dx, dy, scale)` and the line page at the raw panel bbox, both at native
pixel resolution, matching the line panel's exact pixel dimensions (no
`line_box` column is written, since these panels are native 1:1 by
construction — see note below). Dry-run writes a QC montage only; `--save`
writes PNG pairs plus `manifest.csv`/`manifest.json` with a `native_long_side`
column, directly compatible with `build_region_valid_masks.py`.

Downstream pipeline reused as-is (no changes needed), matching the already-
validated ako5ver2/fitness/housei native-strict recipe:

1. `materialize_koma_panels.py --save` -> `dataset/regions_housei_koma_panels_20260726/`
   (59 rough/line PNG pairs + manifest)
2. `build_region_valid_masks.py --image-size 0 --reuse-source-images
   --support-px 20 --window 61 --expand-ignore 16 --close-ignore 16` (same
   native-scale settings validated for ako5ver2 native and housei) ->
   `dataset/regions_housei_koma_panels_20260726_masked_line_conservative/`
3. `tile_region_manifest_480.py` with the ako5ver2-validated strict gates
   (`--ink-min 0.012 --ink-max 0.08 --max-black-component-ratio 0.025
   --max-thick-ink-ratio 0.015 --max-line-width-p50 6.0
   --max-long-line-ratio 0.25 --min-support 0.90 --score-mode strict
   --duplicate-overlap 0.50 --max-per-region 4 --min-tile-score 2.5
   --dedup-scope page`), plus housei's already-established
   `--max-soft-ink-ratio 0.50` relaxation

One real gotcha, not specific to koma panels: the `src_per_out` region-scale
gate (`--min-src-per-out`/`--max-src-per-out`) requires a `line_box` column
to compute a real scale ratio; without it, `region_scale_stats()` falls back
to `source_long_side = native_long_side`, giving a flat `src_per_out = 1.0`
for every region, which a `--min-src-per-out 1.2` gate (ako5ver2's validated
value) then rejects entirely (`region rejects: src_per_out_low=59`, 0
accepted on the first attempt). This gate exists to catch *resized* regions
(where source and output resolution differ); koma panels are native 1:1 by
construction (no resize step), so the gate is inapplicable and was left at
its default `0` (disabled) rather than worked around with a synthetic
`line_box`. Record this if any other native-1:1-by-construction source hits
the same all-rejected symptom.

Also cosmetic-only gotcha: `tile_region_manifest_480.py --name-prefix`
defaults to the literal string `"ako5k281"` (hardcoded leftover from the tool's
original ako5ver2-only origin) — output tile filenames read
`ako5k281_NNNN_...` regardless of source unless `--name-prefix` is passed
explicitly. Cosmetic only (`source_name`/`source_page` columns in the CSV are
correct regardless), but always pass an explicit `--name-prefix` for a new
source to avoid confusing filenames.

Final result: 58 tiles accepted from 287 raw candidates across 59 panels
(59/59 regions used). Saved:

- list: `dataset/pairs_480/valid_train_housei_koma_native_strict_20260726.txt`
- line dir: `dataset/pairs_480/train/line_housei_koma_native_strict_20260726`
- rough dir: shared `dataset/pairs_480/train/rough` (tiles prefixed
  `houseikoma_`)
- tile CSV: `results/housei_koma_tiles_480_strict.csv`
- QC: `results/housei_koma_tiles_480_strict_qc.png` /
  `_qc_tail.png` (tail reviewed at full native tile resolution before saving:
  sparse/simple single-stroke fragments, no content mismatches)
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_housei_koma_native_strict.csv`)

Status: superseded by the sub-region split below (housei_004 fix + content-
density splitting produced a larger, cleaner set from the same panels). The
58-tile line dir/list above are left on disk as historical reference, not
deleted, but should not be used for new work — use
`housei_koma_subregion_native_strict_20260726` instead.

### housei_004 Fixed (v3), And Ink-Density Sub-Region Split (2026-07-26)

Two follow-ups landed together, both increasing yield over the 58-tile panel-
level set above.

**housei_004 fixed.** User supplied `dataset/raw_zips/dataset_housei_v3.zip`
with a replaced `housei_004_sketch.jpg` (confirmed as the only changed file
vs. v2 by hashing every member of both archives). The manifest's
`sketch_layer_name` field for this entry is still the stale
`"n.houseiA_001-0"` label (not updated), but the image content is a genuine
下絵 now: a whole-page downsampled edge overlay against the line page shows
tight scale/position correspondence throughout, unlike the wildly
oversized-relative-to-rough line content seen with the old ラフ asset.
Re-ran `match_koma_panels.py` for this one page only (page index 3) against
v3: chamfer improved from a 30-40 range (all 7 panels boundary-hit and
excluded) to a 12.3-18.6 range (all 7 panels now pass the `chamfer<=20.0`
gate). Merged these 7 rows into `results/housei_koma_panels_20260726.csv`,
replacing the old bad housei_004 rows. Panel-level accepted count is now
66 (up from 59).

**Ink-density sub-region split.** Root-caused via a per-gate funnel
measurement (evaluating every gate independently on a stride=480,
non-overlapping sample of 480px windows across all accepted panels, mirroring
`diagnose_gate_funnel.py`'s approach but implemented ad hoc against this
region-manifest pipeline since that tool is built for the other zip-based
route): `ink_range` alone rejected 80.9% of candidate windows (median line
ink density 0.0018, far below the 0.012 floor; 75.9% specifically too
sparse), `soft_ink_ratio` rejected 61.5%, while the alignment gate rejected
only 0.7%. Diagnosis: koma panels are defined by panel-border *geometry*, not
content density (unlike ako5ver2's region proposals, which are built from
ink-connected-components in the first place), so a large fraction of a
panel's area is blank background/margin, and a naive 480px sliding window
over the whole panel wastes most candidates there.

New tool: `tools/pair_extraction/split_koma_panel_subregions.py`. Reuses the
same ink-connected-component region-proposal logic already validated for
hamlabi's whole-page region finding (`match_hamlabi_regions.py`'s
`region_proposals()`/`line_ink_mask()`, same default thresholds — no
rescaling needed since koma panels are native-resolution crops from the same
source pages), scoped to one already-aligned koma panel instead of a whole
page. Sub-regions inherit the panel's already-verified alignment (rough and
line were already re-cropped pixel-aligned at materialization time using the
panel's best found `dx`/`dy`/`scale`), so no re-alignment is needed, only a
content-aware crop. If a panel has no dense sub-region, the whole panel is
kept as a fallback so nothing is silently dropped (0/66 panels needed the
fallback in practice).

Performance note: the 121px morphological dilate kernel (reused unchanged
from `match_hamlabi_regions.py`, calibrated for whole-page grouping) is slow
on some of these large panel images — a 66-panel dry-run took 6m52s, right at
the edge of this environment's background-job kill window (see
"Environment: Long Background Jobs" above). It finished both times it was
run, but if this needs to run on more panels later (ako5ver2/hamlabi), either
downscale the image before the dilate+connected-components step (scaling
boxes back up afterward) or chunk the run.

Result: 179 sub-regions from 66 panels (avg 2.7/panel, 0 fallback). Ran
through the unchanged `build_region_valid_masks.py` (same native settings)
and `tile_region_manifest_480.py` (same strict gates) pipeline:

- tiles: 75 accepted from 349 raw candidates (up from 58/287 at the panel
  level — the sub-region split alone improved yield ~29% at matched gates,
  on top of the 7 additional panels housei_004 contributed)
- list: `dataset/pairs_480/valid_train_housei_koma_subregion_native_strict_20260726.txt`
- line dir: `dataset/pairs_480/train/line_housei_koma_subregion_native_strict_20260726`
- rough dir: shared `dataset/pairs_480/train/rough` (tiles prefixed
  `houseikomasub_`)
- QC reviewed top and tail at full native tile resolution before saving:
  same quality pattern as the panel-level set (tight correspondence at the
  top, sparse-but-correct single-stroke fragments at the tail, no mismatches)
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_housei_koma_subregion_native_strict.csv`)

Status: superseded by the per-sub-region alignment refinement below.

### Per-Sub-Region Alignment Refinement (2026-07-26)

User's observation, from looking at the sub-region tile QC directly: "panel
by panel the same picture is there, but zoomed in it's still fairly
misaligned — how much would character/region-level alignment within one
panel improve this?" This is exactly the gap in the sub-region split above:
each sub-region inherited its parent panel's single `(dx, dy, scale)`, but
that transform is only the best *average* fit for the whole panel — a busy
panel with multiple content islands (e.g. two characters at different
depths) can have per-island residual misalignment, the same non-uniform-
deformation finding already documented for whole panels, recurring one level
down.

Added `--refine-alignment` (on by default) to
`split_koma_panel_subregions.py`: after finding each sub-region box, runs a
small local translation+scale search (`--refine-max-shift 48
--refine-shift-step 8`, default `PANEL_SCALES` 0.85-1.15) starting from the
panel's own alignment rather than a wide from-scratch search, since the
panel is already roughly right and only a small residual needs finding.
Reuses the same sparse-edge-coordinate scoring approach as the panel-level
search in `match_koma_panels.py` for speed.

Result across all 179 sub-regions (3 chunks): sub-region chamfer median
improved modestly at every chunk (14.54->13.40, 13.08->11.96, 14.31->12.85 —
roughly 8-11% each), confirming the effect is real but modest in aggregate,
consistent with panels already being reasonably aligned overall (chamfer<=20
gate). Individual sub-regions occasionally needed a substantial correction
(e.g. one case found a 32px shift, chamfer 19.1->16.3), confirming genuine
per-content-island residual misalignment exists, not just noise — but most
sub-regions only needed a small nudge.

Re-tiled through the same unchanged mask+tile pipeline:

- tiles: 85 accepted from 372 raw candidates (up from 75/349 without
  refinement, up from 58/287 at the original whole-panel level — refinement
  alone added ~13% on top of the sub-splitting's ~29%)
- list: `dataset/pairs_480/valid_train_housei_koma_subregion_refined_native_strict_20260726.txt`
- line dir: `dataset/pairs_480/train/line_housei_koma_subregion_refined_native_strict_20260726`
- rough dir: shared `dataset/pairs_480/train/rough` (tiles prefixed
  `houseikomasubrefined_`)
- QC reviewed top and tail at full native tile resolution: same quality
  pattern as every earlier stage (tight correspondence at top, sparse-but-
  correct single-stroke fragments at tail, no mismatches)
- integrity audit: 0 findings
  (`results/pair_dataset_integrity_summary_housei_koma_subregion_refined_native_strict.csv`)

Full progression across all three koma-panel extraction stages, same source
panels throughout: **58 -> 75 -> 85 tiles** (whole-panel tiling -> ink-density
sub-region split -> + per-sub-region alignment refinement).

Status: reviewed, saved, integrity-audited; this is now the current housei
koma-panel training source, superseding both the 58-tile panel-level-only set
and the 75-tile unrefined-sub-region set above (both left on disk as
historical reference, not deleted, but should not be used for new work). Not
yet trained on; not yet mixed with the existing `housei_native_strict`
grid-based tile set (65 tiles, different extraction route — grid+local-offset
anchored vs. this panel/sub-region-anchored route) — keep as a separate
source until a deliberate mixing/comparison experiment is designed.

## Template For New Raw Dataset Notes

Use this shape for future entries:

```text
## <dataset_name>

Raw archive:

- `<path>`

Current review target:

- `<path>`

### <Issue Name>

Observed rows / review indices:

- ...

Decision:

- ...

Observation:

- ...

Why this matters:

- ...

Current handling:

- ...

Future handling options:

- ...
```
