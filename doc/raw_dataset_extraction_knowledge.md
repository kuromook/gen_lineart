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
