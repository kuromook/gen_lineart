# Raw Pair Extraction Rules

Updated: 2026-07-18

Use this when creating or rebuilding training pairs from raw manuscripts.
Historical extraction logs are archived under `doc/archive/*.txt`; do not use
their model-quality conclusions as current guidance.

## Non-Negotiable Flow

1. Create a dry-run first. Do not write training pairs on the first pass.
2. Diagnose rough/line alignment before same-coordinate tiling.
3. Generate CSV plus QC montage for every extraction candidate set.
4. Review QC before `--save`.
5. Build train/eval lists only after saved files are stable.
6. Run integrity audit before training.
7. Record the exact list, line directory, QC, metrics, and known exceptions.

## Alignment Preflight

Run `tools/pair_extraction/diagnose_pair_alignment.py` for any new raw dataset.

Use the recommendation as the route:

- `same_coordinate_ok`: same-coordinate tiling is allowed.
- `needs_global_or_local_alignment`: align or locally refine before extraction.
- `needs_region_matching`: search corresponding rough regions from line anchors.
- `hold_for_manual_review`: do not train from this dataset yet.

The kurip failure came from trusting same-coordinate extraction too early. Do
not repeat that.

## Matching Strategy

When page or tile correspondence is uncertain:

- Match pages or regions before tile extraction.
- Prefer line-anchored extraction: choose clean line tiles, then find the rough
  crop that best corresponds to that line crop.
- Keep original page IDs, rough coordinates, line coordinates, offsets, and
  match scores in CSV.
- Use VLM or manual review only as a secondary filter, not as a replacement for
  geometric/edge checks.

Relevant scripts:

- `tools/pair_extraction/match_ako5_pages.py`
- `tools/pair_extraction/match_ako5_regions.py`
- `tools/pair_extraction/match_kurip_regions.py`
- `tools/pair_extraction/extract_ako5_region_tiles.py`
- `tools/pair_extraction/extract_kurip_matched_tiles.py`
- `tools/pair_extraction/vlm_review_kurip_matches.py`

## Scale Check

Before committing a new raw extraction scale, run a crop-scale diagnostic when
the source manuscript resolution or visible context differs from existing data.

Relevant script:

- `tools/pair_extraction/evaluate_crop_scales.py`

Track these fields in CSV or metadata for scaled crops:

- `source_width`
- `source_height`
- `crop_source_size`
- `output_size`
- `scale_factor`
- rough and line crop coordinates

Previous useful diagnostic range was `480,720,960,1200 -> 480`, but treat that
as a search range, not a fixed rule.

## Candidate Filters

Use conservative filters unless the experiment explicitly studies relaxed data.

Current defaults to preserve:

- rough std must be high enough after the chosen rough preprocessing.
- line ink must avoid near-blank and black-fill tiles.
- rough and line edge counts must be sufficient.
- edge F1 and chamfer must both pass.
- orientation entropy should reject trivial straight fragments.
- support/coverage must be near complete for warped crops.
- deduplicate overlapping tiles before writing lists.

Known concrete defaults from the ako5 high-confidence extractor:

- tile size: `480`
- stride: `240`
- min region score: `4.0`
- min tile score: `3.0`
- min rough std: `15.0`
- line ink: `0.015` to `0.15`
- min support: `0.98`
- min edge pixels: `300`
- min edge F1: `0.35`
- max chamfer: `8.0`
- min orientation entropy: `0.55`
- duplicate overlap: `0.60`

These are starting points, not universal thresholds. If changed, write the
reason and the resulting QC artifacts.

## QC Requirements

Every extraction run must produce:

- full candidate CSV
- top-ranked QC montage
- tail or low-ranked QC montage when available
- evenly-spaced sample QC montage when available
- final saved list only after visual review

Reject or revise if QC shows:

- wrong page or wrong panel
- rough/line semantic mismatch
- local shift beyond intended matching
- line art not present in rough
- large black fills or screentone-like noise
- mostly blank tiles
- tiny line fragments unless the experiment explicitly targets fragments
- train/eval page or tile leakage risk

## Save Policy

`--save` should only happen after QC review. Save into explicit experiment paths
when possible, especially for alternate cleaned line directories.

Avoid overwriting stable lists or shared line directories. Prefer creating a new
list named with the extraction condition.

## Integrity Audit

Before training, run:

```bash
./venv/bin/python tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists <train-list> \
  --eval-lists dataset/pairs_480/valid_test.txt dataset/pairs_480/eval_fixed_clean_lineart004.txt \
  --output-summary results/pair_dataset_integrity_summary_<experiment>.csv \
  --output-findings results/pair_dataset_integrity_findings_<experiment>.csv
```

If using alternate line directories, pass explicit overrides:

```bash
--line-dir-override <source>=<line-dir>
```

Training candidates should have:

- no train/eval canonical tile overlap
- no train/eval exact rough hash overlap
- no train/eval exact line hash overlap
- no missing rough files
- no missing line files
- no train-internal exact duplicates unless explicitly allowed by the experiment

## Metadata

After extraction, update or regenerate metadata when category-aware experiments
matter:

- `tools/metadata/build_pair_metadata.py`
- `tools/metadata/make_content_review.py`

Minimum metadata to preserve:

- source dataset
- content category if known
- pair quality
- alignment quality
- rough path
- line path
- crop scale
- notes for manual/VLM review decisions
