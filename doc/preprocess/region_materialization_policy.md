# Region Materialization Policy

## Source Format

Region datasets should be kept first as variable-aspect manifests:

- rough path
- line path
- source page
- line bbox
- rough bbox
- review decision
- feature tags

The canonical hamlabi review artifact is currently:

- `dataset/regions_hamlabi_final_review/manifest.csv`
- `dataset/regions_hamlabi_final_review/manifest.json`

## Training Loader Policy

Preferred loader mode:

- read the region manifest directly
- preserve aspect ratio
- resize by long side
- pad to a square with white background

In `scripts/train_i2i_survey.py`, use:

```bash
--region-manifest dataset/regions_hamlabi_final_review/manifest.csv \
--region-fit-mode square_pad \
--image-size 480
```

`--image-size 768` is allowed for experiments, but expect much higher memory
cost. Use small batch sizes.

## Native-Scale Re-Materialization

Long-side normalization is not scale-neutral. When a manifest's source regions
span a wide size range, normalizing every region to the same long side makes
output pixels mean different amounts of manuscript, and downstream fixed tiles
then mix upscaled fragments with downscaled page composition. See the scale-band
section of `doc/region_dataset_extraction_policy.md`.

When all source pages share a resolution, prefer re-materializing at a fixed
source-to-output ratio instead:

```bash
venv/bin/python tools/pair_extraction/materialize_region_manifest_native.py \
  --manifest <reviewed-region-manifest.csv> \
  --zip dataset/raw_zips/<archive>.zip \
  --zip-root <root> \
  --out-base dataset/<name>_native_<date> \
  --source-scale-divisor 1.0 \
  --save
```

Notes:

- `--source-scale-divisor 1.0` keeps native manuscript resolution. Choose the
  divisor by measuring output stroke width, not by picking a long side; target
  roughly 2 to 4 px median stroke width for 480 px training tiles.
- Alignment must be redone. Offsets found at a smaller normalized size scale up
  with the region's downscale factor, so the tool uses the reviewed
  `align_dx` / `align_dy` only as a starting offset and refines at output scale.
- `--min-output-size` drops regions whose native size cannot fill one tile.
  Those regions were only ever viable through upscaling.

Downstream stages accept native variable-aspect input:

- `tools/pair_extraction/build_region_valid_masks.py --image-size 0` skips square
  fitting, and `--reuse-source-images` avoids a second full-size image copy.
- `tools/pair_extraction/tile_region_manifest_480.py` re-reads tiles on demand
  instead of holding every candidate in memory, which native-resolution regions
  would otherwise exhaust.

## Fixed Materialization Policy

Only materialize fixed-size copies when a downstream tool cannot read region
manifests.

Allowed modes:

- `square_pad`: preferred, preserves aspect ratio
- `resize_stretch`: compatibility only, distorts geometry and should not be the
  default for raw manuscript regions

Use:

```bash
venv/bin/python tools/pair_extraction/materialize_region_manifest_square.py \
  --manifest dataset/regions_hamlabi_final_review/manifest.csv \
  --out-base dataset/regions_hamlabi_final_review_480_squarepad \
  --size 480 \
  --fit-mode square_pad \
  --name-prefix hamlabif
```

For 768:

```bash
venv/bin/python tools/pair_extraction/materialize_region_manifest_square.py \
  --manifest dataset/regions_hamlabi_final_review/manifest.csv \
  --out-base dataset/regions_hamlabi_final_review_768_squarepad \
  --size 768 \
  --fit-mode square_pad \
  --name-prefix hamlabif
```

## Promotion Rule

Do not append region materializations to existing `dataset/pairs_480/*.txt`
lists by default.

If a fixed-size materialization is used for training, keep it in its own dataset
directory with its own manifest and file list. Merge only after a deliberate
experiment design specifies source ratios and sampling weights.
