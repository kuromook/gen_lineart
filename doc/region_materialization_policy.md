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
