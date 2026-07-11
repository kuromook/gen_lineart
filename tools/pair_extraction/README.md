# Pair Extraction Tools

Scripts in this folder prepare paired rough/line datasets from raw source data.
Run them from the repository root so their relative paths resolve against the
standard dataset layout:

```text
dataset/
  raw/
  pairs/
  pairs_256/
  pairs_480/
```

Typical flow for a new raw dataset:

1. Diagnose whether same-coordinate extraction is valid with
   `diagnose_pair_alignment.py`.
2. If same-coordinate extraction is valid, use the dataset-specific preparation
   script such as `prepare_kurip_tiles.py` or `prepare_ako5.py`.
3. If the raw pages are shifted or rotated, search candidate matches with
   `match_*_regions.py` and extract accepted pairs with an `extract_*` script.
4. Optionally run `vlm_review_kurip_matches.py` for visual secondary filtering.
