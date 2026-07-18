# Metadata Tools

These scripts build category metadata used by the MoE branch experiments.

- `build_pair_metadata.py` creates `dataset/pairs_480/pair_metadata.csv` from
  existing training lists and filename conventions.

Dataset metadata outputs live under `dataset/` and are ignored by git. Regenerate
them from the scripts when needed.
