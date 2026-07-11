# Checkpoints Layout

Model checkpoints are stored under this directory instead of root-level
`checkpoints_*` folders.

Mapping from the old layout:

- `checkpoints/` -> `checkpoints/base/`
- `checkpoints_shape1/` -> `checkpoints/shape1/`
- `checkpoints_kurip_clean540/` -> `checkpoints/kurip_clean540/`
- `checkpoints_dataset_gate/` -> `checkpoints/dataset_gate/`
- `checkpoints_<name>/` -> `checkpoints/<name>/`

Checkpoint files are generated artifacts and are ignored by default. Force-add
only small or release-critical snapshots when needed.
