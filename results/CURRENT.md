# Current Results

Updated: 2026-07-18 22:30 JST

`results/` contains many historical experiment artifacts. Do not infer current
model quality from the presence of a file alone.

Do not inspect `results/archive/` unless the user explicitly asks for archived
results or historical audit material.

## Current Clean Evaluation

Primary clean lineart004 comparison:

- `results/compare_clean_baselines_lineart004.png`
- `results/fixed_output_metrics_base_clean_lineart004_compare.csv`

Summary:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 |
| `shape1_base_clean_unique_bce` | 0.2786 | 7.718 | 1.600 |

`shape1_base_clean_unique_bce` is not an improvement.

## Pending Epoch020 Evaluation

The watcher `lineart-watch-moredupes-epoch020-eval.service` will create:

- `results/shape1_std15_clean_split_moredupes_bce_epoch020/`
- `results/fixed_output_metrics_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.csv`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`

## Deprecated For Adoption

These are useful for audit but should not drive adoption decisions:

- `results/archive/leaky_pre_clean_eval/`: old leaky/pre-clean-eval model
  outputs, comparisons, and metrics moved out of the top-level namespace.
- `checkpoints/kurip/`: dirty kurip fine-tune checkpoint.
- Broad historical top-level `compare_kurip_*` montages from before clean eval.
- One-off outputs moved to `results/archive/legacy_root_oneoffs/`, including
  `housei_001_*`, `gan_130_*`, `edge_loss*.png`, `simple.png`,
  `sharp_output.png`, `plan2.png`, and stray `lineart_004_*_out.png` files.

## Cleanup Rule

Prefer adding entries here or to `config/results_manifest.json` before moving
files. Many scripts still use stable `results/` paths directly.
