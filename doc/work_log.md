# Work Log - Clean Rebuild

## 2026-07-18

### Reset Context After Leakage

The previous project history contained substantial leakage-driven assumptions.
Treat pre-clean-eval model scores and conclusions as historical only.

Current rule:

- Do not use old `shape1` metrics as adoption targets.
- Use leaky `shape1` only as a visual failure reference.
- Prefer clean lineart004 evaluation and integrity-audited training lists.
- Keep MoE/specialist work paused until a clean base model behaves reasonably.

Historical Markdown logs were moved out of the active `doc/*.md` set:

- `doc/archive/work_log_pre_clean_eval_leak_history.txt`
- `doc/archive/moe_plan_pre_clean_eval_leak_history.txt`
- `doc/archive/autoloop_handoff_shape1_base_clean_unique_bce.txt`
- `doc/archive/MEMORY_pre_clean_eval_leak_history.txt`
- `doc/archive/NOTES_early_history.txt`

Archive rule: do not read files under `archive/` directories unless the user
explicitly asks for archived history or audit material.

### Clean BCE Baseline Comparison

Completed comparison:

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_base_clean_unique_bce` | 0.2786 | 7.718 | 1.600 | 0.2661 | 0.2952 |

Conclusion: `shape1_base_clean_unique_bce` is worse than
`shape1_clean_split_bce_lineart004`.

Artifacts:

- `results/compare_clean_baselines_lineart004.png`
- `results/fixed_output_metrics_base_clean_lineart004_compare.csv`

### Moredupes Epoch020 Check

The `shape1_std15_clean_split_moredupes_bce` run is large:

- train list: `dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt`
- rows: 6,598
- eval list: `dataset/pairs_480/eval_fixed_clean_lineart004.txt`
- full run: 200 epochs, too slow for a first signal

Added and launched an epoch020 watcher:

- script: `experiments/watch_moredupes_epoch020_eval.sh`
- unit: `lineart-watch-moredupes-epoch020-eval.service`

The watcher waits for:

- `checkpoints/shape1_std15_clean_split_moredupes_bce/epoch020.pth`

Then it stops:

- `lineart-exp1-moredupes-bce.service`

And writes:

- `results/shape1_std15_clean_split_moredupes_bce_epoch020/`
- `results/fixed_output_metrics_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.csv`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `logs/shape1_std15_clean_split_moredupes_bce_epoch020.done`

### Results Cleanup

Moved old root one-off files from `results/` into:

- `results/archive/legacy_root_oneoffs/`

Moved old leaky/pre-clean-eval model outputs, comparisons, and metrics into:

- `results/archive/leaky_pre_clean_eval/`

Updated:

- `config/results_manifest.json`
- `results/CURRENT.md`
- `results/README.md`

### Extraction Rules Restored

Added `doc/EXTRACTION_RULES.md` so raw manuscript extraction rules stay visible
after archiving leak-era logs.

Key preserved rules:

- diagnose alignment before same-coordinate tiling
- dry-run plus CSV/QC before `--save`
- use region matching when correspondence is uncertain
- run crop-scale diagnostics when source scale/context changes
- audit train/eval leakage and missing files before training

### Next Actions

1. Wait for the epoch020 watcher to finish.
2. Inspect `results/compare_exp1_moredupes_epoch020_lineart004.png`.
3. Compare against `shape1_clean_split_bce_lineart004`.
4. If epoch020 is still poor, pivot to data reconstruction or loss/model changes
   instead of waiting for epoch200.
