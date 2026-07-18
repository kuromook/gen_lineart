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

### Moredupes Epoch020 Result

The scratch `shape1_std15_clean_split_moredupes_bce` run was stopped at
epoch020 and evaluated on clean lineart004-only samples.

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_std15_clean_split_moredupes_bce_epoch020` | 0.1853 | 8.696 | 0.301 | 0.3857 | 0.1257 |

Conclusion: scratch training on the large duplicated list is not a useful
overnight direction. It under-produces ink and is worse than the clean baseline.

Artifacts:

- `results/compare_exp1_moredupes_epoch020_clean_lineart004.png`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_clean_lineart004_compare.csv`

## 2026-07-19

### Mild Duplicate Low-LR Fine-Tune

Started one conservative overnight run:

- experiment: `shape1_clean_split_bce_milddup800_ft10_lr1e5`
- unit: `lineart-milddup800-ft.service`
- log: `logs/train_shape1_clean_split_bce_milddup800_ft10_lr1e5.service.log`
- resume: `checkpoints/shape1_clean_split_bce/best.pth`
- train list: `dataset/pairs_480/valid_train_milddup800_clean.txt`
- eval list: `dataset/pairs_480/eval_clean_lineart004_8.txt`
- epochs: 10
- lr: `1e-5`
- loss weights: BCE only, `pos_weight=5.0`

The train list is generated from:

- seed: `dataset/pairs_480/valid_train_base_clean_unique.txt` (469 rows)
- pool: `dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt` (6,598 rows)
- target rows: 800
- max per canonical tile: 2
- exact duplicates: allowed intentionally for this experiment

Dry-run/build verification:

- rows: 800
- canonicals: 633
- duplicate canonicals: 167

Committed and pushed:

- `2eb995c Add mild duplicate fine-tune experiment`

Expected outputs when complete:

- `logs/shape1_clean_split_bce_milddup800_ft10_lr1e5.done`
- `results/fixed_output_metrics_shape1_clean_split_bce_milddup800_ft10_lr1e5_clean_lineart004.csv`
- `results/fixed_output_metrics_shape1_clean_split_bce_milddup800_ft10_lr1e5_compare.csv`
- `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5_clean_lineart004.png`
- `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5_vs_clean_split_bce.png`

### Next Actions

1. Check whether `lineart-milddup800-ft.service` completed.
2. Inspect the compare montage against `shape1_clean_split_bce_lineart004`.
3. Prefer the milddup fine-tune only if it improves F1/chamfer without obvious
   visual degradation or over-thick ink.
4. If it fails, keep `shape1_clean_split_bce_lineart004` as the current clean
   baseline and pivot to better extraction/data rules rather than larger
   duplicated training.
