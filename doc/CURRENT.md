# Current Project State

Updated: 2026-07-18 22:30 JST

This file is the first document to read. Older notes in this directory are
historical and may contain pre-leakage assumptions.

Do not read files under `archive/` directories unless the user explicitly asks
for archived history or audit material.

## Active Goal

Recover a non-leaky, clean evaluation path before continuing MoE or specialist
work.

The old `shape1` score is now treated as leaky and must not be used as an
adoption target. Use it only as a failure reference when comparing against
clean-split models.

## Current Running Work

- Running unit: `lineart-exp1-moredupes-bce.service`
- Experiment: `shape1_std15_clean_split_moredupes_bce`
- Train list: `dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt`
- Train rows: 6,598
- Eval list: `dataset/pairs_480/eval_fixed_clean_lineart004.txt`
- Intended full run: 200 epochs, but this is too slow for a first signal.

An epoch020 watcher is active:

- Unit: `lineart-watch-moredupes-epoch020-eval.service`
- Script: `experiments/watch_moredupes_epoch020_eval.sh`
- Behavior: wait for `checkpoints/shape1_std15_clean_split_moredupes_bce/epoch020.pth`,
  stop the long training unit, then run clean lineart004 inference, metrics,
  montage, and notification.

Expected epoch020 outputs:

- `results/shape1_std15_clean_split_moredupes_bce_epoch020/`
- `results/fixed_output_metrics_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.csv`
- `results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv`
- `results/compare_shape1_std15_clean_split_moredupes_bce_epoch020_lineart004.png`
- `results/compare_exp1_moredupes_epoch020_lineart004.png`
- `logs/shape1_std15_clean_split_moredupes_bce_epoch020.done`

## Latest Clean BCE Comparison

The `bce` vs `unique_bce` comparison is already complete.

| model | F1@2px | chamfer | ink_ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 | 0.2998 | 0.2974 |
| `shape1_base_clean_unique_bce` | 0.2786 | 7.718 | 1.600 | 0.2661 | 0.2952 |

Conclusion: `shape1_base_clean_unique_bce` is worse than
`shape1_clean_split_bce_lineart004` on the current clean lineart004 eval.

Primary artifacts:

- `results/compare_clean_baselines_lineart004.png`
- `results/fixed_output_metrics_base_clean_lineart004_compare.csv`
- archived handoff: `doc/archive/autoloop_handoff_shape1_base_clean_unique_bce.txt`

## Data Integrity Rules

Raw extraction procedure lives in `doc/EXTRACTION_RULES.md`.

All future training candidates must pass at least:

- no train/eval canonical tile overlap
- no train/eval exact rough or line hash overlap
- no train-internal exact duplicate unless the experiment explicitly studies duplicates
- no missing rough/line files, except for an explicit alternate line directory

Run:

```bash
./venv/bin/python tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists <train-list> \
  --eval-lists dataset/pairs_480/valid_test.txt dataset/pairs_480/eval_fixed_clean_lineart004.txt \
  --output-summary results/pair_dataset_integrity_summary_<experiment>.csv \
  --output-findings results/pair_dataset_integrity_findings_<experiment>.csv
```

## Current Interpretation

- `shape1` remains useful as a visual reference for what leakage made possible,
  but its high metrics are not a valid target.
- Clean BCE baselines are currently poor, so the next question is whether data
  quantity and controlled train-internal duplicates recover quality.
- The moredupes run intentionally allows train-internal duplicate/similar tiles,
  but keeps lineart004 eval clean.
- Do not start full MoE until clean base behavior is understood.

## Next Actions

1. Wait for the epoch020 watcher to finish.
2. Inspect `results/compare_exp1_moredupes_epoch020_lineart004.png`.
3. Compare metrics against `shape1_clean_split_bce_lineart004`.
4. If epoch020 is still poor, avoid waiting for epoch200 and pivot to data
   reconstruction or loss/model changes.
