# Results Layout

`results/` contains current clean-eval artifacts, extraction QC, and archived
historical output. Many old comparison scripts still reference legacy paths, so
archived paths should only be restored intentionally for audit.

Use `config/results_manifest.json` as the current index. It categorizes files into:

- `comparisons`: top-level `compare_*.png` montages.
- `inference_outputs`: model output folders such as `shape1/`, `kurip_clean540/`, `routed/`.
- `qc_and_dataset_build`: dataset matching, tile CSVs, and QC montages.
- `metrics_and_gate`: metrics CSVs and dataset-gate reports.
- `legacy_loose_outputs`: old one-off root-level outputs.
- `other`: uncategorized files that need manual review before moving.

Current recommended outputs:

- Current clean-eval index: `results/CURRENT.md`
- Current clean BCE comparison:
  - `results/compare_clean_baselines_lineart004.png`
  - `results/fixed_output_metrics_base_clean_lineart004_compare.csv`
- Pending epoch020 moredupes comparison:
  - `results/compare_exp1_moredupes_epoch020_lineart004.png`
  - `results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv`

Deprecated or non-recommended experiment outputs should remain available for
audit, but should not be used by routing:

- `results/archive/leaky_pre_clean_eval/`: old leaky/pre-clean-eval model
  outputs, comparisons, and metrics.
- `results/archive/legacy_root_oneoffs/`: old root one-off outputs moved out of
  the top-level results namespace.
- `checkpoints/kurip/`: dirty kurip fine-tune checkpoint; noisy.

Safe cleanup policy:

1. Update `config/results_manifest.json` before moving files.
2. Search code references with `rg "results/" --glob '!results/**'`.
3. Move only unreferenced one-off files into `results/archive/`.
4. Keep stable script-facing paths intact unless the scripts are updated in the same change.
5. Re-run the relevant comparison or evaluation script after any path change.
