# Results Layout

`results/` still contains legacy fixed paths used by scripts. Do not move
those paths blindly; many comparison and evaluation scripts reference them
directly.

Use `config/results_manifest.json` as the current index. It categorizes files into:

- `comparisons`: top-level `compare_*.png` montages.
- `inference_outputs`: model output folders such as `shape1/`, `kurip_clean540/`, `routed/`.
- `qc_and_dataset_build`: dataset matching, tile CSVs, and QC montages.
- `metrics_and_gate`: metrics CSVs and dataset-gate reports.
- `legacy_loose_outputs`: old one-off root-level outputs.
- `other`: uncategorized files that need manual review before moving.

Current recommended outputs:

- Default model: `results/shape1/`
- Kurip specialist: `results/kurip_clean540/`
- Routed fixed samples: `results/routed/`
- Routed kurip samples: `results/routed_kurip_samples/`
- Main routing comparisons:
  - `results/compare_routed_vs_shape1.png`
  - `results/compare_routed_kurip_vs_shape1.png`
- Kurip specialist comparison:
  - `results/compare_kurip_clean540_vs_shape1.png`
  - `results/compare_kurip_clean540_train_vs_shape1.png`

Deprecated or non-recommended experiment outputs should remain available for
audit, but should not be used by routing:

- `results/kurip/` and `checkpoints/kurip/`: dirty kurip fine-tune; noisy.
- `results/kurip_clean540_noac/`: did not beat `kurip_clean540`.
- `results/kurip_balanced_dense/`: did not beat `kurip_clean540`.

Safe cleanup policy:

1. Update `config/results_manifest.json` before moving files.
2. Search code references with `rg "results/" --glob '!results/**'`.
3. Move only unreferenced one-off files into `results/archive/`.
4. Keep stable script-facing paths intact unless the scripts are updated in the same change.
5. Re-run the relevant comparison or evaluation script after any path change.
