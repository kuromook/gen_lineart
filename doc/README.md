# Documentation Map

Read `doc/CURRENT.md` first. It is the current operating state.

Unless the user explicitly asks for archived history, do not read files under
any `archive/` directory. Archive content is for audit only and may contain
leak-era assumptions.

## Current

- `doc/CURRENT.md`: authoritative current status and next actions.
- `doc/EXTRACTION_RULES.md`: current raw-data extraction procedure and gates.
- `doc/raw_dataset_extraction_knowledge.md`: dataset-specific raw extraction
  observations, exceptions, layer differences, and review knowledge.
- `doc/region_dataset_extraction_policy.md`: region-matched variable-aspect
  extraction rules, including the source-scale band and tile-score policy.
- `doc/region_materialization_policy.md`: manifest materialization policy,
  including native-scale re-materialization.
- `doc/documentation_maintenance_policy.md`: when and how to compact/split
  large Markdown context, especially `doc/work_log.md`.
- `doc/work_log_compaction_review_20260725.md`: current human-review proposal
  for trimming and archiving old `work_log.md` sections.
- `doc/dataset_status.md`: current usable manifests, review targets, held-out
  rows, and dataset next actions.
- `doc/model_results_summary.md`: compact current model-family conclusions and
  checkpoint/status references.
- `doc/RESULTS.md`: results layout and cleanup policy.
- `doc/work_log.md`: short clean-rebuild log from the leakage reset onward.

## Historical Or Noisy

- `doc/archive/*.txt`: old leak-era or pre-clean-eval history. Read only for
  audit, not for current decisions.
- `results/archive/**`: old generated artifacts. Do not inspect unless the user
  explicitly asks for archived results.

When documents conflict, prefer:

1. `doc/CURRENT.md`
2. `doc/EXTRACTION_RULES.md` for raw pair extraction
3. `doc/raw_dataset_extraction_knowledge.md` for dataset-specific extraction
   exceptions and review knowledge
4. `doc/documentation_maintenance_policy.md` when Markdown context is large
5. latest dated section in `doc/work_log.md`
6. `doc/RESULTS.md` / `results/CURRENT.md`
7. archived text files only for historical audit
