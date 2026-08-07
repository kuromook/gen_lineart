# Documentation Map

Read `doc/CURRENT.md` first. It is the current operating state.

Unless the user explicitly asks for archived history, do not read files under
any `archive/` directory. Archive content is for audit only and may contain
leak-era assumptions.

## Layout

`doc/*.md` (top level) holds the research side of the project: model
architecture, math, training recipes, evaluation metrics, and results —
i.e. the rough-to-line-art conversion work itself.

`doc/preprocess/*.md` holds the raw-page/panel/pair extraction pipeline
(raw manuscript -> koma panels -> region-matched pairs -> tiles). This is
a separate, lower-level concern from the architecture work, and is split
into its own directory so a plain `read doc/*.md` (non-recursive glob)
loads only the research-side context by default, without the bulk of the
preprocess knowledge base. Read `doc/preprocess/*.md` explicitly when
working on extraction/tiling/panel-matching.

## Current — Architecture / Research (doc/*.md)

- `doc/CURRENT.md`: authoritative current status and next actions.
- `doc/architecture_decisions.md`: per-architecture reference (purpose,
  math, code, visual review status, montage location) plus the evaluation
  metrics glossary. Kept in sync across `cleanup-refiner` and
  `diffusion-controlnet` branches.
- `doc/model_directions.md`: numbered "Direction N" survey status
  (which architecture families were tried, adopted, or ruled out).
- `doc/model_results_summary.md`: compact current model-family conclusions and
  checkpoint/status references.
- `doc/badrough_lucy_thin_threshold_notes.md`: bad-rough/lucy_thin
  threshold-tuning results, ako5/ako6-derived.
- `doc/diffusion_fidelity_budget_policy.md`: acceptance policy for
  domain-only LoRA generation quality (`diffusion` branch) -- how much
  deviation toward the base checkpoint's own habits is treated as within
  control vs. a real failure.
- `doc/worktree_policy.md`: which git worktree/branch each active
  direction (halo-loss, router-moe, cleanup-refiner, diffusion-controlnet)
  lives on.
- `doc/documentation_maintenance_policy.md`: when and how to compact/split
  large Markdown context, especially `doc/work_log.md`.
- `doc/work_log_compaction_review_20260725.md`: current human-review proposal
  for trimming and archiving old `work_log.md` sections.
- `doc/RESULTS.md`: results layout and cleanup policy.
- `doc/work_log.md`: short clean-rebuild log from the leakage reset onward.

## Current — Preprocess (doc/preprocess/*.md)

- `doc/preprocess/EXTRACTION_RULES.md`: current raw-data extraction procedure and gates.
- `doc/preprocess/raw_dataset_extraction_knowledge.md`: dataset-specific raw extraction
  observations, exceptions, layer differences, and review knowledge.
- `doc/preprocess/raw_dataset_storage_policy.md`: raw manuscript zip/storage
  conventions.
- `doc/preprocess/region_dataset_extraction_policy.md`: region-matched variable-aspect
  extraction rules, including the source-scale band and tile-score policy.
- `doc/preprocess/region_materialization_policy.md`: manifest materialization policy,
  including native-scale re-materialization.
- `doc/preprocess/region_search_loop.md`: region-search iteration loop notes.
- `doc/preprocess/dataset_status.md`: current usable manifests, review targets, held-out
  rows, and dataset next actions.

## Historical Or Noisy

- `doc/archive/*.txt`: old leak-era or pre-clean-eval history. Read only for
  audit, not for current decisions.
- `results/archive/**`: old generated artifacts. Do not inspect unless the user
  explicitly asks for archived results.

When documents conflict, prefer:

1. `doc/CURRENT.md`
2. `doc/architecture_decisions.md` for architecture/math/code and its
   metrics glossary
3. `doc/preprocess/EXTRACTION_RULES.md` for raw pair extraction
4. `doc/preprocess/raw_dataset_extraction_knowledge.md` for dataset-specific extraction
   exceptions and review knowledge
5. `doc/documentation_maintenance_policy.md` when Markdown context is large
6. latest dated section in `doc/work_log.md`
7. `doc/RESULTS.md` / `results/CURRENT.md`
8. archived text files only for historical audit
