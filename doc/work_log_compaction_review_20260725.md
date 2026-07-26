# Work Log Compaction Review - 2026-07-25

Purpose: record the human-reviewed cleanup of `doc/work_log.md` before it grows
past the 5,000-line maintenance threshold.

Status: accepted by user and applied.

Archived detailed work log:

- `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`

Current compact active work log:

- `doc/work_log.md`

Pre-compaction line count:

- `doc/work_log.md`: about 4,506 lines after the latest ako5 review entries

Post-compaction line count:

- `doc/work_log.md`: about 263 lines

Policy:

- `doc/documentation_maintenance_policy.md`

## Proposed Human-Check Workflow

1. Review this proposal.
2. Mark each proposed action as accept / revise / hold.
3. Extract or update focused docs for accepted knowledge.
4. Replace old `work_log.md` sections with short pointers only after review.
5. Move old chronological detail to `doc/archive/` only after the focused docs
   contain the reusable knowledge.

## Proposed Keep In Active Work Log

Keep the latest active recovery context and current data work near the end of
`doc/work_log.md`:

- `2026-07-25: hamlabi Filtered398 UNet480 e40 Review`
- `2026-07-25: Raw Dataset Zip Storage`
- `2026-07-25: ako5ver2 Region Pair Extraction`
- `2026-07-25: ako5ver2 hamlabi-style Variable Region Extraction`
- `2026-07-25: ako5ver2 User Review Flags`

Reason:

- These are the most likely sections needed for immediate continuation.

## Proposed Extracted Knowledge Already Covered

These topics now have focused docs and should not need full historical detail
in active `work_log.md`:

- raw extraction procedure:
  - `doc/EXTRACTION_RULES.md`
- raw dataset storage:
  - `doc/raw_dataset_storage_policy.md`
- region extraction policy:
  - `doc/region_dataset_extraction_policy.md`
- region materialization:
  - `doc/region_materialization_policy.md`
- region search loop:
  - `doc/region_search_loop.md`
- worktree policy:
  - `doc/worktree_policy.md`
- model direction summary:
  - `doc/model_directions.md`
- dataset-specific raw extraction knowledge:
  - `doc/raw_dataset_extraction_knowledge.md`
- documentation maintenance:
  - `doc/documentation_maintenance_policy.md`

Review question:

- Are these focused docs sufficient before older detailed sections are archived?

## Proposed Archive Candidates

### 2026-07-18 To 2026-07-19 Clean Baseline Reset

Candidate sections:

- `Reset Context After Leakage`
- `Clean BCE Baseline Comparison`
- `Moredupes Epoch020 Check`
- `Moredupes Epoch020 Result`
- `Mild Duplicate Low-LR Fine-Tune`
- `Milddup800 Result And Model Survey`

Proposed action:

- keep a short summary in active work log
- archive detailed command/history sections

Reason:

- The non-leaky rule is already captured in `CURRENT.md` and
  `EXTRACTION_RULES.md`.
- Detailed early baseline experiments are historical context.

Human check:

- confirm whether any early clean-baseline metrics still need to remain inline.

### 2026-07-20 Model/Halo Survey History

Candidate sections:

- halo loss result
- atari halo diagnosis
- aux strength/loss halo diagnosis
- rough cleanup sweeps
- cleaned-rough model surveys
- ResNet-GAN / 2ch refiner / cleanup / MSGAN / width / structure surveys
- Lucy / mask deep surveys
- router/MoE oracle and initial line-field result

Proposed action:

- keep only a compact model-state summary in active work log
- ensure `doc/model_directions.md` and any future model summary doc contain the
  reusable conclusions
- archive detailed run-by-run history

Reason:

- The experiment chain is long and mostly resolved into current model
  interpretation:
  - Lucy mild/thin and dog hints are useful candidates
  - cleanup/msgan family is promising but artifact-prone
  - router oracle is useful as an upper bound
  - initial line-field overproduces ink

Human check:

- decide whether to create a separate `doc/model_results_summary.md` before
  archiving these sections.

### 2026-07-24 Bad-Rough Ako / Lucy-Thin Cleanup History

Candidate sections:

- ako5 uninterpretable rough origin check
- hard exclusion lists
- broader bad-rough audit and reformed lists
- bad-rough retrain / ink-width / Lucy-thin / threshold-loss surveys
- Lucy-thin cleanup tuning pause

Proposed action:

- keep the high-level note in
  `doc/badrough_lucy_thin_threshold_notes.md`
- move source/dataset-specific findings into
  `doc/raw_dataset_extraction_knowledge.md` if not already captured
- archive detailed run history

Reason:

- The current decision is already summarized: threshold-aware Lucy-thin is a
  useful reference, not production; preprocessing and clean data expansion are
  next.

Human check:

- confirm whether all bad-rough exclusion lists are discoverable from current
  files before archiving details.

### 2026-07-24 hamlabi Region Workflow History

Candidate sections:

- hamlabi raw import
- hamlabi region matching branch start
- parent/child review materialization
- Codex VLM final review
- loader/materialization policy
- final review v2 correction
- training trials
- pair expansion policy
- region search loop
- valid mask / post-align trials
- auto-review 398-pair training start

Proposed action:

- keep the final 398-pair state and latest review result inline
- move reusable workflow details to existing region docs
- archive exploratory command history

Reason:

- The policy docs now define the intended hamlabi-style workflow.

Human check:

- confirm whether the exact command transcript for hamlabi should stay in
  active log until ako5 training is complete.

## Focused Docs Created In Trial Run

Created after this review proposal was drafted:

- `doc/dataset_status.md`
  - current usable manifests, review targets, held-out rows, and dataset next
    actions
- `doc/model_results_summary.md`
  - compact current model-family conclusions and checkpoint/status references

These files should be reviewed before deleting or archiving the corresponding
long `work_log.md` sections.

Review question:

- Are `doc/dataset_status.md` and `doc/model_results_summary.md` accurate
  enough to let older run-by-run details move to archive after user approval?

## Approval List And Applied Result

Active `work_log.md` compaction has been performed after user approval.

| block | user decision | applied result |
|---|---|---|
| 2026-07-18 to 2026-07-19 clean baseline reset | accepted | compact summary kept; details archived |
| 2026-07-20 model and halo survey history | accepted | `model_results_summary.md` used; details archived |
| 2026-07-24 bad-rough / Lucy-thin history | accepted | badrough note plus focused docs used; details archived |
| 2026-07-24 hamlabi exploratory workflow | accepted | final 398 state kept; detailed commands archived |
| 2026-07-25 ako5ver2 current work | accepted active | kept in active `work_log.md` |

## Proposed Active Work Log Shape After Compaction

Target:

- keep active `doc/work_log.md` under roughly 1,500 to 2,000 lines

Suggested shape:

```text
# Work Log - Clean Rebuild

## Current Pointers

- current state: doc/CURRENT.md
- extraction rules: doc/EXTRACTION_RULES.md
- dataset-specific extraction knowledge: doc/raw_dataset_extraction_knowledge.md
- model directions: doc/model_directions.md
- archived detailed history: doc/archive/<file>.md

## Recent Active History

<only latest accepted/current sections>
```

## Applied Result

- active `doc/work_log.md` is now a compact pointer/current-history file
- full detail is preserved in
  `doc/archive/work_log_clean_rebuild_detail_20260718_20260725.md`
- future `doc/*.md` reads should start with `doc/README.md` and
  `doc/CURRENT.md`, then use focused docs before archived detail
