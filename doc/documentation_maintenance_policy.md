# Documentation Maintenance Policy

Updated: 2026-07-25

Use this policy when Markdown context becomes too large to read efficiently.

## Work Log Size Gate

`doc/work_log.md` is a chronological handoff log, not the long-term home for
all project knowledge.

When `doc/work_log.md` exceeds 5,000 lines, do documentation maintenance before
or during the next broad `doc/*.md` context read.

Check with:

```bash
wc -l doc/work_log.md
```

Current threshold:

- below 5,000 lines: normal reads are acceptable
- above 5,000 lines: summarize, extract, and archive before relying on a full
  read

## Maintenance Actions

When the size gate is crossed:

1. Extract reusable knowledge from recent work-log sections into focused docs.
2. Keep only current state, recent decisions, and active next actions in
   `doc/work_log.md`.
3. Move old chronological detail into `doc/archive/` if it is needed only for
   audit.
4. Update `doc/CURRENT.md` with the active goal and immediate next actions.
5. Update `doc/README.md` if a new focused document is added.

## Where To Move Knowledge

Use focused docs instead of adding more long sections to `work_log.md`.
Preprocess (raw page/panel/pair extraction) docs live under
`doc/preprocess/*.md`, separate from architecture/research docs at
`doc/*.md`, so a plain `read doc/*.md` stays compact -- see "Layout" in
`doc/README.md`. Put new preprocess docs under `doc/preprocess/`, and new
architecture/research docs directly under `doc/`.

- extraction procedure and gates:
  - `doc/preprocess/EXTRACTION_RULES.md`
- dataset-specific raw extraction knowledge:
  - `doc/preprocess/raw_dataset_extraction_knowledge.md`
- region extraction policy:
  - `doc/preprocess/region_dataset_extraction_policy.md`
  - `doc/preprocess/region_materialization_policy.md`
- architecture/math/code per model, plus metrics glossary:
  - `doc/architecture_decisions.md`
- model direction summaries:
  - `doc/model_directions.md`
- results layout:
  - `doc/RESULTS.md`
- branch/worktree operating rules:
  - `doc/worktree_policy.md`

If no focused doc exists, create one with a narrow name and add it to
`doc/README.md`.

## Work Log Compaction Rule

When compacting `doc/work_log.md`, preserve:

- latest active run state
- latest successful artifacts
- current recommended model/data path
- open blockers and next actions
- links to extracted focused docs

Do not preserve full command transcripts in the active work log once they are
no longer needed for immediate recovery. Keep them in archive only when audit or
reproduction requires them.

## Reading Rule

For future broad Markdown reads:

1. Read `doc/README.md`.
2. Read `doc/CURRENT.md`.
3. Check `wc -l doc/work_log.md`.
4. If `work_log.md` is above 5,000 lines, read only the latest relevant tail
   plus focused docs unless the user explicitly asks for full history.
