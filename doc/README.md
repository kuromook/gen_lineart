# Documentation Map

Read `doc/CURRENT.md` first. It is the current operating state.

Unless the user explicitly asks for archived history, do not read files under
any `archive/` directory. Archive content is for audit only and may contain
leak-era assumptions.

## Current

- `doc/CURRENT.md`: authoritative current status and next actions.
- `doc/EXTRACTION_RULES.md`: current raw-data extraction procedure and gates.
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
3. latest dated section in `doc/work_log.md`
4. `doc/RESULTS.md` / `results/CURRENT.md`
5. archived text files only for historical audit
