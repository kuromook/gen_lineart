# Results Directory

`results/` is mostly generated experiment output and is ignored by default.

Read `results/CURRENT.md` first for the current clean-eval artifacts and the
known deprecated/noisy outputs.

Stable historical snapshots that were already tracked remain tracked. For new
experiments, commit only representative artifacts with `git add -f`, usually:

- final comparison montages
- metrics CSVs
- dataset/matching CSVs needed to reproduce a selected run
- small JSON summaries

Do not commit full inference folders, VLM panel dumps, or checkpoints here.
Use `doc/RESULTS.md` for the higher-level layout and cleanup policy.
