# Worktree Operation Policy

Updated: 2026-09-06 (integration branch, active track list, and per-track
documentation convention brought up to date; the 2026-07-20 original named
`MoE` as the integration branch)

## Purpose

Line-art work has multiple active directions that can conflict if they share
one branch. Use git worktrees to keep code changes isolated while preserving a
common project context.

## Integration Tree

Keep the current repository as the integration / main working tree:

- `/home/sh1/deepl/lineart`
  - branch: `diffusion`
  - role: current integration branch and shared documentation
  - `doc/CURRENT.md` here is the authoritative project status document

Branch new tracks off `diffusion`:

```bash
git worktree add ../lineart-<topic> -b <topic> diffusion
```

## Active Worktrees

| directory | branch | role |
|---|---|---|
| `lineart` | `diffusion` | integration, shared docs, dataset pipeline |
| `lineart-controlnet-sd15-refine` | `controlnet-sd15-refine` | SD1.5 ControlNet refinement from the cs3.5 best config |
| `lineart-controlnet-sdxl-fidelity` | `controlnet-sdxl-fidelity` | SDXL condition fidelity (1024 re-baseline first) |
| `lineart-cleanup-refiner` | `cleanup-refiner` | dormant |
| `lineart-halo-loss` | `halo-loss` | dormant |
| `lineart-router-moe` | `router-moe` | dormant |

`lineart-controlnet-realpairs` is **not** a worktree — it was created as a
plain directory because it needed physical copies of the pair data and
checkpoints. Its branch `controlnet-realpairs` holds the track's history.
That track closed 2026-09-06; see `doc/track_proposal_20260906.md`.

Prefer a worktree. Use a plain directory only when a track genuinely needs its
own physical copy of large data, and say so in the track's briefing.

## Artifact Rules

Do not rely on worktree-local `results/`, `checkpoints/`, or `logs/` names to
separate experiments. Always use explicit tags in runner output names
(`halo_loss_e2_*`, `router_moe_*`, `controlnet_lora_*`, ...). This keeps
artifacts searchable and reduces confusion if multiple worktrees write under
the same project root.

## Documentation Rules

Each track keeps two documents, and only two:

- `inbox/initial_notice.md` — the briefing. Current state, next moves, and
  operating rules only. Keep it short; do not let it accumulate history.
- `doc/work_log.md` (the worktree's own copy) — the chronological experiment
  log. Everything time-ordered goes here.

This split exists because on `controlnet-realpairs` the briefing doubled as the
work log and grew past 900 lines, at which point it stopped being readable.
Start every new track with both files.

Avoid concurrent edits to the integration tree's `doc/work_log.md` from
multiple worktrees. When a branch produces a meaningful result, summarize it
back into `doc/work_log.md` on `diffusion`, and update `doc/CURRENT.md` if the
result changes project-level state or direction.

A track that closes writes a proposal for its successors into the integration
tree's `doc/` (not `inbox/`/`outbox/`, which carry correspondence with the
external extraction tool).

## Schedule Rules

This PC is used for another project from Monday through Thursday daytime.
Plan line-art GPU-heavy jobs around that constraint.

- Daytime work: implementation, analysis, short smoke tests, documentation.
- Long GPU jobs (training runs, multi-model sweeps): treat as night-batch work.
  Schedule primarily for Thursday, Friday, and Saturday nights, launch before
  sleep, inspect the next morning, and send only the final completion
  notification unless explicitly requested.
- Avoid starting long GPU jobs during Monday-Thursday daytime.

## Commit Rules

- Commit focused branch work in its own worktree.
- Push branch commits before switching context for long-running experiments.
- Keep generated large artifacts out of commits unless explicitly needed
  (`data/`, `checkpoints/`, and raw zips are gitignored).
- Before merging back to `diffusion`, record: goal, changed files, commands
  run, metrics/montage paths, current recommendation.
