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

- `doc/initial_notice.md` — the briefing. Current state, next moves, and
  operating rules only. Keep it short; do not let it accumulate history.
  The goal is that a fresh session in that track can resume work from this
  file alone, without reading the integration tree's `doc/work_log.md`.
- `doc/work_log.md` (the worktree's own copy) — the chronological experiment
  log. Everything time-ordered goes here.

This split exists because on `controlnet-realpairs` the briefing doubled as the
work log and grew past 900 lines, at which point it stopped being readable.
Start every new track with both files.

The briefing lives in `doc/`, not `inbox/`, as of 2026-09-06. It was
originally placed in `inbox/` by analogy with the extraction-tool
correspondence, but `inbox/` and `outbox/` are `.gitignore`d — which left the
one document the whole track pattern depends on unversioned and lost whenever
a track folder was cleaned up. `inbox/`/`outbox/` are for correspondence with
the external extraction tool only; project documents go in `doc/`.

Avoid concurrent edits to the integration tree's `doc/work_log.md` from
multiple worktrees. When a branch produces a meaningful result, summarize it
back into `doc/work_log.md` on `diffusion`, and update `doc/CURRENT.md` if the
result changes project-level state or direction.

A track that closes writes a proposal for its successors into the integration
tree's `doc/` (not `inbox/`/`outbox/`, which carry correspondence with the
external extraction tool).

## Schedule Rules

Updated 2026-09-06. This supersedes the earlier rule that Monday-Thursday
daytime was reserved for another project and that long jobs belonged to
Thursday/Friday/Saturday nights.

The user is away from the room for most of the week. Monday through Thursday is
therefore the **preferred window for long GPU work**, run as one continuous
multi-day batch rather than a series of overnight runs.

- A job that needs several days of wall-clock time is acceptable. Plan around
  VRAM, checkpointing, and crash recovery, not around finishing quickly --
  runtime alone is not a reason to reject an approach or to downscale it.
- Launch fully detached (`nohup ... & disown`, verify `PPID=1`) so nothing
  depends on the session staying open, and checkpoint often enough that a crash
  on day three does not cost the whole run.
- Smoke-test first. A multi-day batch that fails on a bad argument is the
  expensive failure mode here.
- Weekends are the window for interactive work: review, analysis, short
  experiments, and deciding what the next long batch should be.
- Send only the final completion notification unless explicitly requested.

## Commit Rules

- Commit focused branch work in its own worktree.
- Push branch commits before switching context for long-running experiments.
- Keep generated large artifacts out of commits unless explicitly needed
  (`data/`, `checkpoints/`, and raw zips are gitignored).
- Before merging back to `diffusion`, record: goal, changed files, commands
  run, metrics/montage paths, current recommendation.
