# Worktree Operation Policy

Updated: 2026-07-20

## Purpose

Current line-art work has multiple active directions that can conflict if they
share one branch:

- halo suppression / faint gray penalty
- router / MoE expert selection
- line-field correction
- rough-line agreement and data selection

Use git worktrees to keep code changes isolated while preserving a common
project context.

## Recommended Worktrees

Keep the current repository as the integration / main working tree:

- `/home/sh1/deepl/lineart`
  - branch: `MoE`
  - role: current integration branch and shared documentation

Create focused worktrees as needed:

```bash
git worktree add ../lineart-halo-loss -b halo-loss MoE
git worktree add ../lineart-router-moe -b router-moe MoE
git worktree add ../lineart-line-field -b line-field MoE
git worktree add ../lineart-data-agreement -b data-agreement MoE
```

Initial priority:

```bash
git worktree add ../lineart-halo-loss -b halo-loss MoE
git worktree add ../lineart-router-moe -b router-moe MoE
```

Add `line-field` and `data-agreement` worktrees when those directions become
active again.

## Branch Roles

- `halo-loss`
  - halo-band loss
  - faint gray penalty
  - dark/white output distribution constraints
- `router-moe`
  - oracle-to-router work
  - expert feature extraction
  - expert selection policy and routed inference
- `line-field`
  - centerline / offset / width field models
  - probabilistic line correction targets
- `data-agreement`
  - rough-line agreement scoring
  - high/mid/low split design
  - source-controlled data selection
- `MoE`
  - integration branch
  - stable shared experiment runners
  - project-level docs and summaries

## Artifact Rules

Do not rely on worktree-local `results/`, `checkpoints/`, or `logs/` names to
separate experiments. Always use explicit tags in runner output names.

Examples:

- `halo_loss_e2_*`
- `router_moe_*`
- `linefield_*`
- `agreement_*`

This keeps artifacts searchable and reduces confusion if multiple worktrees
write under the same project root.

## Documentation Rules

Avoid frequent concurrent edits to `doc/work_log.md` from multiple worktrees.

Use direction-specific logs while a branch is active:

- `doc/work_log_halo_loss.md`
- `doc/work_log_router_moe.md`
- `doc/work_log_line_field.md`
- `doc/work_log_data_agreement.md`

When a branch produces a meaningful result, summarize it back into
`doc/work_log.md` on the `MoE` integration branch.

## Commit Rules

- Commit focused branch work in its own worktree.
- Push branch commits before switching context for long-running experiments.
- Keep generated large artifacts out of commits unless explicitly needed.
- Before merging back to `MoE`, record:
  - goal
  - changed files
  - commands run
  - metrics/montage paths
  - current recommendation

## Current Next Step

The next active direction is halo suppression. Start with `halo-loss` worktree
and test direct output-distribution constraints:

- halo-band loss
- faint gray penalty
- high-agreement plus halo loss, if useful
- compare against `lucy_mild`, `lucy_thin`, and `agreement_halo_e2_high_agreement`
