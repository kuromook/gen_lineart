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
  - branch: `foundation`
  - role: current integration branch and shared documentation
  - `doc/CURRENT.md` here is the authoritative project status document

Branch new tracks off `foundation`:

```bash
git worktree add ../lineart-<topic> -b <topic> foundation
```

## Who pushes (user instruction, 2026-10-18)

**A track pushes its own branch. The foundation does not push it for them.**
The foundation records and routes: when it notices a branch is behind, it
**sends a notice** naming the branch and the command, and the track runs it.
The foundation pushes only `foundation`, which is its own.

Precedent: on 2026-10-18 the foundation pushed `stroke-topology` itself, having
found it with no upstream at all and 42 commits living on one disk. The user's
instruction is that the notice, not the push, was the foundation's part. A
track opened without `-u` accumulates silently, so the notice carries the exact
form: `git push -u origin <branch>`.

This is the same line as checking for live sessions before touching a track
worktree -- a branch belongs to whoever is seated in it.

## Active Worktrees

Updated 2026-09-30. `doc/CURRENT.md`'s Track Ledger is the authoritative status
of each track; this table records only where each one physically lives, because
**that has stopped being uniform** — see the two exceptions below it.

| directory | branch | note |
|---|---|---|
| `lineart` | `foundation` | integration, shared docs, dataset pipeline |
| `lineart-stroke-selection` | `stroke-selection` | worktree of `lineart` |
| `lineart-pair-signal` | `pair-signal` | worktree of `lineart` |
| `lineart-stroke-grammar` | `stroke-grammar` | worktree of `lineart` |
| `lineart-aesthetic-judge` | `aesthetic-judge` | worktree of `lineart` |
| `lineart-stroke-topology` | `stroke-topology` | worktree of `lineart`, **Track K**, opened 2026-10-16 |
| `lineart-controlnet-sd15-refine` | `controlnet-sd15-refine` | worktree of `lineart` |
| `lineart-controlnet-sdxl-fidelity` | `controlnet-sdxl-fidelity` | worktree of `lineart` |
| `lineart-cleanup-refiner` | `cleanup-refiner` | worktree of `lineart`, dormant |
| `lineart-panel-generation` | `panel-generation` | **separate clone** of the same remote |
| `lineart-face-words` | `face-words` | worktree **of that clone**, not of `lineart` |

**Two things are not reachable the way this file used to imply:**

- `lineart-panel-generation` and `lineart-face-words` do not appear in
  `git worktree list` run from `lineart`, because the first is an independent
  clone and the second hangs off it. Anyone auditing tracks by listing
  worktrees will miss both. Discovered 2026-09-30.
- `lineart-halo-loss` and `lineart-router-moe` **no longer exist on disk.**
  Their registrations have already been pruned; their branches survive both
  locally and on `origin` at their 2026-07-20 tips, so nothing is lost, but
  there is no working tree to open. Do not cite those paths.

`lineart-controlnet-realpairs` is **not** a worktree — it was created as a
plain directory because it needed physical copies of the pair data and
checkpoints. Its branch `controlnet-realpairs` holds the track's history.
That track closed 2026-09-06; see `doc/track_proposal_20260906.md`.

Prefer a worktree of `lineart`. Use a plain directory only when a track
genuinely needs its own physical copy of large data, and say so in the track's
briefing. **If a track ends up as a separate clone, or as a worktree of one,
record it here** — otherwise it becomes invisible to anyone auditing from the
integration tree, which is exactly what happened with the two above.

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
back into `doc/work_log.md` on `foundation`, and update `doc/CURRENT.md` if the
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
- Before merging back to `foundation`, record: goal, changed files, commands
  run, metrics/montage paths, current recommendation.

## The branch was renamed `diffusion` -> `foundation` (2026-10-18)

User's observation: the name did not describe the thing. `diffusion` was the
name of a *direction* when this branch was cut from `diffusion-controlnet` on
2026-08-04 -- "look at each domain's generation quality instead of the
conversion task". That direction is closed (Tracks A and B: diffusion
generation does not beat the raw preprocessor), the branch became the common
foundation every track is cut from, and only the name stayed behind.

What the rename did and did not touch:

- `git branch -m`, then `git push -u origin foundation`. No history rewritten,
  no worktree moved: all ten track worktrees sit on their own branches and are
  unaffected.
- **`origin/diffusion` is left in place, frozen at `67d7bf7`**, so a live
  session that still has it configured does not break mid-run. It is not
  pushed to again. Delete it once the live tracks (J, K, H) have acknowledged.
  **All three have now acknowledged, each with its own audit** (2026-10-18):
  - **Track H**, 0 references to fix, `face-words` in sync at `43a6e88`.
  - **Track K**, 1 reference (this file's copy on `stroke-topology`), fixed,
    with a line naming `../lineart/doc/worktree_policy.md` (`8bbe924`) as the
    source of truth. Citations in its `work_log.md` left alone.
  - **Track J**, 0 references to fix, from a `git grep` over all tracked
    files: **no place resolves the branch name at run time** — `diffusion`
    appears in no `checkout` / `merge` / `rebase` / `push` / `worktree add` /
    `origin/...` form. The remaining hits are four kinds, each of which must
    *not* be changed: (1) citations of where a 2026-08 entry was written,
    (2) **the name of a technique** in Track J's prohibition 1
    ("do not try to draw fills with diffusion / ControlNet",
    `doc/initial_notice.md:28` and three places in its `work_log.md`) — this
    is a method, not a branch, and replacing it breaks the clause,
    (3) the separate branch `diffusion-controlnet`, (4) the file path
    `~/disk/checkpoint/Stable-diffusion/`.
  Its reply is `inbox/note_track_j_reply_foundation_rename_20261018.md`;
  it sat unread here for a day, which is the foundation's own miss.
  **That last condition is now met too** (2026-10-19). `lineart-panel-generation`
  is a second clone of the same GitHub repository, not a worktree of this one,
  and `lineart-face-words` is a worktree *of that clone*; the user pushed that
  clone, and it is in sync at `13e241b` with nothing uncommitted. Audited in
  both clones: `origin/diffusion` survives there only as a **stale
  remote-tracking ref** -- no branch has it as upstream (`panel-generation` ->
  `origin/panel-generation`, `face-words` -> `origin/face-words`), and the only
  tracked-file hit is a prose line in that clone's `work_log.md` recording that
  its `origin/diffusion` matched the frozen `67d7bf7`.
  So **nothing anywhere resolves that branch name at run time**, and deleting
  it is recoverable: `67d7bf7` is an ancestor of `foundation` and exists in
  every clone.
  **Done, 2026-10-19: the user ran the delete** (`git push origin --delete
  diffusion`, then `git remote prune origin` in both clones). The foundation
  session's permission mode refuses a destructive remote operation, so it
  recorded the audit and handed over the command rather than running it.
  Verified afterwards: `origin` carries no `diffusion` head, and both clones
  hold only `origin/diffusion-controlnet`, which is a different branch and was
  never in scope. **The rename is closed.** It remains recoverable --
  `67d7bf7` is an ancestor of `foundation` and still exists in every clone --
  but nothing references it.
- **Do not fix these references with a repo-wide `sed`** (warning from Track J,
  2026-10-18, asked to be carried to the other tracks). `experiments/*.sh` and
  `scripts/*.py` hold `BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/...` in
  a dozen or more places, so `sed s/diffusion/foundation/` over the repository
  points **every checkpoint path at a directory that does not exist** — and
  nothing says so until a job is launched. The rename needed exactly six
  operative lines changed, all in this file; it was never a string
  substitution.
- Track J asks, and the answer is yes: a track **does not** fix this file's old
  wording in its own tree. This file is the foundation's copy of record, and
  `8bbe924` rewrote 52 of its lines here; editing the same lines on a track
  only produces a conflict when that track catches up. The correct path is that
  the source of truth arrives with the merge. (Track K took the other route and
  edited its copy, which is why that one will conflict; it also added the
  pointer line, so the conflict resolves toward this file.)
- `main` was **not** used: from the 2026-07-29 split point (`00b2739`) `main`
  carries 13 commits this branch does not have and this branch carries 86, so
  making `main` the foundation is a separate reconciliation job, not a rename.
- Only this file's operative references changed. **Every other `diffusion`
  mention in `doc/` and `scripts/` is a citation of where a 2026-08 entry was
  written** ("doc/work_log.md, `diffusion` branch, 2026-08-08") and is left
  alone: that *was* the branch's name then, and rewriting it would make the
  citations wrong.
- GitHub's own default branch could not be checked from here (`gh` is not
  authenticated). If the repository's default is still `diffusion`, switch it
  in the web UI; nothing local depends on it.
