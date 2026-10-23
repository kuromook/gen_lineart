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

**The named exception, and the reason it is narrow (2026-10-19, user
approved).** The foundation committed and pushed two trees itself:
`cleanup-refiner` (`4519bef`) and `controlnet-sd15-refine` (`cea3cf1`). Both
are **dormant or closed with nobody seated in them**, which is exactly the
condition the rule protects against -- it exists to keep the foundation out of
a *live* tree, not to leave orphaned work uncommitted forever. When a track has
a live session, the notice still goes to it and the track still pushes.

## Active Worktrees

Updated 2026-10-24 (three rows were missing: Tracks J, L and M).
`doc/CURRENT.md`'s Track Ledger is the authoritative status
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
| `lineart-solid-fill` | `solid-fill` | worktree of `lineart`, **Track J**, opened 2026-10-20 |
| `lineart-fill-region` | `fill-region` | worktree of `lineart`, **Track L**, opened 2026-10-23 |
| `lineart-colour-manuscripts` | `colour-manuscripts` | worktree of `lineart`, **Track M**, opened 2026-10-24 |
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
  carries 13 commits this branch does not have and this branch carries 86 (114
  by 2026-10-19). That framing made it sound like a merge job, and it is not:
  **all 13 are the paper scout's**, and none of them touch project code. See
  "What `main` actually is" below.
- Only this file's operative references changed. **Every other `diffusion`
  mention in `doc/` and `scripts/` is a citation of where a 2026-08 entry was
  written** ("doc/work_log.md, `diffusion` branch, 2026-08-08") and is left
  alone: that *was* the branch's name then, and rewriting it would make the
  citations wrong.
- GitHub's own default branch **is readable from here after all**, without
  `gh` and without authentication: `git ls-remote --symref origin HEAD` prints
  `ref: refs/heads/main`. The earlier note in this file said it could not be
  checked because `gh auth status` is logged out, which was the wrong command,
  not a missing capability. The default is `main`, not `diffusion`, so the
  rename never touched it; what to do about it is the next section.

## What `main` actually is, and the decision about it (2026-10-19)

`main` is **not** this project's trunk and never became one. From the
2026-07-29 split point its only unique content is one directory,
`tools/research/cross_domain_paper_scout/` (9 files, 782 lines); its 13
commits are the scout's setup plus one report per run. **No project change has
ever landed on it.** Meanwhile `foundation` is 114 commits past that point.

How the scout runs, which is why `main` must keep existing:

- It runs on an **always-on machine, not this one** (there is no crontab
  here), from a **blobless sparse checkout** of just that one directory, which
  does `git checkout main`. Weekly, `0 9 * * 1`.
  **Located 2026-10-19: the host is `nuc` at 192.168.1.8** (user `sh1`, up 20
  days), clone at **`/home/sh1/da/gen_lineart`**, sparse-checkout list exactly
  `tools/research/cross_domain_paper_scout`, branch `main` tracking
  `origin/main`, clean. Its crontab line is
  `0 9 * * 1 /home/sh1/da/gen_lineart/tools/research/cross_domain_paper_scout/scout.sh >> .../cron.log 2>&1`.
  Key-based SSH from this machine was set up by the user the same day
  (`ssh-copy-id`), so the foundation can audit it directly; before that it was
  unreachable (`Permission denied (publickey,password)`).
- **The scout's own open question is now closed**: its README recorded that the
  commit/pull/push block had never been exercised end to end under cron, only
  by hand. The 2026-10-19 run did it -- `cron.log` shows
  `47001e8..c6b236e main -> main` and `run end, exit=0` -- so that block is
  tested.
- Tidied on `nuc` the same day: `git fetch --prune` dropped the ghost
  `origin/diffusion`, `git remote set-head origin -a` re-pointed `origin/HEAD`
  from `main` to `foundation`, and a `git grep` for any run-time branch
  resolution returns **0 hits**. `branch.main.merge` stays `refs/heads/main`,
  which is what the cron depends on. **Nothing there needed renaming**: the
  only local branch is `main`, and `main` must keep its name.
- Each run calls `claude -p` with `--allowedTools "WebSearch,WebFetch,Read,Edit"`
  (no shell, no git for the agent itself), appends to `candidates.md` or
  `candidates_diffusion.md`, and then the *script* commits and does
  `git pull --rebase && git push`.
- Two tracks share the slot by ISO-week parity: odd week topology, even week
  diffusion.

**Decision (user, 2026-10-19): switch GitHub's default branch to `foundation`
and leave `main` alone as the scout's lane. Done the same day** -- the user
changed it in the web UI (the setting is on Settings -> General, not Settings
-> Branches, which now holds only protection rules), and
`git ls-remote --symref origin HEAD` confirms `ref: refs/heads/foundation`.
A new clone now lands on the current state instead of the 2026-07-29 snapshot
plus a scout folder, and the root `README.md` that GitHub shows is the one
rewritten the same day. `origin/HEAD` was also re-pointed locally in both
clones (`git remote set-head origin -a`); it had never been set in either, so
nothing was relying on it. The scout is unaffected: its sparse checkout names
`main` explicitly, and that branch still exists. `main` is **not** renamed to something more
honest (`paper-scout`) because the always-on machine's sparse checkout tracks
`main` by name, and the weekly push would fail silently until someone logs
into that machine to re-point it. `main` is **not** turned into the real trunk
either: that would put an unattended `pull --rebase && push` on the trunk, and
the scout's own README records that its push block has never been exercised
end to end.

**Superseded later the same day (2026-10-19): `main` was renamed to
`paper-scout`.** The reason for not renaming above was that the always-on
machine tracks `main` by name and someone would have to log into it to
re-point it; the rename was done *from* that machine (`nuc`,
`/home/sh1/da/gen_lineart`), so that cost was paid in the same step. What was
done there: `git branch -m main paper-scout`, the scout README's setup line
changed to `git checkout paper-scout` (commit `8a3b044`), `git push -u origin
paper-scout`, then -- after checking `origin/main` was an ancestor of
`paper-scout` -- `git push origin --delete main`. `scout.sh` needs no change:
it runs a bare `git pull --rebase && git push`, which follows the new
upstream. The crontab path is unchanged. The first cron push to
`paper-scout` is the 2026-10-26 run; check `cron.log` on `nuc` for it.
**`origin/main` no longer exists**; on this machine run `git fetch --prune`
and delete or rename any local `main`.
**Verified from the foundation 2026-10-19, so the rename is not waiting on the
2026-10-26 run to be trusted.** On `nuc`: local branch `paper-scout` tracking
`origin/paper-scout`, tree clean, the README's setup line reads
`git checkout paper-scout`, `scout.sh` still at its crontab path. The git half
of the weekly job was then exercised without running the agent --
`git pull --rebase` clean, `git push --dry-run` returns *Everything
up-to-date*, and `@{u}` resolves to `origin/paper-scout` -- which was the one
thing the rename could have broken silently. The agent half had already run
today. In this clone the stale local `main` (at `a41832a`, tracking a ref that
no longer exists) was **renamed rather than deleted**: it is now `paper-scout`
fast-forwarded to `origin/paper-scout`, after checking `a41832a` is an
ancestor of it, so nothing was discarded to tidy a name.

**Triaged 2026-10-19 at the user's request** -- see "The paper scout's nine
reports, triaged against Track K's spec" under `doc/CURRENT.md`'s *Approaches
Never Taken Up*. Until that day, nine reports had accumulated since 2026-08-10
(4 topology, 5 diffusion) and **no document on `foundation` cited the scout at
all**. A tenth arrived from the 2026-10-19 run, hours after the triage, and is
logged there too. The
topology lane's subjects are road-network extraction from satellite imagery,
crack detection in concrete and composites, fiber tracking, and PCB
trace/wire-bond inspection -- all of them the problem of joining broken line
structure, which is Track K's problem. Local `main` also sits 12 commits
behind `origin/main`, so the reports are not even on this disk by default:
read them with `git show origin/paper-scout:tools/research/cross_domain_paper_scout/candidates.md`
after a fetch (`origin/main` before the rename above).
