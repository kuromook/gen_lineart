# Paper Scout

A slow background paper search for this project, running two independent
tracks that alternate weeks. Both are meant to accumulate over months and
be triaged in batches, not acted on immediately.

## Project aim these tracks serve (rewritten 2026-11-04)

Both prompts were rewritten on 2026-11-04 at the user's decision, relayed
by the foundation session (`doc/CURRENT.md` on `foundation`, Next
Actions, "論文の偵察の報告"). The project's aim is no longer
rough-to-line-art translation: per `CLAUDE.md` 第1条 (2026-10-30) it is to
work out what line drawing conveys and make that usable, with generation
as a means and human judgement setting the direction. The old prompts'
numbers (F1@2px 0.20 vs 0.42, ~6x ink, clDice's "genuine improvement")
predate the current bars (the preprocessor lower bound, the bsds_f1 floor
of 0.1941), so the prompts now give them with their date and say they
have not been re-checked.

## Track 1: topology (`topology`)

Started 2026-08-04 from clDice (Shit et al., CVPR 2021), a
vessel-segmentation loss/metric that carried over to the project's
stroke-continuity metrics at the time (see `doc/architecture_decisions.md`
on the `cleanup-refiner` branch, 2026-08-04 clDice entries). The method is
unchanged: look in fields *outside* image/illustration/art for
transferable mechanisms. Per the user (2026-08-04), a genuinely useful
match won't turn up often, so this runs slowly in the background.

Since 2026-11-04 the question is Track M's: the ground truth is the
author's regions from layer masks, and the lines that bound them have
gaps ("not closed, there are gaps"), so the search is for gap-tolerant
contour closure, perceptual grouping, figure-ground / border ownership,
and node/branch topology of line networks.

Files: `focus_areas.txt`, `prompt.md`, `candidates.md`.

## Track 2: line art (`lineart`)

Replaced the diffusion track (2026-08-05 to 2026-11-02) on 2026-11-04. The
diffusion track searched for fine-tuning fixes to ControlNet's loose
condition adherence; no track works on that now, and its log is archived
as `candidates_diffusion.md` (read by the new track for deduplication,
not appended to).

The question is Track N's: compare SD-family line art against line art
from the author's manuscripts along many axes, sorting each into from the
author / from elsewhere / undecided. In scope: where people draw lines,
what lines convey, stroke statistics and style, measuring generated vs
human differences, generation used as a probe, human-aligned evaluation,
and nulls / negative controls. Art papers are in scope.

Files: `focus_areas_lineart.txt`, `prompt_lineart.md`,
`candidates_lineart.md`.

This directory is meant to run on an always-on machine other than the
primary dev machine (which isn't always on), via cron invoking `claude -p`
in headless mode. It's deliberately self-contained -- no dependency on the
rest of this repo -- so it can be checked out on its own with git's
sparse-checkout.

## Setup on the always-on machine

```bash
git clone --filter=blob:none --sparse https://github.com/kuromook/gen_lineart.git
cd gen_lineart
git sparse-checkout set tools/research/cross_domain_paper_scout
git checkout paper-scout
```

Add to crontab (adjust time; weekly is the intended pace -- see "Why
weekly" below):

```
0 9 * * 1 /path/to/gen_lineart/tools/research/cross_domain_paper_scout/scout.sh >> /path/to/gen_lineart/tools/research/cross_domain_paper_scout/cron.log 2>&1
```

`scout.sh` `cd`s to its own directory and resolves the `claude` binary
itself (`command -v`, falling back to `~/.local/bin` and
`/usr/local/bin`), so cron's minimal `PATH` is not a problem. If `claude`
lives elsewhere, set `CLAUDE_BIN=/path/to/claude` in the crontab line.

Requires the `claude` CLI already authenticated on that machine, and git
push access to this repo (the script commits and pushes its own findings
after each run).

The push must work non-interactively. An HTTPS `origin` with no
credential helper will prompt for a username and fail under cron, so use
an SSH remote with a passphrase-less key:

```bash
git remote set-url origin git@github.com:kuromook/gen_lineart.git
ssh -T -o BatchMode=yes git@github.com   # should greet you by username
```

## How it works

One run per week, alternating tracks by ISO week parity: **odd week ->
topology, even week -> lineart**. Each track therefore runs every two
weeks, and its own focus list advances by exactly one entry per run
(`(week / 2) % count`), so alternating does not make either list skip
entries.

- `focus_areas*.txt`: the rotating list of fields/topics for each track,
  one picked per run, so successive runs cover different ground instead
  of repeating the same search.
- `prompt*.md`: the self-contained instructions given to `claude -p` each
  run (headless mode has no memory of previous runs, so these have to
  carry all necessary context on their own -- including the project's
  current aim and the question each track serves).
- `scout.sh`: picks the track and its focus area, runs `claude -p`
  restricted to `WebSearch`/`WebFetch`/`Read`/`Edit` (search the web, edit
  only that track's candidates file -- no shell/git access for the agent
  itself), then commits and pushes that file if it changed. Appends its
  own run log to `scout_run.log` (gitignored). The `claude` call is
  wrapped in a wall-clock `timeout` (default 30m, override with
  `SCOUT_TIMEOUT`) so a hung run can't survive until the next cron firing;
  if it fails or times out the script logs the exit status and still
  commits whatever entries were already appended.
- `candidates*.md`: the accumulating output -- one entry per candidate
  paper, deduplicated against existing entries by the agent each run (it's
  told to read the file first).

To run a specific track by hand, off-schedule:

```bash
SCOUT_TRACK=lineart ./scout.sh    # or SCOUT_TRACK=topology
```

## Reviewing candidates

This is meant to accumulate slowly and get reviewed in batches, not acted
on immediately. Pull the candidates files on the primary machine
periodically (`git pull`, or re-fetch via sparse-checkout) and triage:

- `candidates.md`: is there a specific mechanism for finding the gappy
  group of lines that bounds a region, not just a superficial "also about
  lines" match?
- `candidates_lineart.md`: does anything give a checkable way to say what
  line art conveys, or how human and generated drawings differ? The "Cost
  to try" field is there to make that triage fast.

## Why weekly

The premise (per the user, 2026-08-04) is that real hits will be rare, so
this is meant to run "slowly" in the background rather than burn API
budget searching aggressively. Adding a second track kept the same
one-run-per-week budget rather than doubling it (user, 2026-08-05), which
halves each track's cadence. Since 2026-11-04 the topology track's 10
areas take 20 weeks per full rotation and the line-art track's 8 topics
take 16. Adjust the rotation math in
`scout.sh` if the cadence or the balance between tracks changes.

## Caveats

- CLI flag names were verified against `claude` CLI 2.1.222 on the
  always-on machine (2026-08-05); re-check `claude --help` if that CLI is
  upgraded and runs start failing.
- Non-interactive git push was verified on the always-on machine
  (2026-08-05) after switching `origin` from HTTPS to SSH -- see "Setup"
  above. Note that the `gh` on that machine's `PATH` is *not* GitHub CLI
  (it's an unrelated Python package of the same name), so git's `gh`
  credential fallback does not work there.
- The commit/pull/push block has run end to end under cron since
  2026-10-19 (on `main`, then on `paper-scout` from 2026-10-26).
- If `claude -p` needs a permission-bypass flag on that machine's CLI
  version to run fully unattended (no prompts), add it -- but keep
  `--allowedTools` (or equivalent) scoped to `WebSearch,WebFetch,Read,Edit`
  so an unattended cron job can't take broader action.
