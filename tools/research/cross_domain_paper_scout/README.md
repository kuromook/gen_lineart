# Paper Scout

A slow background paper search for this project, running two independent
tracks that alternate weeks. Both are meant to accumulate over months and
be triaged in batches, not acted on immediately.

## Track 1: topology/continuity transfer from non-art fields (`topology`)

Motivation: clDice (Shit et al., "clDice -- A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021) is a topology loss
and evaluation metric from *medical vessel segmentation* that turned out to
transfer well to this project's line-art stroke-continuity problem (see
`doc/architecture_decisions.md` on the `cleanup-refiner` branch, 2026-08-04
clDice entries). This track looks for more of that: mathematically
transferable ideas -- loss functions, evaluation metrics, algorithms -- from
fields *outside* image/illustration/art that deal with thin, continuous,
topologically important structures (vessels, roads, cracks, wires, neurons,
fault lines, ...). Per the user (2026-08-04), a genuinely useful match like
clDice won't turn up often, so this is meant to run slowly and continuously
in the background rather than be searched for in one sitting.

Files: `focus_areas.txt`, `prompt.md`, `candidates.md`.

## Track 2: diffusion (`diffusion`)

Added 2026-08-05. The deterministic U-Net line is being continued quietly
while diffusion is the active exploration, so the search covers it too.
Unlike track 1 this is an *in-domain* search -- art/anime papers are fine;
the filter is whether a paper's mechanism speaks to a failure mode this
project actually hit.

Scope, per the user (2026-08-05), is two areas:

1. **structure control / condition fidelity** -- the current blocker.
   ControlNet-conditioned SD1.5 produced the project's first crisp binary
   ink, but hallucinates: output corresponds only loosely to the input
   rough (F1@2px 0.20 vs the CNN+GAN baseline's 0.42, ink_ratio ~6x).
   Conditioning/guidance-scale sweeps did not fix it, and a 10x-longer run
   rejected the undertraining hypothesis.
2. **low-data / LoRA / domain adaptation** -- ~1,489 paired tiles is small
   for diffusion fine-tuning, and the current direction includes
   domain-only LoRA training on the plentiful unpaired rough material.

Deliberately out of scope: samplers/distillation/inference speed (not the
quality blocker), and discrete/binary diffusion formulations (binarization
is not what is failing here).

Files: `focus_areas_diffusion.txt`, `prompt_diffusion.md`,
`candidates_diffusion.md`.

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
git checkout main
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
topology, even week -> diffusion**. Each track therefore runs every two
weeks, and its own focus list advances by exactly one entry per run
(`(week / 2) % count`), so alternating does not make either list skip
entries.

- `focus_areas*.txt`: the rotating list of fields/topics for each track,
  one picked per run, so successive runs cover different ground instead
  of repeating the same search.
- `prompt*.md`: the self-contained instructions given to `claude -p` each
  run (headless mode has no memory of previous runs, so these have to
  carry all necessary context on their own -- including, for the
  diffusion track, the current experimental state and what has already
  been ruled out).
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
SCOUT_TRACK=diffusion ./scout.sh    # or SCOUT_TRACK=topology
```

## Reviewing candidates

This is meant to accumulate slowly and get reviewed in batches, not acted
on immediately. Pull the candidates files on the primary machine
periodically (`git pull`, or re-fetch via sparse-checkout) and triage:

- `candidates.md`: does anything look like a genuine, specific transfer
  opportunity (the way clDice was), not just a superficial "also about
  lines" match?
- `candidates_diffusion.md`: does anything address a failure mode this
  project actually has, at a cost it can afford (single GPU, ~1.5k paired
  tiles)? The "Cost to try" field is there to make that triage fast.

## Why weekly

The premise (per the user, 2026-08-04) is that real hits will be rare, so
this is meant to run "slowly" in the background rather than burn API
budget searching aggressively. Adding the diffusion track kept the same
one-run-per-week budget rather than doubling it (user, 2026-08-05), which
costs the topology track half its cadence: its ~13 areas now take ~26
weeks per full rotation (~2 a year) instead of ~4. The 14 diffusion topics
rotate on the same ~26-week period. Adjust the rotation math in
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
- Only the manual push has been exercised, not one issued by `scout.sh`
  itself under cron. The first run that actually finds a candidate is
  still the first end-to-end test of the commit/pull/push block -- check
  `cron.log` after it.
- If `claude -p` needs a permission-bypass flag on that machine's CLI
  version to run fully unattended (no prompts), add it -- but keep
  `--allowedTools` (or equivalent) scoped to `WebSearch,WebFetch,Read,Edit`
  so an unattended cron job can't take broader action.
