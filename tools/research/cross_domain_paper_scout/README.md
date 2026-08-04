# Cross-Domain Paper Scout

Motivation: clDice (Shit et al., "clDice -- A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021) is a topology loss
and evaluation metric from *medical vessel segmentation* that turned out to
transfer well to this project's line-art stroke-continuity problem (see
`doc/architecture_decisions.md` on the `cleanup-refiner` branch, 2026-08-04
clDice entries). This tool looks for more of that: mathematically
transferable ideas -- loss functions, evaluation metrics, algorithms -- from
fields *outside* image/illustration/art that deal with thin, continuous,
topologically important structures (vessels, roads, cracks, wires, neurons,
fault lines, ...). Per the user (2026-08-04), a genuinely useful match like
clDice won't turn up often, so this is meant to run slowly and continuously
in the background rather than be searched for in one sitting.

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
0 9 * * 1 cd /path/to/gen_lineart/tools/research/cross_domain_paper_scout && ./scout.sh >> cron.log 2>&1
```

Requires the `claude` CLI already authenticated on that machine, and git
push access to this repo (the script commits and pushes its own findings
after each run).

## How it works

- `focus_areas.txt`: a rotating list of non-art fields to search. `scout.sh`
  picks one per run based on the current ISO week number, so successive
  runs cover different ground instead of repeating the same search.
- `prompt.md`: the self-contained instructions given to `claude -p` each
  run (headless mode has no memory of previous runs, so this has to carry
  all necessary context on its own).
- `scout.sh`: picks the week's focus area, runs `claude -p` restricted to
  `WebSearch`/`WebFetch`/`Read`/`Edit` (search the web, edit only
  `candidates.md` -- no shell/git access for the agent itself), then
  commits and pushes `candidates.md` if it changed. Appends its own run
  log to `scout_run.log` (gitignored).
- `candidates.md`: the accumulating output -- one entry per candidate
  paper, deduplicated against existing entries by the agent each run (it's
  told to read the file first).

## Reviewing candidates

This is meant to accumulate slowly and get reviewed in batches, not acted
on immediately. Pull `candidates.md` on the primary machine periodically
(`git pull`, or re-fetch via sparse-checkout) and triage: does anything
look like a genuine, specific transfer opportunity (the way clDice was),
not just a superficial "also about lines" match?

## Why weekly

The premise (per the user, 2026-08-04) is that real hits will be rare, so
this is meant to run "slowly" in the background rather than burn API
budget searching aggressively. Weekly gives the ~13 areas in
`focus_areas.txt` roughly 4 full rotations a year. Adjust the rotation
math in `scout.sh` if the cadence changes.

## Caveats

- CLI flag names (`--allowedTools` etc.) in `scout.sh` were not verified
  against the target machine's installed `claude` CLI version from the
  session that authored this script -- check `claude --help` there and
  adjust if flags differ.
- The script assumes non-interactive git push works (credential helper /
  SSH key already configured on that machine).
- If `claude -p` needs a permission-bypass flag on that machine's CLI
  version to run fully unattended (no prompts), add it -- but keep
  `--allowedTools` (or equivalent) scoped to `WebSearch,WebFetch,Read,Edit`
  so an unattended cron job can't take broader action.
