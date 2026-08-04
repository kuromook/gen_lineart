#!/bin/bash
# Cross-domain paper scout: runs `claude -p` in headless mode against a
# rotating focus field (see focus_areas.txt) to search for mathematically
# transferable topology/continuity ideas from non-art fields. See
# README.md for the full design rationale and setup instructions.
#
# Meant to run via cron on an always-on machine, sparse-checked-out from
# the main dev repo (which isn't always on). Self-contained: no
# dependency on the rest of this repo.
set -euo pipefail
cd "$(dirname "$0")"

FOCUS_AREAS_FILE=focus_areas.txt
NUM_AREAS=$(grep -c . "$FOCUS_AREAS_FILE")
WEEK=$((10#$(date -u +%U)))
IDX=$((WEEK % NUM_AREAS))
FOCUS=$(sed -n "$((IDX + 1))p" "$FOCUS_AREAS_FILE")

echo "[$(date -u --iso-8601=seconds)] run start, focus=\"$FOCUS\""

PROMPT=$(cat prompt.md)
PROMPT=${PROMPT//__FOCUS_AREA__/$FOCUS}

# Restricted to web search + editing only this directory's candidates.md;
# no shell/git access for the agent itself -- git is handled below.
# NOTE: verify these flag names against `claude --help` on this machine;
# not independently verified from the session that authored this script.
claude -p "$PROMPT" \
  --allowedTools "WebSearch,WebFetch,Read,Edit" \
  2>&1 | tee -a scout_run.log

if ! git diff --quiet -- candidates.md; then
  git add candidates.md
  git commit -m "Cross-domain scout: candidates from ${FOCUS} ($(date -u +%Y-%m-%d))"
  git pull --rebase
  git push
  echo "[$(date -u --iso-8601=seconds)] new candidates committed and pushed"
else
  echo "[$(date -u --iso-8601=seconds)] no new candidates this run"
fi
