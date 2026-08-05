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

# cron's default PATH (/usr/bin:/bin) does not include the usual `claude`
# install location, so resolve the binary explicitly. Override with
# CLAUDE_BIN=/path/to/claude if it lives somewhere else.
if [ -z "${CLAUDE_BIN:-}" ]; then
  CLAUDE_BIN=$(command -v claude || true)
fi
if [ -z "$CLAUDE_BIN" ]; then
  for candidate in "$HOME/.local/bin/claude" /usr/local/bin/claude; do
    if [ -x "$candidate" ]; then
      CLAUDE_BIN=$candidate
      break
    fi
  done
fi
if [ -z "$CLAUDE_BIN" ]; then
  echo "[$(date -u --iso-8601=seconds)] error: claude CLI not found; set CLAUDE_BIN" >&2
  exit 1
fi

# Wall-clock cap so a hung run can't sit around until the next cron firing.
RUN_TIMEOUT=${SCOUT_TIMEOUT:-30m}

WEEK=$((10#$(date -u +%U)))

# Two tracks share the one weekly slot, alternating by week parity: odd
# ISO week -> topology, even -> diffusion. Override for a manual run with
# SCOUT_TRACK=topology|diffusion.
TRACK=${SCOUT_TRACK:-}
if [ -z "$TRACK" ]; then
  if [ $((WEEK % 2)) -eq 1 ]; then TRACK=topology; else TRACK=diffusion; fi
fi

case "$TRACK" in
  topology)
    FOCUS_AREAS_FILE=focus_areas.txt
    PROMPT_FILE=prompt.md
    CANDIDATES_FILE=candidates.md
    ;;
  diffusion)
    FOCUS_AREAS_FILE=focus_areas_diffusion.txt
    PROMPT_FILE=prompt_diffusion.md
    CANDIDATES_FILE=candidates_diffusion.md
    ;;
  *)
    echo "[$(date -u --iso-8601=seconds)] error: unknown SCOUT_TRACK \"$TRACK\"" >&2
    exit 1
    ;;
esac

# Select among non-empty lines only, so both the count and the pick stay
# consistent if blank lines are ever added to the file. Each track's
# rotation advances once per *its own* runs (i.e. every two weeks), so
# alternating tracks does not make either list skip entries.
mapfile -t AREAS < <(grep . "$FOCUS_AREAS_FILE")
NUM_AREAS=${#AREAS[@]}
IDX=$(( (WEEK / 2) % NUM_AREAS ))
FOCUS=${AREAS[$IDX]}

# Always log how the run ended, including aborts (set -e, git failures).
trap 'echo "[$(date -u --iso-8601=seconds)] run end, exit=$?"' EXIT

echo "[$(date -u --iso-8601=seconds)] run start, track=$TRACK, focus=\"$FOCUS\""

PROMPT=$(cat "$PROMPT_FILE")
PROMPT=${PROMPT//__FOCUS_AREA__/$FOCUS}

# Restricted to web search + editing only this track's candidates file;
# no shell/git access for the agent itself -- git is handled below.
# Flag names verified against claude CLI 2.1.222 (2026-08-05).
set +e
timeout "$RUN_TIMEOUT" "$CLAUDE_BIN" -p "$PROMPT" \
  --allowedTools "WebSearch,WebFetch,Read,Edit" \
  2>&1 | tee -a scout_run.log
CLAUDE_STATUS=${PIPESTATUS[0]}
set -e

if [ "$CLAUDE_STATUS" -ne 0 ]; then
  # 124 = killed by `timeout`. Fall through either way: any entry the agent
  # already appended is self-contained and worth keeping.
  echo "[$(date -u --iso-8601=seconds)] warning: claude exited with status $CLAUDE_STATUS" \
    | tee -a scout_run.log
fi

if ! git diff --quiet -- "$CANDIDATES_FILE"; then
  git add "$CANDIDATES_FILE"
  git commit -m "Paper scout (${TRACK}): candidates from ${FOCUS} ($(date -u +%Y-%m-%d))"
  git pull --rebase
  git push
  echo "[$(date -u --iso-8601=seconds)] new candidates committed and pushed"
else
  echo "[$(date -u --iso-8601=seconds)] no new candidates this run"
fi
