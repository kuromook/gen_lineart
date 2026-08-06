#!/bin/bash
# Detached watcher: waits for domain_lora_roughclean_20260805 (training +
# contact-sheet generation) to finish, then launches
# run_domain_lora_roughfull_e10_20260805.sh immediately, with NO
# interactive review gate in between.
#
# Explicit user direction (2026-08-05 evening): the roughclean run
# (~2.5-3h) may finish after the user has gone to sleep, and they want
# option 1 (epoch-count isolation on the full pool) to start unattended
# right after rather than wait for morning approval. Runs as its own
# nohup+disown'd process (not a Bash-tool run_in_background call) so it
# survives the CLI session ending, matching the existing pattern used for
# the 2026-07-29 combined-koma training completion notifier.
#
# Safety: if roughclean's process disappears WITHOUT its .done marker
# (i.e. it crashed rather than completed), this does NOT launch roughfull
# -- an unattended ~17h run on top of an already-failed premise would
# waste the whole overnight/daytime GPU window for nothing. It just logs
# and exits so the failure is visible to review in the morning.
set -uo pipefail
cd "$(dirname "$0")/.."

WATCH_DONE=logs/domain_lora_roughclean_20260805.done
WATCH_PROC_PATTERN="run_domain_lora_roughclean_20260805.sh"
CHAIN_LOG=logs/domain_lora_chain_roughclean_then_roughfull_20260805.log

mkdir -p logs
echo "[$(date --iso-8601=seconds)] chain watcher started (pid $$), waiting for $WATCH_DONE" >> "$CHAIN_LOG"

while true; do
  if [ -f "$WATCH_DONE" ]; then
    echo "[$(date --iso-8601=seconds)] $WATCH_DONE found -- launching roughfull_e10 run" >> "$CHAIN_LOG"
    bash experiments/run_domain_lora_roughfull_e10_20260805.sh >> "$CHAIN_LOG" 2>&1
    echo "[$(date --iso-8601=seconds)] chain complete" >> "$CHAIN_LOG"
    exit 0
  fi
  if ! pgrep -f "$WATCH_PROC_PATTERN" > /dev/null; then
    echo "[$(date --iso-8601=seconds)] roughclean process gone but no $WATCH_DONE marker -- treating as failed/incomplete, NOT auto-launching roughfull. Review logs/domain_lora_roughclean_20260805.log manually." >> "$CHAIN_LOG"
    exit 1
  fi
  sleep 60
done
