#!/bin/bash
# Wait for a user service to finish, then notify based on a done marker.
set -euo pipefail
cd "$(dirname "$0")/.."

UNIT=${1:?unit name required}
DONE=${2:?done marker required}
TITLE=${3:-"Lineart job complete"}
BODY=${4:-"Review generated artifacts."}
INTERVAL=${INTERVAL:-30}

while systemctl --user is-active --quiet "$UNIT"; do
  sleep "$INTERVAL"
done

if [[ -s "$DONE" ]]; then
  experiments/send_autoloop_notification.sh "$TITLE" "$BODY" || true
else
  experiments/send_autoloop_notification.sh \
    "Lineart job stopped without done marker" \
    "$UNIT stopped before writing $DONE" || true
fi
