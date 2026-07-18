#!/bin/bash
# Send autoloop completion notifications through locally configured channels.
set -euo pipefail

TITLE=${1:-"Lineart autoloop complete"}
BODY=${2:-"Review generated artifacts."}
CONFIG=${AUTOLOOP_NOTIFY_CONFIG:-config/autoloop_notify.env}

if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
fi

MESSAGE="$TITLE
$BODY"

if command -v notify-send >/dev/null 2>&1; then
  notify-send "$TITLE" "$BODY" || true
fi

if command -v wall >/dev/null 2>&1; then
  printf '%s\n' "$MESSAGE" | wall || true
fi

if [[ -n "${AUTOLOOP_NOTIFY_EMAIL_TO:-}" ]] && command -v mail >/dev/null 2>&1; then
  printf '%s\n' "$BODY" | mail -s "$TITLE" "$AUTOLOOP_NOTIFY_EMAIL_TO" || true
fi

if [[ -n "${AUTOLOOP_NOTIFY_WEBHOOK_TEXT_URL:-}" ]] && command -v curl >/dev/null 2>&1; then
  CURL_HEADERS=("--header" "Content-Type: text/plain; charset=utf-8")
  if [[ "${AUTOLOOP_NOTIFY_WEBHOOK_TEXT_URL}" == https://ntfy.sh/* ]]; then
    CURL_HEADERS+=(
      "--header" "Title: $TITLE"
      "--header" "Priority: ${AUTOLOOP_NOTIFY_NTFY_PRIORITY:-high}"
      "--header" "Tags: ${AUTOLOOP_NOTIFY_NTFY_TAGS:-warning}"
    )
  fi
  curl --fail --silent --show-error \
    --max-time "${AUTOLOOP_NOTIFY_TIMEOUT:-20}" \
    "${CURL_HEADERS[@]}" \
    --data-binary "$MESSAGE" \
    "$AUTOLOOP_NOTIFY_WEBHOOK_TEXT_URL" || true
fi

if [[ -n "${AUTOLOOP_NOTIFY_WEBHOOK_JSON_URL:-}" ]] && command -v curl >/dev/null 2>&1; then
  JSON_PAYLOAD=$(
    TITLE="$TITLE" BODY="$BODY" "${PYTHON:-python3}" - <<'PY'
import json
import os

print(json.dumps({
    "text": f"{os.environ['TITLE']}\n{os.environ['BODY']}",
    "title": os.environ["TITLE"],
    "body": os.environ["BODY"],
}))
PY
  )
  curl --fail --silent --show-error \
    --max-time "${AUTOLOOP_NOTIFY_TIMEOUT:-20}" \
    --header "Content-Type: application/json" \
    --data-binary "$JSON_PAYLOAD" \
    "$AUTOLOOP_NOTIFY_WEBHOOK_JSON_URL" || true
fi

if [[ -n "${AUTOLOOP_NOTIFY_SSH_TARGET:-}" ]] && command -v ssh >/dev/null 2>&1; then
  SSH_COMMAND=${AUTOLOOP_NOTIFY_SSH_COMMAND:-"cat >> ~/lineart_autoloop_notifications.log"}
  printf '%s\n' "$MESSAGE" | ssh \
    -o BatchMode=yes \
    -o ConnectTimeout="${AUTOLOOP_NOTIFY_TIMEOUT:-20}" \
    "$AUTOLOOP_NOTIFY_SSH_TARGET" \
    "$SSH_COMMAND" || true
fi

printf '\a' || true
