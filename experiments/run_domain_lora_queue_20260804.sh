#!/bin/bash
# Runs the two domain-LoRA training variants sequentially (line first,
# ~1.6h, then rough, ~9.3h at 4 epochs -- GPU is free, no prior job to
# wait on).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/queue_domain_lora_20260804.log
mkdir -p logs
: > "$LOG"

echo "[$(date --iso-8601=seconds)] starting line-domain LoRA run" >> "$LOG"
bash experiments/run_domain_lora_line_20260804.sh >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] line-domain done, starting rough-domain LoRA run" >> "$LOG"
bash experiments/run_domain_lora_rough_20260804.sh >> "$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] both domain-LoRA runs complete" >> "$LOG"
