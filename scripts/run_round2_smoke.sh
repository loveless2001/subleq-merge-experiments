#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-round2_smoke}"
SAVE_DIR="$ROOT_DIR/checkpoints/$RUN_NAME"
LOG_DIR="$ROOT_DIR/runs/$RUN_NAME"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

cd "$UPSTREAM_DIR"

"$PYTHON_BIN" train.py \
  --total-steps "${TOTAL_STEPS:-500}" \
  --data-size "${DATA_SIZE:-4000}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --log-every "${LOG_EVERY:-50}" \
  --eval-every "${EVAL_EVERY:-250}" \
  --regen-every "${REGEN_EVERY:-250}" \
  --save-dir "$SAVE_DIR" \
  "$@" | tee "$LOG_DIR/train.log"
