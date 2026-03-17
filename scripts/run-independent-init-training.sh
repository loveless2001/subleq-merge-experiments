#!/usr/bin/env bash
# Train an independent-init specialist from scratch (no ancestor).
# Usage: bash scripts/run-independent-init-training.sh specialist_a|specialist_b <seed>
# Example: bash scripts/run-independent-init-training.sh specialist_a 42
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

PROFILE="${1:?Usage: $0 specialist_a|specialist_b <seed>}"
SEED="${2:?Usage: $0 specialist_a|specialist_b <seed>}"
TOTAL_STEPS="${TOTAL_STEPS:-80000}"
RUN_NAME="independent_${PROFILE}_seed${SEED}"
SAVE_DIR="$ROOT_DIR/checkpoints/$RUN_NAME"
LOG_DIR="$ROOT_DIR/runs/$RUN_NAME"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Training $PROFILE from scratch (seed=$SEED, steps=$TOTAL_STEPS)"
echo "Save: $SAVE_DIR"

cd "$UPSTREAM_DIR"

"$PYTHON_BIN" train.py \
  --total-steps "$TOTAL_STEPS" \
  --save-dir "$SAVE_DIR" \
  --seed "$SEED" \
  --profile "$PROFILE" \
  "${@:3}" | tee "$LOG_DIR/train.log"
