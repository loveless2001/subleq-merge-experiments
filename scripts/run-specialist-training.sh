#!/usr/bin/env bash
# Train a specialist model from the ancestor checkpoint.
# Usage: bash scripts/run-specialist-training.sh specialist_a|specialist_b
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

PROFILE="${1:?Usage: $0 specialist_a|specialist_b}"
ANCESTOR="${ANCESTOR:-$ROOT_DIR/checkpoints/round2_ancestor_40k/best_model.pt}"
TOTAL_STEPS="${TOTAL_STEPS:-25000}"
SAVE_DIR="$ROOT_DIR/checkpoints/$PROFILE"
LOG_DIR="$ROOT_DIR/runs/$PROFILE"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "Training $PROFILE from ancestor: $ANCESTOR"
echo "Steps: $TOTAL_STEPS | Save: $SAVE_DIR"

cd "$UPSTREAM_DIR"

"$PYTHON_BIN" train-specialist-from-ancestor-checkpoint.py \
  --ancestor-checkpoint "$ANCESTOR" \
  --profile "$PROFILE" \
  --save-dir "$SAVE_DIR" \
  --total-steps "$TOTAL_STEPS" \
  "${@:2}" | tee "$LOG_DIR/train.log"
