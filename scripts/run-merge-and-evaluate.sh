#!/usr/bin/env bash
# Merge two specialists and evaluate the result.
# Usage: bash scripts/run-merge-and-evaluate.sh <method> [extra merge args...]
# Example: bash scripts/run-merge-and-evaluate.sh task_arithmetic --scaling 0.7
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

METHOD="${1:?Usage: $0 naive_average|task_arithmetic|ties|slerp [extra args...]}"
shift

ANCESTOR="${ANCESTOR:-$ROOT_DIR/checkpoints/round2_ancestor_40k/best_model.pt}"
SPEC_A="${SPEC_A:-$ROOT_DIR/checkpoints/specialist_a/best_model.pt}"
SPEC_B="${SPEC_B:-$ROOT_DIR/checkpoints/specialist_b/best_model.pt}"
MERGED_DIR="$ROOT_DIR/checkpoints/merged_${METHOD}"
MERGED_PATH="$MERGED_DIR/merged_model.pt"
LOG_DIR="$ROOT_DIR/runs/eval"

mkdir -p "$MERGED_DIR" "$LOG_DIR"

echo "=== Merging with method: $METHOD ==="
cd "$UPSTREAM_DIR"

"$PYTHON_BIN" merge-specialist-checkpoints.py \
  --ancestor "$ANCESTOR" \
  --specialist-a "$SPEC_A" \
  --specialist-b "$SPEC_B" \
  --method "$METHOD" \
  --output "$MERGED_PATH" \
  "$@"

echo ""
echo "=== Evaluating merged model ==="
"$PYTHON_BIN" eval.py "$MERGED_PATH" | tee "$LOG_DIR/merged_${METHOD}.log"
