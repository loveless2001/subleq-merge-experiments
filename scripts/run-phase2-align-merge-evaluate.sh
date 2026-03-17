#!/usr/bin/env bash
# Phase 2: Align model Y to model X using Git Re-Basin, then merge and evaluate.
# Usage: bash scripts/run-phase2-align-merge-evaluate.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

MODEL_X="${MODEL_X:-$ROOT_DIR/checkpoints/independent_specialist_a_seed42/best_model.pt}"
MODEL_Y="${MODEL_Y:-$ROOT_DIR/checkpoints/independent_specialist_b_seed123/best_model.pt}"
ALIGNED_DIR="$ROOT_DIR/checkpoints/independent_y_aligned"
LOG_DIR="$ROOT_DIR/runs/eval"

mkdir -p "$ALIGNED_DIR" "$LOG_DIR"
cd "$UPSTREAM_DIR"

run_eval() {
  local model_path="$1"
  local log_path="$2"
  set +e
  "$PYTHON_BIN" eval.py "$model_path" | tee "$log_path"
  local status=$?
  set -e
  return $status
}

echo "============================================"
echo "  Phase 2: Independent Init Merge"
echo "============================================"

# Step 1: Evaluate unaligned models
echo ""
echo ">>> Evaluating Model X (specialist_a, seed42)..."
run_eval "$MODEL_X" "$LOG_DIR/independent_x.log" || true

echo ""
echo ">>> Evaluating Model Y (specialist_b, seed123)..."
run_eval "$MODEL_Y" "$LOG_DIR/independent_y.log" || true

# Step 2: Naive merge (expected to fail)
echo ""
echo ">>> Naive average (no alignment, expected to fail)..."
"$PYTHON_BIN" merge-specialist-checkpoints.py \
  --ancestor "$MODEL_X" \
  --specialist-a "$MODEL_X" \
  --specialist-b "$MODEL_Y" \
  --method naive_average \
  --output "$ROOT_DIR/checkpoints/independent_naive_avg/merged_model.pt"
run_eval "$ROOT_DIR/checkpoints/independent_naive_avg/merged_model.pt" \
  "$LOG_DIR/independent_naive_avg.log" || true

# Step 3: Align Y to X
echo ""
echo ">>> Aligning Model Y to Model X (Git Re-Basin)..."
"$PYTHON_BIN" align-models-git-rebasin.py \
  --model-a "$MODEL_X" \
  --model-b "$MODEL_Y" \
  --output "$ALIGNED_DIR/aligned_model.pt"

# Step 4: Merge aligned models
echo ""
echo ">>> Aligned average..."
"$PYTHON_BIN" merge-specialist-checkpoints.py \
  --ancestor "$MODEL_X" \
  --specialist-a "$MODEL_X" \
  --specialist-b "$ALIGNED_DIR/aligned_model.pt" \
  --method naive_average \
  --output "$ROOT_DIR/checkpoints/independent_aligned_avg/merged_model.pt"
run_eval "$ROOT_DIR/checkpoints/independent_aligned_avg/merged_model.pt" \
  "$LOG_DIR/independent_aligned_avg.log" || true

# Step 5: SLERP on aligned models
echo ""
echo ">>> SLERP on aligned models..."
"$PYTHON_BIN" merge-specialist-checkpoints.py \
  --ancestor "$MODEL_X" \
  --specialist-a "$MODEL_X" \
  --specialist-b "$ALIGNED_DIR/aligned_model.pt" \
  --method slerp \
  --slerp-t 0.5 \
  --output "$ROOT_DIR/checkpoints/independent_aligned_slerp/merged_model.pt"
run_eval "$ROOT_DIR/checkpoints/independent_aligned_slerp/merged_model.pt" \
  "$LOG_DIR/independent_aligned_slerp.log" || true

# Step 6: Cross-skill eval on best merged model
echo ""
echo ">>> Cross-skill composition eval..."
"$PYTHON_BIN" eval-cross-skill-composition.py \
  "$ROOT_DIR/checkpoints/independent_aligned_avg/merged_model.pt" \
  | tee "$LOG_DIR/independent_aligned_avg_crossskill.log"

echo ""
echo "============================================"
echo "  Phase 2 complete!"
echo "  Logs in: $LOG_DIR/"
echo "============================================"
