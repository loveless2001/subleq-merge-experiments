#!/usr/bin/env bash
# Run the full merge experiment: all methods, all evaluations.
# Assumes ancestor + both specialists are already trained.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

echo "============================================"
echo "  SUBLEQ Merge Experiment — Full Pipeline"
echo "============================================"

# Evaluate ancestor baseline
echo ""
echo ">>> Evaluating ancestor baseline..."
bash "$SCRIPTS_DIR/run_round2_eval.sh" "$ROOT_DIR/checkpoints/round2_ancestor_40k/best_model.pt"

# Evaluate specialist A
echo ""
echo ">>> Evaluating specialist A..."
bash "$SCRIPTS_DIR/run_round2_eval.sh" "$ROOT_DIR/checkpoints/specialist_a/best_model.pt"

# Evaluate specialist B
echo ""
echo ">>> Evaluating specialist B..."
bash "$SCRIPTS_DIR/run_round2_eval.sh" "$ROOT_DIR/checkpoints/specialist_b/best_model.pt"

# Merge experiments
echo ""
echo ">>> Merge: naive_average..."
bash "$SCRIPTS_DIR/run-merge-and-evaluate.sh" naive_average

echo ""
echo ">>> Merge: task_arithmetic (scaling=1.0)..."
bash "$SCRIPTS_DIR/run-merge-and-evaluate.sh" task_arithmetic --scaling 1.0

echo ""
echo ">>> Merge: task_arithmetic (scaling=0.7)..."
MERGED_DIR_OVERRIDE=merged_task_arithmetic_07
bash "$SCRIPTS_DIR/run-merge-and-evaluate.sh" task_arithmetic --scaling 0.7

echo ""
echo ">>> Merge: ties (top_k=0.2)..."
bash "$SCRIPTS_DIR/run-merge-and-evaluate.sh" ties --ties-top-k 0.2

echo ""
echo ">>> Merge: slerp (t=0.5)..."
bash "$SCRIPTS_DIR/run-merge-and-evaluate.sh" slerp --slerp-t 0.5

echo ""
echo "============================================"
echo "  All merge experiments complete!"
echo "  Logs in: $ROOT_DIR/runs/eval/"
echo "============================================"
