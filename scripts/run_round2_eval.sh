#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="$ROOT_DIR/subleq-transformer/round2_trained"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/run_round2_eval.sh checkpoints/<run_name>/best_model.pt [eval args...]" >&2
  exit 1
fi

MODEL_PATH="$1"
shift

mkdir -p "$ROOT_DIR/runs/eval"

RUN_LABEL="$(basename "$(dirname "$MODEL_PATH")")"
LOG_PATH="$ROOT_DIR/runs/eval/${RUN_LABEL}.log"

cd "$UPSTREAM_DIR"

"$PYTHON_BIN" eval.py "$MODEL_PATH" "$@" | tee "$LOG_PATH"
