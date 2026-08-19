#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-flow}"
mkdir -p "$MPLCONFIGDIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-act/bin/python}"
DEVICE="${DEVICE:-cuda}"
FLOW_STEPS="${FLOW_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NHEADS="${NHEADS:-8}"
IMG_W="${IMG_W:-320}"
IMG_H="${IMG_H:-240}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.0001}"

TASKS=(
  ind_task_01
  ind_task_02
  ind_task_03
  lab_task_01
  lab_task_03
)

for task in "${TASKS[@]}"; do
  echo "===== $(date '+%F %T') start ${task} flow ====="
  "$PYTHON_BIN" tools/train/train_flow.py \
    --task-dir "data/${task}" \
    --ckpt-dir "checkpoints/${task}_flow" \
    --num-steps "$FLOW_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --nheads "$NHEADS" \
    --img-w "$IMG_W" --img-h "$IMG_H" \
    --use-aug \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --validate-every 500 \
    --save-every 5000 \
    --device "$DEVICE"
  echo "===== $(date '+%F %T') done ${task} flow ====="
done
