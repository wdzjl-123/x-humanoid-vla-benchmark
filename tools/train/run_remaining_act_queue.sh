#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-act}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$MPLCONFIGDIR"

TASKS=(
  ind_task_02
  ind_task_03
  lab_task_01
  lab_task_03
)

for task in "${TASKS[@]}"; do
  echo "===== $(date '+%F %T') start ${task} ====="
  .venv-act/bin/python tools/train/train_act.py \
    --task-dir "data/${task}" \
    --ckpt-dir "checkpoints/${task}_act" \
    --num-steps 100000 \
    --batch-size 8 \
    --chunk-size 50 \
    --img-w 320 --img-h 240 \
    --use-aug \
    --num-workers 4 \
    --validate-every 500 \
    --save-every 2000
  echo "===== $(date '+%F %T') done ${task} ====="
done
