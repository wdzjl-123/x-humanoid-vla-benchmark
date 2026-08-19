#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train_act_5tasks.sh — 在 tmux 后台训练 ACT（默认 320×240），支持任意任务前缀
#
# 第一个参数是【完整任务名】（含 lab_/ind_ 前缀），因此同一个脚本可训练
# lab_* 和 ind_* 全部 5 个任务（不是只能拼 lab_task_<id>）。
#
# 用法:
#   bash tools/train/run_train_act_5tasks.sh <task_name> [num_steps] [batch_size] [resume] [chunk_size]
#
# 示例:
#   bash tools/train/run_train_act_5tasks.sh lab_task_01
#   bash tools/train/run_train_act_5tasks.sh lab_task_03
#   bash tools/train/run_train_act_5tasks.sh ind_task_01
#   bash tools/train/run_train_act_5tasks.sh ind_task_02 50000          # 自定义步数
#   bash tools/train/run_train_act_5tasks.sh ind_task_01 100000 8 best  # 从 agent_best.ckpt 续训
#
# 5 个任务: lab_task_01  lab_task_03  ind_task_01  ind_task_02  ind_task_03
#
# resume 参数:
#   不填 / ""   → 从头训练
#   best        → 使用 <ckpt_dir>/agent_best.ckpt
#   <数字>      → 使用 <ckpt_dir>/policy_step_<数字>.ckpt
#   <路径>      → 直接使用该路径
#
# Python 解释器:
#   默认 python3。若你的 python3 没有 torch（例如 tmux 里默认是 Anaconda），
#   用环境变量覆盖：  PYTHON_BIN=/usr/bin/python3 bash tools/train/run_train_act_5tasks.sh lab_task_01
#   脚本会在起训前预检 torch，缺失时立即报错并给出此提示，而非在 tmux 内闷头失败。
#
# 查看进度:  tmux attach -t act_<task_name>
# 查看日志:  tail -f checkpoints/<task_name>_act/train.log
# ---------------------------------------------------------------------------
set -euo pipefail

TASK_FULL="${1:-}"
NUM_STEPS="${2:-100000}"
BATCH_SIZE="${3:-8}"
RESUME_ARG="${4:-}"
CHUNK_SIZE="${5:-50}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# 图像分辨率：默认 320x240。试 640 用：  IMG_W=640 IMG_H=480 bash $0 <task>
# 训练分辨率必须与推理一致（act_policy.ACTPolicy.IMG_W/H，act 默认即 320）。
IMG_W="${IMG_W:-320}"
IMG_H="${IMG_H:-240}"

if [[ -z "$TASK_FULL" ]]; then
    echo "[ERROR] 必须提供完整任务名，例如: bash $0 ind_task_01"
    echo "        5 个任务: lab_task_01  lab_task_03  ind_task_01  ind_task_02  ind_task_03"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_DIR="$PROJECT_ROOT/data/${TASK_FULL}"
CKPT_DIR="$PROJECT_ROOT/checkpoints/${TASK_FULL}_act"
SESSION="act_${TASK_FULL}"

# --- 预检 1: 数据目录 ---
# 扁平发布布局 data/<task>/train/<ep_id>.hdf5（见 README §2）。
if [[ ! -d "$TASK_DIR/train" ]]; then
    echo "[ERROR] 训练集不存在: $TASK_DIR/train"
    echo "        期望结构: data/${TASK_FULL}/train/<ep_id>.hdf5（见 README §2）"
    exit 1
fi

# --- 预检 2: Python 有 torch（避免 tmux 内无 torch 的解释器闷头失败）---
if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    echo "[ERROR] '$PYTHON_BIN' 无法 import torch。"
    echo "        请用带 torch 的解释器，例如:"
    echo "        PYTHON_BIN=/usr/bin/python3 bash $0 $*"
    exit 1
fi

mkdir -p "$CKPT_DIR"

# --- 解析 resume checkpoint ---
RESUME_FLAG=""
RESUME_PATH=""
if [[ -n "$RESUME_ARG" ]]; then
    if [[ "$RESUME_ARG" == "best" ]]; then
        RESUME_PATH="$CKPT_DIR/agent_best.ckpt"
    elif [[ "$RESUME_ARG" =~ ^[0-9]+$ ]]; then
        RESUME_PATH="$CKPT_DIR/policy_step_${RESUME_ARG}.ckpt"
    else
        RESUME_PATH="$RESUME_ARG"
    fi
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "[ERROR] resume checkpoint 不存在: $RESUME_PATH"
        exit 1
    fi
    RESUME_FLAG="--resume '$RESUME_PATH'"
    echo "[INFO] 续训 from: $RESUME_PATH"
fi

# --- 若 tmux session 已存在则提示 ---
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[WARN] tmux session '$SESSION' 已存在。"
    echo "       Attach:  tmux attach -t $SESSION"
    echo "       Restart: tmux kill-session -t $SESSION && bash $0 $*"
    exit 1
fi

# --- 构造训练命令 ---
TRAIN_CMD="cd '$PROJECT_ROOT' && $PYTHON_BIN tools/train/train_act.py \
  --task-dir '$TASK_DIR' \
  --ckpt-dir  '$CKPT_DIR' \
  --num-steps  $NUM_STEPS \
  --batch-size $BATCH_SIZE \
  --chunk-size $CHUNK_SIZE \
  --img-w $IMG_W --img-h $IMG_H \
  --use-aug \
  --num-workers 4 \
  --validate-every 500 \
  --save-every 2000 \
  $RESUME_FLAG"

# --- 启动 tmux session ---
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter

echo "=============================================="
echo "  ACT 训练已在 tmux 启动: $SESSION"
echo "  Task:       $TASK_FULL"
echo "  Data:       $TASK_DIR"
echo "  Steps:      $NUM_STEPS"
echo "  Batch:      $BATCH_SIZE"
echo "  Chunk:      $CHUNK_SIZE"
echo "  Python:     $PYTHON_BIN"
echo "  Resume:     ${RESUME_PATH:-从头训练}"
echo "  Checkpoint: $CKPT_DIR"
echo "----------------------------------------------"
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl-B then D"
echo "  Log:     tail -f $CKPT_DIR/train.log"
echo "=============================================="
