#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_model.sh r7 [runner arguments]

External model files must be mounted below /models by default:
  /models/r7/policy_last.ckpt
  /models/r7/dataset_stats.pkl
Set MODEL_ROOT to use a different mounted model directory. The remaining
arguments are passed directly to the selected benchmark runner. Example:
  run_model.sh r7 --task ind_task_01 --sim-host "$SIM_HOST" \
    --zmq-recv-port "$ZMQ_RECV_PORT" --zmq-send-port "$ZMQ_SEND_PORT" --zmq-bind-host 0.0.0.0 \
    --task-publish-delay 5 --chunk-size 50 --temporal-agg \
    --act-temporal-decay 0.01 --act-temporal-priority oldest \
    --multitask-hand-state measured --multitask-hand-feedback measured \
    --act-image-color-order rgb --act-debug-every 50
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

profile="$1"
shift
model_root="${MODEL_ROOT:-/models}"
device="${DEVICE:-cuda}"

case "$profile" in
    r7)
        runner="/opt/xhumanoid/runner_r7.py"
        model_dir="$model_root/r7"
        extra_args=(--prewarm-steps 5)
        ;;
    *)
        printf 'Unknown model profile: %s\n\n' "$profile" >&2
        usage >&2
        exit 2
        ;;
esac

source_model_path="$model_dir/policy_last.ckpt"
source_stats_path="$model_dir/dataset_stats.pkl"
if [[ ! -f "$source_model_path" || ! -f "$source_stats_path" ]]; then
    printf 'Missing model files for profile %s below %s\n' "$profile" "$model_dir" >&2
    exit 3
fi

# The policy writes its audit log alongside its checkpoint. Keep externally
# supplied weights immutable by copying the selected pair into a writable,
# container-local runtime directory before the runner loads them.
runtime_root="${XHUMANOID_RUNTIME_DIR:-/tmp/xhumanoid-models}"
runtime_dir="$runtime_root/$profile"
mkdir -p "$runtime_dir"
cp -f "$source_model_path" "$runtime_dir/policy_last.ckpt"
cp -f "$source_stats_path" "$runtime_dir/dataset_stats.pkl"
model_path="$runtime_dir/policy_last.ckpt"

exec python3 -u "$runner" \
    --policy multitask_act \
    --model-path "$model_path" \
    --device "$device" \
    "${extra_args[@]}" \
    "$@"
