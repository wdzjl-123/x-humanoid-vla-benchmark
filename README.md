# X-Humanoid VLA Benchmark

This repository is a sanitized reproducibility snapshot of a five-task
RGB-D ACT policy developed for the X-Humanoid VLA benchmark.

## Included

- `tools/`, `common/`, `tests/`, and `splits/`: training, evaluation, policy,
  and runtime code used for the shared ACT pipeline.
- `submission/`: inference-only Docker source and the R7 checkpoint.
- `videos/`: one local Isaac Sim closed-loop diagnostic recording per task.
- `MODEL_CARD.md`, `RESULTS.md`, and `VIDEOS.md`: model identity, reported
  results, provenance, and evidence boundaries.

The raw HDF5 dataset, SSH keys, virtual environments, official-platform logs,
and intermediate checkpoints are intentionally excluded.

The R7 checkpoint is stored with Git LFS. Install Git LFS and clone with
`git lfs install` before checking out the model artifact.

## Model

R7 is one shared task-conditioned RGB-D ACT checkpoint for:

```text
RGB + depth + 26D robot state + 14D end-effector pose + 5-way task ID
    -> shared ACT Transformer (action chunk size 50)
    -> 26D arm/hand action
```

The same checkpoint is used for all five task names; the task ID is a
conditioning input, not a selector for task-specific weights.

## Local inference

The submission runner expects the model directory to be mounted at `/models`.
Replace the simulator host and ports with values for your own environment:

```bash
docker build -t xhumanoid-vla-r7:20260819 submission
docker run --rm --gpus all --network host \
  -v "$PWD/submission/models:/models:ro" \
  xhumanoid-vla-r7:20260819 r7 \
  --task ind_task_01 --sim-host "$SIM_HOST" \
  --zmq-recv-port "$ZMQ_RECV_PORT" --zmq-send-port "$ZMQ_SEND_PORT" \
  --zmq-bind-host 0.0.0.0 --chunk-size 50 --temporal-agg \
  --multitask-hand-state measured --multitask-hand-feedback measured \
  --act-image-color-order rgb
```

The runner is an inference service only. Training requires the public dataset
from the official ModelScope project and is not bundled here.

## Evidence boundary

The fixed-split R5 development result reduced normalized-action L1 from
`0.138117` to `0.126258` on 101 held-out episodes. This is an offline
action-reconstruction metric, not a task-success rate. The reported official
25-inference scores and the local video provenance are documented separately
in `RESULTS.md` and `VIDEOS.md`.

## Security and provenance

Do not commit private keys, platform credentials, raw benchmark endpoints, or
private evaluation links. This snapshot contains no SSH key material.

The benchmark and dataset are maintained by Open-X-Humanoid. See
`UPSTREAM_NOTICE.md` for source and dataset references.
