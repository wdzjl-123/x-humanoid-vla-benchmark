# R7 Inference Package

This directory contains the inference-only Docker source and the R7 model
files. The checkpoint is one shared RGB-D task-conditioned ACT policy for all
five benchmark task names.

Build and run with simulator-specific host and port values supplied by the
operator:

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

Run `bash scripts/verify_submission.sh` before deployment. The model files
are kept outside the image and mounted read-only; `run_model.sh` copies the
selected pair to a temporary writable directory for its audit log.
