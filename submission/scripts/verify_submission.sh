#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

test -f Dockerfile
test -x scripts/run_model.sh
test -f models/r7/policy_last.ckpt
test -f models/r7/dataset_stats.pkl
python3 -m py_compile src/xhumanoid/runner_r7.py

if [[ -f SHA256SUMS ]]; then
    sha256sum -c SHA256SUMS
fi

printf 'Submission source and model artifacts are structurally valid.\n'
