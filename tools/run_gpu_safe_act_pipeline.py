#!/usr/bin/env python3
"""Run the approved R1 evaluation and R2 train/evaluation sequence serially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEFAULT_SPLIT = "splits/multitask_act_80_10_10_seed20260728.json"
DEFAULT_R1_DIR = "checkpoints/vla_multitask_act_rgbd_robust_weighted_mild_b160_s20260730"
DEFAULT_R2_DIR = "checkpoints/vla_multitask_act_rgbd_robust_weighted_strong_b160_s20260730"
STATUS_PATH = ROOT / "reports/gpu_safe_act_pipeline_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(stage: str, **extra: Any) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), "stage": stage, **extra}
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def run_stage(stage: str, command: list[str]) -> None:
    write_status(stage, command=command)
    print(f"[pipeline] stage={stage}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} exited with status {result.returncode}")


def eval_command(model_path: Path, output_json: Path, split_manifest: Path) -> list[str]:
    return [
        str(PYTHON),
        "tools/eval_multitask_act.py",
        "--data-root", "data",
        "--split-manifest", str(split_manifest),
        "--split", "test",
        "--model-path", str(model_path),
        "--batch-size", "128",
        "--num-workers", "8",
        "--output-json", str(output_json),
    ]


def r2_train_command(r2_dir: Path, split_manifest: Path) -> list[str]:
    return [
        str(PYTHON),
        "-u", "tools/train/train_multitask_act.py",
        "--data-root", "data",
        "--ckpt-dir", str(r2_dir),
        "--split-manifest", str(split_manifest),
        "--num-steps", "16800",
        "--batch-size", "160",
        "--num-workers", "8",
        "--validate-every", "1000",
        "--max-val-batches", "0",
        "--save-every", "4000",
        "--log-every", "100",
        "--use-aug",
        "--task-sampling-weights", "2", "3", "3", "3", "2",
        "--device", "cuda",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial R1-eval -> R2-train -> R2-eval controller")
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT)
    parser.add_argument("--r1-dir", default=DEFAULT_R1_DIR)
    parser.add_argument("--r2-dir", default=DEFAULT_R2_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_manifest = (ROOT / args.split_manifest).resolve()
    r1_dir = (ROOT / args.r1_dir).resolve()
    r2_dir = (ROOT / args.r2_dir).resolve()
    r1_model = r1_dir / "agent_best.ckpt"
    r2_model = r2_dir / "agent_best.ckpt"
    r2_last = r2_dir / "policy_last.ckpt"
    r1_report = ROOT / "reports/multitask_act_robust_weighted_mild_b160_test.json"
    r2_report = ROOT / "reports/multitask_act_robust_weighted_strong_b160_test.json"

    try:
        if not split_manifest.is_file():
            raise FileNotFoundError(f"split manifest is missing: {split_manifest}")
        if not r1_model.is_file():
            raise FileNotFoundError(f"R1 best checkpoint is missing: {r1_model}")

        if not r1_report.is_file():
            run_stage("r1_test", eval_command(r1_model, r1_report, split_manifest))

        if not r2_last.is_file():
            if r2_dir.exists() and any(r2_dir.iterdir()):
                raise RuntimeError(
                    f"R2 directory already contains incomplete output: {r2_dir}; "
                    "refusing to overwrite or resume automatically"
                )
            run_stage("r2_training", r2_train_command(r2_dir, split_manifest))

        if not r2_model.is_file():
            raise FileNotFoundError(f"R2 best checkpoint is missing after training: {r2_model}")
        if not r2_report.is_file():
            run_stage("r2_test", eval_command(r2_model, r2_report, split_manifest))

        write_status(
            "awaiting_promotion",
            r1_report=str(r1_report.relative_to(ROOT)),
            r2_report=str(r2_report.relative_to(ROOT)),
        )
        print("[pipeline] R1 and R2 reports are ready; promotion comparison is required before final training", flush=True)
    except BaseException as exc:
        write_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
