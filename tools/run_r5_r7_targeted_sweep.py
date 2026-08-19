#!/usr/bin/env python3
"""Train and evaluate the approved R5/R6 ACT sweep, then train R7 on all data."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SPLIT_MANIFEST = ROOT / "splits/multitask_act_80_10_10_seed20260728.json"
STATUS_PATH = ROOT / "reports/r5_r7_targeted_sweep_status.json"
MIN_FREE_MIB = 24 * 1024
TASK_ORDER = ("ind_task_01", "ind_task_02", "ind_task_03", "lab_task_01", "lab_task_03")
HARD_TASKS = ("ind_task_02", "lab_task_01")

R3_DIR = ROOT / "checkpoints/vla_multitask_act_rgbd_robust_weighted_mild_b64_s20260730"
R3_REPORT = ROOT / "reports/multitask_act_robust_weighted_mild_b64_test.json"
R5_DIR = ROOT / "checkpoints/vla_multitask_act_rgbd_robust_targeted_r5_b64_s20260730"
R5_REPORT = ROOT / "reports/multitask_act_robust_targeted_r5_b64_test.json"
R6_DIR = ROOT / "checkpoints/vla_multitask_act_rgbd_robust_targeted_r6_finetune_b64_s20260730"
R6_REPORT = ROOT / "reports/multitask_act_robust_targeted_r6_finetune_b64_test.json"
R7_DIR = ROOT / "checkpoints/vla_multitask_act_rgbd_robust_targeted_r7_full_s20260731"

R3_WEIGHTS = (1, 2, 2, 2, 1)
TARGETED_WEIGHTS = (1, 3, 2, 3, 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(stage: str, **extra: Any) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), "stage": stage, **extra}
    temporary = STATUS_PATH.with_suffix(STATUS_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def gpu_free_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one GPU memory value, got {values!r}")
    return int(values[0])


def require_gpu_headroom(stage: str) -> None:
    free_mib = gpu_free_mib()
    if free_mib < MIN_FREE_MIB:
        raise RuntimeError(
            f"{stage} requires {MIN_FREE_MIB} MiB free GPU memory, found {free_mib} MiB"
        )


def run_stage(stage: str, command: list[str]) -> None:
    require_gpu_headroom(stage)
    write_status(stage, command=command, gpu_free_mib=gpu_free_mib())
    print(f"[r5-r7] stage={stage}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} exited with status {result.returncode}")


def completed(directory: Path, *, require_best: bool = True) -> bool:
    last = directory / "policy_last.ckpt"
    best = directory / "agent_best.ckpt"
    return last.is_file() and (best.is_file() if require_best else True)


def refuse_incomplete(directory: Path, *, require_best: bool = True) -> None:
    if directory.exists() and any(directory.iterdir()) and not completed(directory, require_best=require_best):
        raise RuntimeError(f"incomplete output exists at {directory}; refusing to overwrite or resume automatically")


def train_command(
    directory: Path,
    *,
    num_steps: int,
    weights: tuple[int, ...],
    split_manifest: Path | None,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    resume: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        "-u",
        "tools/train/train_multitask_act.py",
        "--data-root", "data",
        "--ckpt-dir", str(directory),
        "--num-steps", str(num_steps),
        "--batch-size", "64",
        "--num-workers", "8",
        "--validate-every", "1000",
        "--max-val-batches", "0",
        "--save-every", "4000",
        "--log-every", "100",
        "--lr", str(lr),
        "--lr-backbone", str(lr_backbone),
        "--use-aug",
        "--task-sampling-weights", *(str(weight) for weight in weights),
        "--device", "cuda",
    ]
    if split_manifest is None:
        command.extend(["--val-ratio", "0"])
    else:
        command.extend(["--split-manifest", str(split_manifest)])
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def eval_command(model_path: Path, output_path: Path) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        "tools/eval_multitask_act.py",
        "--data-root", "data",
        "--split-manifest", str(SPLIT_MANIFEST),
        "--split", "test",
        "--model-path", str(model_path),
        "--batch-size", "128",
        "--num-workers", "8",
        "--seed", "20260728",
        "--device", "cuda",
        "--output-json", str(output_path),
    ]


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid report payload: {path}")
    return payload


def l1(report: dict[str, Any], task_name: str | None = None) -> float:
    section = report["summary"] if task_name is None else report["per_task"][task_name]
    return float(section["normalized_action_l1"])


def eligible(report: dict[str, Any], reference: dict[str, Any]) -> bool:
    if l1(report) >= l1(reference):
        return False
    return all(l1(report, task) <= l1(reference, task) * 1.02 for task in HARD_TASKS)


def choose_recipe() -> tuple[str, tuple[int, ...], dict[str, Any]]:
    reference = load_report(R3_REPORT)
    candidates = [("r3", R3_WEIGHTS, reference)]
    for name, weights, report_path in (
        ("r5", TARGETED_WEIGHTS, R5_REPORT),
        ("r6", TARGETED_WEIGHTS, R6_REPORT),
    ):
        report = load_report(report_path)
        if eligible(report, reference):
            candidates.append((name, weights, report))
    return min(candidates, key=lambda item: l1(item[2]))


def main() -> None:
    try:
        if not SPLIT_MANIFEST.is_file():
            raise FileNotFoundError(f"split manifest is missing: {SPLIT_MANIFEST}")
        if not (R3_DIR / "agent_best.ckpt").is_file() or not R3_REPORT.is_file():
            raise FileNotFoundError("R3 checkpoint or fixed-test report is missing")

        if not completed(R5_DIR):
            refuse_incomplete(R5_DIR)
            run_stage("r5_training", train_command(
                R5_DIR,
                num_steps=16800,
                weights=TARGETED_WEIGHTS,
                split_manifest=SPLIT_MANIFEST,
            ))
        if not R5_REPORT.is_file():
            run_stage("r5_test", eval_command(R5_DIR / "agent_best.ckpt", R5_REPORT))

        if not completed(R6_DIR):
            refuse_incomplete(R6_DIR)
            run_stage("r6_training", train_command(
                R6_DIR,
                num_steps=19000,
                weights=TARGETED_WEIGHTS,
                split_manifest=SPLIT_MANIFEST,
                lr=3e-5,
                lr_backbone=3e-6,
                resume=R3_DIR / "agent_best.ckpt",
            ))
        if not R6_REPORT.is_file():
            run_stage("r6_test", eval_command(R6_DIR / "agent_best.ckpt", R6_REPORT))

        winner, weights, report = choose_recipe()
        write_status(
            "r7_selected",
            winner=winner,
            selected_weights=list(weights),
            selected_test_l1=l1(report),
            r5_report=str(R5_REPORT.relative_to(ROOT)),
            r6_report=str(R6_REPORT.relative_to(ROOT)),
        )
        if not completed(R7_DIR, require_best=False):
            refuse_incomplete(R7_DIR, require_best=False)
            run_stage("r7_full_training", train_command(
                R7_DIR,
                num_steps=21000,
                weights=weights,
                split_manifest=None,
            ))
        write_status(
            "completed",
            winner=winner,
            selected_weights=list(weights),
            selected_test_l1=l1(report),
            final_model=str((R7_DIR / "policy_last.ckpt").relative_to(ROOT)),
        )
        print(f"[r5-r7] completed with {winner} recipe", flush=True)
    except BaseException as exc:
        write_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
