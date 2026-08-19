"""Shared contracts for the single-checkpoint multi-task VLA policy."""
from __future__ import annotations

import numpy as np


TASK_NAMES = (
    "ind_task_01",
    "ind_task_02",
    "ind_task_03",
    "lab_task_01",
    "lab_task_03",
)
TASK_TO_ID = {name: index for index, name in enumerate(TASK_NAMES)}

JOINT_STATE_DIM = 26
EEF_POSE_DIM = 14
TASK_CONDITION_DIM = len(TASK_NAMES)
MULTITASK_STATE_DIM = JOINT_STATE_DIM + EEF_POSE_DIM + TASK_CONDITION_DIM
ACTION_DIM = 26


def task_one_hot(task_name: str) -> np.ndarray:
    """Return the fixed task condition used by the one shared network."""
    try:
        task_id = TASK_TO_ID[task_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark task {task_name!r}; expected one of {TASK_NAMES}"
        ) from exc
    encoded = np.zeros(TASK_CONDITION_DIM, dtype=np.float32)
    encoded[task_id] = 1.0
    return encoded


def depth_to_meters(depth: np.ndarray, max_depth_meters: float = 5.0) -> np.ndarray:
    """Convert benchmark depth in millimetres (or already-metres) to metres.

    Offline PNG depth is uint16 millimetres. The ZMQ contract describes a
    float32 millimetre map. Accepting already-metre input keeps the policy
    usable with local simulators that expose metric depth directly.
    """
    if max_depth_meters <= 0:
        raise ValueError("max_depth_meters must be positive")
    values = np.asarray(depth, dtype=np.float32)
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size and float(np.median(finite_positive)) > 20.0:
        values = values / 1000.0
    values = np.nan_to_num(values, nan=0.0, posinf=max_depth_meters, neginf=0.0)
    return np.clip(values, 0.0, max_depth_meters).astype(np.float32, copy=False)
