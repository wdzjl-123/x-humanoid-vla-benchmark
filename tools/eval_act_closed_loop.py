"""Offline hand-closed-loop replay for ACT inference candidates.

The replay keeps recorded camera frames and arm positions, but feeds the
policy's own hand command into subsequent observations when commanded feedback
is enabled. It therefore exposes inference-state drift and action-selection
differences without claiming to simulate object dynamics.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.policies.act_policy import ACTPolicy
from tools.train.dataset import _concat_qpos
from tools.train.h5_reader import H5Reader


ACTION_GROUPS = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 13),
    "right_arm": slice(13, 20),
    "right_hand": slice(20, 26),
}


class QuietACTPolicy(ACTPolicy):
    """ACT policy variant used for replay without touching checkpoint logs."""

    def _open_debug_log(self):
        self._dbg_file = None

    def _dbg(self, msg: str):
        del msg


def build_replay_observation(image_dict: dict, ctrl: dict, camera_name: str) -> tuple[dict, np.ndarray]:
    """Build an inference observation from one aligned HDF5 timestep."""
    target = _concat_qpos(ctrl["puppet"])[0].astype(np.float32)
    bgr = image_dict["color_images"][camera_name]
    rgb = bgr[:, :, ::-1].copy()
    obs = {
        "puppet": {
            "arm_left_position_raw": {"data": target[:7]},
            "end_effector_left_position_raw": {"data": target[7:13]},
            "arm_right_position_raw": {"data": target[13:20]},
            "end_effector_right_position_raw": {"data": target[20:26]},
        },
        "camera_observations": {"color_images": {camera_name: rgb}},
    }
    return obs, target


def summarize_actions(
    actions: np.ndarray,
    targets: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict:
    error = np.abs(actions - targets)
    group_mae = {
        name: float(error[:, dims].mean())
        for name, dims in ACTION_GROUPS.items()
    }
    if len(actions) > 1:
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        action_delta_mean = float(deltas.mean())
        action_delta_p95 = float(np.percentile(deltas, 95))
    else:
        action_delta_mean = 0.0
        action_delta_p95 = 0.0
    out_of_bounds = (actions < action_min) | (actions > action_max)
    return {
        "steps": int(len(actions)),
        "mae": float(error.mean()),
        "group_mae": group_mae,
        "action_delta_mean": action_delta_mean,
        "action_delta_p95": action_delta_p95,
        "out_of_bounds_fraction": float(out_of_bounds.mean()),
    }


def evaluate_episode(
    policy: QuietACTPolicy,
    reader: H5Reader,
    episode_path: str,
    max_steps: int,
) -> dict:
    episode_len = reader.episode_length(episode_path)
    steps = episode_len if max_steps <= 0 else min(episode_len, max_steps)
    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    policy.reset()
    started = time.perf_counter()
    for step in range(steps):
        image_dict, ctrl = reader.read(episode_path, camera_frame=step, chunk_size=1)
        obs, target = build_replay_observation(image_dict, ctrl, policy.CAM_NAME)
        action = policy.infer(obs).astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"non-finite action at {Path(episode_path).name}:{step}")
        actions.append(action)
        targets.append(target)

    metrics = summarize_actions(
        np.stack(actions),
        np.stack(targets),
        np.asarray(policy.norm_stats["action_min"], dtype=np.float32),
        np.asarray(policy.norm_stats["action_max"], dtype=np.float32),
    )
    metrics["episode"] = Path(episode_path).name
    metrics["seconds"] = round(time.perf_counter() - started, 3)
    return metrics


def aggregate_episode_metrics(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("no episodes evaluated")
    weights = np.asarray([episode["steps"] for episode in episodes], dtype=np.float64)
    summary = {
        "episodes": len(episodes),
        "steps": int(weights.sum()),
        "mae": float(np.average([episode["mae"] for episode in episodes], weights=weights)),
        "action_delta_mean": float(
            np.average([episode["action_delta_mean"] for episode in episodes], weights=weights)
        ),
        "action_delta_p95_max": float(max(episode["action_delta_p95"] for episode in episodes)),
        "out_of_bounds_fraction": float(
            np.average([episode["out_of_bounds_fraction"] for episode in episodes], weights=weights)
        ),
        "group_mae": {
            name: float(
                np.average([episode["group_mae"][name] for episode in episodes], weights=weights)
            )
            for name in ACTION_GROUPS
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ACT hand-closed-loop replay")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=3, help="Number of sorted train episodes to replay")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 evaluates each selected episode fully")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-chunk-size", type=int, default=50)
    parser.add_argument("--hand-state", choices=ACTPolicy.HAND_STATE_MODES, default="legacy_home")
    parser.add_argument("--hand-feedback", choices=ACTPolicy.HAND_FEEDBACK_MODES, default=None)
    parser.add_argument("--image-color-order", choices=ACTPolicy.IMAGE_COLOR_ORDERS, default="rgb")
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument("--temporal-agg", dest="temporal_agg", action="store_true")
    temporal.add_argument("--no-temporal-agg", dest="temporal_agg", action="store_false")
    parser.set_defaults(temporal_agg=True)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--temporal-decay", type=float, default=0.01)
    parser.add_argument("--temporal-priority", choices=ACTPolicy.TEMPORAL_PRIORITIES, default="oldest")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_paths = sorted(glob.glob(str(Path(args.task_dir) / "train" / "*.hdf5")))
    selected = episode_paths[args.episode_offset:args.episode_offset + args.episodes]
    if not selected:
        raise FileNotFoundError(f"no episodes selected under {Path(args.task_dir) / 'train'}")
    if args.max_steps < 0 or args.episode_offset < 0 or args.episodes <= 0:
        raise ValueError("episodes, episode_offset, and max_steps must be non-negative with episodes > 0")

    reader = H5Reader(camera_names=[ACTPolicy.CAM_NAME])
    policy = QuietACTPolicy(
        args.model_path,
        device=args.device,
        task_name=Path(args.task_dir).name,
        hand_state_mode=args.hand_state,
        hand_feedback=args.hand_feedback,
        image_color_order=args.image_color_order,
        debug_every=max(args.max_steps + 1, 1),
        chunk_size=args.model_chunk_size,
        temporal_agg=args.temporal_agg,
        action_horizon=args.action_horizon,
        temporal_decay=args.temporal_decay,
        temporal_priority=args.temporal_priority,
    )
    episodes = [evaluate_episode(policy, reader, path, args.max_steps) for path in selected]
    result = {
        "config": {
            "task_dir": str(Path(args.task_dir)),
            "model_path": str(Path(args.model_path)),
            "device": args.device,
            "model_chunk_size": policy.chunk_size,
            "temporal_agg": policy.temporal_agg,
            "action_horizon": policy.action_horizon,
            "temporal_decay": policy.temporal_decay,
            "temporal_priority": policy.temporal_priority,
            "hand_state": policy.hand_state_mode,
            "hand_feedback": policy.hand_feedback,
            "image_color_order": policy.image_color_order,
            "max_steps": args.max_steps,
        },
        "summary": aggregate_episode_metrics(episodes),
        "episodes": episodes,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
