"""RGB-D, pose, and task-conditioned dataset for one unified ACT checkpoint."""
from __future__ import annotations

import bisect
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.multitask import (
    ACTION_DIM,
    EEF_POSE_DIM,
    JOINT_STATE_DIM,
    MULTITASK_STATE_DIM,
    TASK_NAMES,
    depth_to_meters,
    task_one_hot,
)


JOINT_KEYS = (
    "arm_left_position_align",
    "end_effector_left_position_align",
    "arm_right_position_align",
    "end_effector_right_position_align",
)
POSE_KEYS = (
    "end_effector_left_pose_align",
    "end_effector_right_pose_align",
)
CAMERA_NAME = "camera_head"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class EpisodeRef:
    task_name: str
    path: str
    length: int


@dataclass(frozen=True)
class RobustAugmentationConfig:
    """Training-only perturbations that preserve RGB/depth pixel alignment."""

    translate_fraction: float = 0.03
    scale_jitter: float = 0.03
    depth_noise_std: float = 0.01
    depth_dropout_probability: float = 0.01
    normalized_state_noise_std: float = 0.01


def build_task_schedule(
    task_names: tuple[str, ...],
    task_sampling_weights: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Build a deterministic task schedule with optional integer weights."""
    if not task_names:
        raise ValueError("task_names must not be empty")
    if task_sampling_weights is None:
        return task_names

    unknown = set(task_sampling_weights) - set(task_names)
    missing = set(task_names) - set(task_sampling_weights)
    if unknown or missing:
        raise ValueError(
            "task_sampling_weights must contain exactly the selected tasks; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )

    schedule: list[str] = []
    for task_name in task_names:
        weight = task_sampling_weights[task_name]
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"Task sampling weight for {task_name!r} must be a positive integer")
        schedule.extend([task_name] * weight)
    return tuple(schedule)


def _episode_refs(task_name: str, paths: list[str]) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for path in paths:
        try:
            refs.append(EpisodeRef(task_name, path, _episode_length(path)))
        except Exception as exc:
            print(f"[multitask_dataset] Skip unreadable episode {path}: {exc}")
    if not refs:
        raise RuntimeError(f"No readable trajectories found for {task_name}")
    return refs


def create_episode_split_manifest(
    data_root: str,
    task_names: tuple[str, ...] = TASK_NAMES,
    split_seed: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """Create a deterministic, episode-level split without moving HDF5 files."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must be positive and sum to 1")

    root = Path(data_root).resolve()
    tasks: dict[str, dict[str, list[str]]] = {}
    for task_name in task_names:
        files = sorted(glob.glob(str(root / task_name / "train" / "*.hdf5")))
        if len(files) < len(SPLIT_NAMES):
            raise ValueError(f"Need at least {len(SPLIT_NAMES)} episodes for {task_name}, found {len(files)}")

        rng = np.random.default_rng(split_seed + TASK_NAMES.index(task_name))
        shuffled = [files[index] for index in rng.permutation(len(files))]
        val_count = max(1, round(len(files) * val_ratio))
        test_count = max(1, round(len(files) * test_ratio))
        train_count = len(files) - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Split leaves no training episodes for {task_name}")

        partition = {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:train_count + val_count],
            "test": shuffled[train_count + val_count:],
        }
        tasks[task_name] = {
            split_name: sorted(str(Path(path).resolve().relative_to(root)) for path in paths)
            for split_name, paths in partition.items()
        }

    return {
        "version": 1,
        "data_root": str(Path(data_root)),
        "split_seed": split_seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "tasks": tasks,
    }


def write_episode_split_manifest(manifest: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_episode_split_manifest(
    data_root: str,
    manifest_path: str,
    task_names: tuple[str, ...] = TASK_NAMES,
) -> dict[str, list[EpisodeRef]]:
    """Load and validate a manifest produced by ``create_episode_split_manifest``."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read split manifest {path}: {exc}") from exc
    if manifest.get("version") != 1 or not isinstance(manifest.get("tasks"), dict):
        raise ValueError(f"Unsupported split manifest format: {path}")

    root = Path(data_root).resolve()
    refs_by_split: dict[str, list[EpisodeRef]] = {split_name: [] for split_name in SPLIT_NAMES}
    seen_paths: set[str] = set()
    for task_name in task_names:
        task_splits = manifest["tasks"].get(task_name)
        if not isinstance(task_splits, dict):
            raise ValueError(f"Split manifest is missing task {task_name!r}")
        for split_name in SPLIT_NAMES:
            relative_paths = task_splits.get(split_name)
            if not isinstance(relative_paths, list) or not relative_paths:
                raise ValueError(f"Split manifest has no {split_name!r} episodes for {task_name!r}")
            resolved_paths: list[str] = []
            for relative_path in relative_paths:
                if not isinstance(relative_path, str):
                    raise ValueError(f"Invalid episode path for {task_name!r}/{split_name!r}")
                candidate = (root / relative_path).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Episode escapes data root: {relative_path}") from exc
                if not candidate.is_file():
                    raise FileNotFoundError(f"Episode from manifest does not exist: {candidate}")
                canonical_path = str(candidate)
                if canonical_path in seen_paths:
                    raise ValueError(f"Episode appears in multiple manifest partitions: {candidate}")
                seen_paths.add(canonical_path)
                resolved_paths.append(canonical_path)
            refs_by_split[split_name].extend(_episode_refs(task_name, resolved_paths))
    return refs_by_split


def _dataset_path(key: str) -> str:
    return f"puppet/{key}/data"


def _episode_length(path: str) -> int:
    with h5py.File(path, "r", libver="latest") as root:
        return int(root[_dataset_path(JOINT_KEYS[0])].shape[0])


def _read_joint_chunk(root: h5py.File, start: int, chunk_size: int | None) -> np.ndarray:
    parts = []
    for key in JOINT_KEYS:
        data = root[_dataset_path(key)]
        values = data[start:] if chunk_size is None else data[start:start + chunk_size]
        parts.append(np.asarray(values, dtype=np.float32))
    result = np.concatenate(parts, axis=-1)
    if result.shape[-1] != JOINT_STATE_DIM:
        raise ValueError(f"Expected {JOINT_STATE_DIM} joint values, got {result.shape[-1]}")
    return result


def _read_pose(root: h5py.File, start: int) -> np.ndarray:
    parts = []
    for key in POSE_KEYS:
        values = np.asarray(root[_dataset_path(key)][start], dtype=np.float32).ravel()
        if values.size != 7:
            raise ValueError(f"Expected seven values for {key}, got {values.size}")
        parts.append(values)
    pose = np.concatenate(parts, axis=-1)
    if pose.size != EEF_POSE_DIM:
        raise ValueError(f"Expected {EEF_POSE_DIM} pose values, got {pose.size}")
    return np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)


def _decode_rgb(encoded: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode RGB image")
    bgr = cv2.resize(bgr, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    return (bgr[:, :, ::-1].copy() / 255.0).astype(np.float32)


def _decode_depth(
    encoded: np.ndarray,
    image_width: int,
    image_height: int,
    max_depth_meters: float,
) -> np.ndarray:
    raw_depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise ValueError("Unable to decode depth image")
    depth = depth_to_meters(raw_depth, max_depth_meters)
    return cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


def _state_vector(root: h5py.File, task_name: str, step: int) -> np.ndarray:
    joints = _read_joint_chunk(root, step, 1)[0]
    pose = _read_pose(root, step)
    state = np.concatenate([joints, pose, task_one_hot(task_name)])
    if state.size != MULTITASK_STATE_DIM:
        raise AssertionError(f"Expected state dim {MULTITASK_STATE_DIM}, got {state.size}")
    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _split_task_episodes(
    task_name: str,
    task_dir: str,
    val_ratio: float,
    split_seed: int,
    max_episodes: int | None,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    files = sorted(glob.glob(os.path.join(task_dir, "train", "*.hdf5")))
    if max_episodes is not None:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No HDF5 trajectories found for {task_name} under {task_dir}/train")

    refs = _episode_refs(task_name, files)

    if val_ratio <= 0 or len(refs) == 1:
        return refs, []
    val_count = min(len(refs) - 1, max(1, round(len(refs) * val_ratio)))
    rng = np.random.default_rng(split_seed + TASK_NAMES.index(task_name))
    val_indices = set(rng.choice(len(refs), size=val_count, replace=False).tolist())
    train_refs = [ref for index, ref in enumerate(refs) if index not in val_indices]
    val_refs = [ref for index, ref in enumerate(refs) if index in val_indices]
    return train_refs, val_refs


def build_episode_splits(
    data_root: str,
    task_names: tuple[str, ...] = TASK_NAMES,
    val_ratio: float = 0.2,
    split_seed: int = 1,
    max_episodes_per_task: int | None = None,
    split_manifest: str | None = None,
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    unknown = set(task_names) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"Unknown task names: {sorted(unknown)}")
    if split_manifest is not None:
        split_refs = load_episode_split_manifest(data_root, split_manifest, task_names)
        for task_name in task_names:
            print(
                f"[multitask_dataset] {task_name}: "
                f"train={sum(ref.task_name == task_name for ref in split_refs['train'])} "
                f"val={sum(ref.task_name == task_name for ref in split_refs['val'])} "
                f"test={sum(ref.task_name == task_name for ref in split_refs['test'])}"
            )
        return split_refs["train"], split_refs["val"]
    train_refs: list[EpisodeRef] = []
    val_refs: list[EpisodeRef] = []
    for task_name in task_names:
        train_task, val_task = _split_task_episodes(
            task_name,
            str(Path(data_root) / task_name),
            val_ratio,
            split_seed,
            max_episodes_per_task,
        )
        print(
            f"[multitask_dataset] {task_name}: "
            f"train={len(train_task)} val={len(val_task)}"
        )
        train_refs.extend(train_task)
        val_refs.extend(val_task)
    return train_refs, val_refs


def compute_multitask_stats(episodes: list[EpisodeRef]) -> dict:
    """Stream global train-only state/action statistics without loading images."""
    if not episodes:
        raise ValueError("Cannot compute statistics without training episodes")
    state_sum = np.zeros(MULTITASK_STATE_DIM, dtype=np.float64)
    state_square_sum = np.zeros(MULTITASK_STATE_DIM, dtype=np.float64)
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_square_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    state_min = np.full(MULTITASK_STATE_DIM, np.inf, dtype=np.float64)
    state_max = np.full(MULTITASK_STATE_DIM, -np.inf, dtype=np.float64)
    action_min = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    count = 0

    for episode in episodes:
        with h5py.File(episode.path, "r", libver="latest") as root:
            actions = _read_joint_chunk(root, 0, None)
            poses = np.concatenate(
                [np.asarray(root[_dataset_path(key)][:], dtype=np.float32) for key in POSE_KEYS],
                axis=-1,
            )
        if actions.shape[0] != poses.shape[0]:
            raise ValueError(f"Mismatched joint and pose lengths in {episode.path}")
        one_hot = np.broadcast_to(task_one_hot(episode.task_name), (len(actions), len(TASK_NAMES)))
        states = np.concatenate([actions, poses, one_hot], axis=-1)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        state_sum += states.sum(axis=0)
        state_square_sum += np.square(states).sum(axis=0)
        action_sum += actions.sum(axis=0)
        action_square_sum += np.square(actions).sum(axis=0)
        state_min = np.minimum(state_min, states.min(axis=0))
        state_max = np.maximum(state_max, states.max(axis=0))
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        count += len(states)

    if count == 0:
        raise RuntimeError("No aligned state samples found")
    state_mean = state_sum / count
    action_mean = action_sum / count
    state_var = np.maximum(state_square_sum / count - np.square(state_mean), 1e-4)
    action_var = np.maximum(action_square_sum / count - np.square(action_mean), 1e-4)
    epsilon = 1e-4
    return {
        "qpos_mean": state_mean.astype(np.float32),
        "qpos_std": np.sqrt(state_var).astype(np.float32),
        "qpos_min": (state_min - epsilon).astype(np.float32),
        "qpos_max": (state_max + epsilon).astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.sqrt(action_var).astype(np.float32),
        "action_min": (action_min - epsilon).astype(np.float32),
        "action_max": (action_max + epsilon).astype(np.float32),
        "task_names": tuple(TASK_NAMES),
        "state_dim": MULTITASK_STATE_DIM,
        "action_dim": ACTION_DIM,
    }


class MultiTaskEpisodicDataset(Dataset):
    """Balanced episode sampler over all task names for one shared ACT model."""

    def __init__(
        self,
        episodes: list[EpisodeRef],
        norm_stats: dict,
        chunk_size: int,
        image_width: int,
        image_height: int,
        max_depth_meters: float,
        use_aug: bool = False,
        augmentation_config: RobustAugmentationConfig | None = None,
        motion_sampling_alpha: float = 0.0,
        motion_sampling_max_ratio: float = 5.0,
        task_sampling_weights: Mapping[str, int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if motion_sampling_alpha < 0:
            raise ValueError("motion_sampling_alpha must be non-negative")
        if motion_sampling_max_ratio < 1:
            raise ValueError("motion_sampling_max_ratio must be at least one")
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.image_width = image_width
        self.image_height = image_height
        self.max_depth_meters = max_depth_meters
        self.use_aug = use_aug
        self.augmentation_config = augmentation_config if use_aug else None
        self.motion_sampling_alpha = motion_sampling_alpha
        self.motion_sampling_max_ratio = motion_sampling_max_ratio
        self.task_names = tuple(sorted({episode.task_name for episode in episodes}, key=TASK_NAMES.index))
        self.task_schedule = build_task_schedule(self.task_names, task_sampling_weights)
        self.by_task: dict[str, list[EpisodeRef]] = {task_name: [] for task_name in self.task_names}
        for episode in episodes:
            self.by_task[episode.task_name].append(episode)
        self.cumulative_lengths: dict[str, list[int]] = {}
        self.total_lengths: dict[str, int] = {}
        for task_name, refs in self.by_task.items():
            cumulative = np.cumsum([episode.length for episode in refs]).astype(int).tolist()
            self.cumulative_lengths[task_name] = cumulative
            self.total_lengths[task_name] = cumulative[-1]
        self.length = max(self.total_lengths.values()) * len(self.task_schedule)

        self.motion_cdfs: dict[str, np.ndarray] = {}
        if motion_sampling_alpha > 0:
            self._build_motion_cdfs()

        if use_aug:
            import torchvision.transforms as transforms
            self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return self.length

    def _build_motion_cdfs(self) -> None:
        """Bias train sampling toward contact and transition phases, per task."""
        for task_name, refs in self.by_task.items():
            weights_by_episode: list[np.ndarray] = []
            for episode in refs:
                with h5py.File(episode.path, "r", libver="latest") as root:
                    actions = _read_joint_chunk(root, 0, None)
                motion = np.zeros(len(actions), dtype=np.float64)
                if len(actions) > 1:
                    motion[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
                positive = motion[motion > 1e-6]
                scale = float(np.median(positive)) if positive.size else 1.0
                normalized_motion = np.clip(motion / max(scale, 1e-6), 0.0, self.motion_sampling_max_ratio)
                weights_by_episode.append(1.0 + self.motion_sampling_alpha * normalized_motion)
            self.motion_cdfs[task_name] = np.cumsum(np.concatenate(weights_by_episode))

    def _locate(self, index: int) -> tuple[EpisodeRef, int]:
        task_name = self.task_schedule[index % len(self.task_schedule)]
        if task_name in self.motion_cdfs:
            cdf = self.motion_cdfs[task_name]
            sampled = np.random.random() * float(cdf[-1])
            offset = min(int(np.searchsorted(cdf, sampled, side="right")), self.total_lengths[task_name] - 1)
        else:
            offset = (index // len(self.task_schedule)) % self.total_lengths[task_name]
        cumulative = self.cumulative_lengths[task_name]
        episode_index = bisect.bisect_right(cumulative, offset)
        previous_end = 0 if episode_index == 0 else cumulative[episode_index - 1]
        return self.by_task[task_name][episode_index], offset - previous_end

    def _augment_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        config = self.augmentation_config
        if config is None:
            return rgb, depth

        height, width = depth.shape
        translate_x = float(np.random.uniform(-config.translate_fraction, config.translate_fraction) * width)
        translate_y = float(np.random.uniform(-config.translate_fraction, config.translate_fraction) * height)
        scale = float(np.random.uniform(1.0 - config.scale_jitter, 1.0 + config.scale_jitter))
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, scale)
        matrix[:, 2] += (translate_x, translate_y)
        rgb = cv2.warpAffine(
            rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        depth = cv2.warpAffine(
            depth,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        valid_depth = depth > 0
        if config.depth_noise_std > 0 and np.any(valid_depth):
            noise = np.random.normal(0.0, config.depth_noise_std, depth.shape).astype(np.float32)
            depth = depth.copy()
            depth[valid_depth] += noise[valid_depth]
        if config.depth_dropout_probability > 0 and np.any(valid_depth):
            dropout = (np.random.random(depth.shape) < config.depth_dropout_probability) & valid_depth
            depth = depth.copy()
            depth[dropout] = 0.0
        return rgb, np.clip(depth, 0.0, self.max_depth_meters).astype(np.float32, copy=False)

    def __getitem__(self, index: int):
        episode, step = self._locate(index)
        with h5py.File(episode.path, "r", libver="latest") as root:
            state = _state_vector(root, episode.task_name, step)
            action_chunk = _read_joint_chunk(root, step, self.chunk_size)
            rgb = _decode_rgb(
                root[f"camera_observations/color_images/{CAMERA_NAME}"][step],
                self.image_width,
                self.image_height,
            )
            depth = _decode_depth(
                root[f"camera_observations/depth_images/{CAMERA_NAME}"][step],
                self.image_width,
                self.image_height,
                self.max_depth_meters,
            )

        rgb, depth = self._augment_rgbd(rgb, depth)

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if len(action_chunk) < self.chunk_size:
            padded = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
            padded[:len(action_chunk)] = action_chunk
            is_pad[len(action_chunk):] = True
            action_chunk = padded

        normalized_state = (state - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.augmentation_config is not None and self.augmentation_config.normalized_state_noise_std > 0:
            normalized_state = normalized_state + np.random.normal(
                0.0,
                self.augmentation_config.normalized_state_noise_std,
                normalized_state.shape,
            ).astype(np.float32)
        normalized_action = (
            (action_chunk - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        )
        image_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        if self.color_jitter is not None:
            image_t = self.color_jitter(image_t)
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        return (
            image_t,
            depth_t,
            torch.from_numpy(normalized_state).float(),
            torch.from_numpy(normalized_action).float(),
            torch.from_numpy(is_pad).bool(),
        )


def build_multitask_dataloaders(
    data_root: str,
    batch_size_train: int,
    batch_size_val: int,
    chunk_size: int,
    image_width: int,
    image_height: int,
    max_depth_meters: float,
    task_names: tuple[str, ...] = TASK_NAMES,
    val_ratio: float = 0.2,
    split_seed: int = 1,
    use_aug: bool = False,
    augmentation_config: RobustAugmentationConfig | None = None,
    motion_sampling_alpha: float = 0.0,
    motion_sampling_max_ratio: float = 5.0,
    task_sampling_weights: Mapping[str, int] | None = None,
    num_workers: int = 4,
    max_episodes_per_task: int | None = None,
    split_manifest: str | None = None,
) -> tuple[DataLoader, DataLoader | None, dict]:
    train_refs, val_refs = build_episode_splits(
        data_root=data_root,
        task_names=task_names,
        val_ratio=val_ratio,
        split_seed=split_seed,
        max_episodes_per_task=max_episodes_per_task,
        split_manifest=split_manifest,
    )
    norm_stats = compute_multitask_stats(train_refs)
    train_dataset = MultiTaskEpisodicDataset(
        train_refs,
        norm_stats,
        chunk_size,
        image_width,
        image_height,
        max_depth_meters,
        use_aug=use_aug,
        augmentation_config=augmentation_config,
        motion_sampling_alpha=motion_sampling_alpha if use_aug else 0.0,
        motion_sampling_max_ratio=motion_sampling_max_ratio,
        task_sampling_weights=task_sampling_weights,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    if not val_refs:
        return train_loader, None, norm_stats
    val_dataset = MultiTaskEpisodicDataset(
        val_refs,
        norm_stats,
        chunk_size,
        image_width,
        image_height,
        max_depth_meters,
        use_aug=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, norm_stats
