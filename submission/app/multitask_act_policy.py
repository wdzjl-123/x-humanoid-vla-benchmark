"""Inference policy for the single-checkpoint, task-conditioned RGB-D ACT."""
from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
import torch

from tools.multitask import MULTITASK_STATE_DIM, TASK_NAMES, depth_to_meters, task_one_hot
from tools.policies.act_policy import ACTPolicy, build_act_model


class MultiTaskACTPolicy(ACTPolicy):
    """One ACT network conditioned on the benchmark task name.

    The process still announces one task to the benchmark through the normal
    runner handshake. Crucially, all task names load the same checkpoint; task
    identity is an input feature rather than a selector for separate weights.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        task_name: str = "",
        hand_state_mode: str = "measured",
        hand_feedback: str = "measured",
        **kwargs,
    ):
        if task_name not in TASK_NAMES:
            raise ValueError(
                "multitask_act requires --task set to one of "
                f"{TASK_NAMES}, got {task_name!r}"
            )
        self.max_depth_meters = 5.0
        self.checkpoint_task_names: tuple[str, ...] = ()
        super().__init__(
            model_path,
            device=device,
            task_name=task_name,
            hand_state_mode=hand_state_mode,
            hand_feedback=hand_feedback,
            **kwargs,
        )

    def _open_debug_log(self):
        log_path = os.path.join(os.path.dirname(self.model_path), "multitask_act_debug.log")
        self._dbg_file = open(log_path, "w", buffering=1)
        print(f"[MultiTaskACTPolicy] Debug log -> {log_path}")

    def _load_model(self):
        checkpoint_dir = os.path.dirname(self.model_path)
        self._open_debug_log()
        stats_path = os.path.join(checkpoint_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found in {checkpoint_dir}")
        with open(stats_path, "rb") as handle:
            self.norm_stats = pickle.load(handle)
        if len(self.norm_stats["qpos_mean"]) != MULTITASK_STATE_DIM:
            raise ValueError(
                "multitask_act requires 45-dimensional unified state statistics; "
                f"got {len(self.norm_stats['qpos_mean'])}"
            )

        checkpoint = torch.load(self.model_path, map_location="cpu")
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("multitask_act checkpoint is missing its model config")
        self.checkpoint_task_names = tuple(config.get("task_names", ()))
        if self.task_name not in self.checkpoint_task_names:
            raise ValueError(
                f"Task {self.task_name!r} is not present in checkpoint tasks "
                f"{self.checkpoint_task_names}"
            )
        if int(config.get("state_dim", -1)) != MULTITASK_STATE_DIM:
            raise ValueError("Checkpoint state_dim is incompatible with multitask_act")
        if not config.get("use_depth_image", False):
            raise ValueError("multitask_act requires a depth-enabled checkpoint")

        self.chunk_size = int(config["chunk_size"])
        self.model_chunk_size = self.chunk_size
        self.IMG_W = int(config["img_w"])
        self.IMG_H = int(config["img_h"])
        self.max_depth_meters = float(config.get("max_depth_meters", self.max_depth_meters))
        policy = build_act_model(
            chunk_size=self.chunk_size,
            backbone=str(config.get("backbone", "resnet18")),
            hidden_dim=int(config.get("hidden_dim", 512)),
            dim_feedforward=int(config.get("dim_feedforward", 3200)),
            enc_layers=int(config.get("enc_layers", 4)),
            dec_layers=int(config.get("dec_layers", 7)),
            nheads=int(config.get("nheads", 8)),
            action_dim=int(config.get("action_dim", 26)),
            state_dim=MULTITASK_STATE_DIM,
            kl_weight=int(config.get("kl_weight", 10)),
            lr=float(config.get("lr", 1e-4)),
            lr_backbone=float(config.get("lr_backbone", 1e-5)),
            device=self.device,
            use_depth_image=True,
        )
        policy.deserialize(checkpoint["nets"])
        policy.eval()
        policy.to(self.device)
        print(
            f"[MultiTaskACTPolicy] Loaded {self.model_path} "
            f"(step {checkpoint.get('step', '?')}, task={self.task_name}, "
            f"tasks={self.checkpoint_task_names})"
        )
        self._init_buffers()
        return policy

    @staticmethod
    def _read_pose(obs: dict, key: str) -> np.ndarray:
        try:
            pose = np.asarray(obs["puppet"][key]["data"], dtype=np.float32).ravel()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing required end-effector pose {key}") from exc
        if pose.size != 7:
            raise ValueError(f"Expected seven values for {key}, got {pose.size}")
        return np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_qpos(self, obs: dict) -> np.ndarray:
        joints = super()._get_qpos(obs)
        left_pose = self._read_pose(obs, "end_effector_left_pose_raw")
        right_pose = self._read_pose(obs, "end_effector_right_pose_raw")
        state = np.concatenate([joints, left_pose, right_pose, task_one_hot(self.task_name)])
        if state.size != MULTITASK_STATE_DIM:
            raise AssertionError(f"Expected {MULTITASK_STATE_DIM}-dimensional state, got {state.size}")
        return state.astype(np.float32)

    def _get_depth(self, obs: dict) -> torch.Tensor:
        try:
            raw_depth = obs["camera_observations"]["depth_images"][self.CAM_NAME]
        except (KeyError, TypeError) as exc:
            raise ValueError("multitask_act requires camera depth_images") from exc
        depth = depth_to_meters(raw_depth, self.max_depth_meters)
        if depth.ndim != 2:
            raise ValueError(f"Expected two-dimensional depth image, got shape {depth.shape}")
        depth = cv2.resize(depth, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()

    def _normalize_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        if qpos.size != MULTITASK_STATE_DIM:
            raise ValueError(f"Expected {MULTITASK_STATE_DIM} state values, got {qpos.size}")
        normalized = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return torch.from_numpy(normalized).float().unsqueeze(0)
