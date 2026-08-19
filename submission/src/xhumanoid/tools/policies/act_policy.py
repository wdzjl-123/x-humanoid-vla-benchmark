"""
Self-contained ACT inference policy for the benchmark.

Uses the custom DETR-VAE implementation in tools/policies/detr/
(no lerobot dependency).

Obs format received from benchmark ZMQ (same as HDF5 dataset structure):
  obs['puppet']['arm_left_position_raw']['data']    shape (7,)
  obs['puppet']['end_effector_left_position_raw']['data']  shape (6,)  (推理不读, 用上一步命令)
  obs['puppet']['arm_right_position_raw']['data']   shape (7,)
  obs['puppet']['end_effector_right_position_raw']['data'] shape (6,)  (推理不读)
  obs['camera_observations']['color_images']['camera_head']  ndarray (H, W, 3) RGB

Action output: np.ndarray shape (26,)
  [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
"""
from __future__ import annotations
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Ensure project root is on path so tools.policies.detr is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.policies.detr.main import build_ACT_model_and_optimizer
from tools.policies.base_policy import BasePolicy

# Inference image resize is a per-policy setting: see ACTPolicy.IMG_W / IMG_H
# class attributes (runner.py sets them from the policy name + optional
# --img-w/--img-h flags), so this single file serves both 320x240 and 640x480.

# Normalization used by ACT backbone (ImageNet stats)
_IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Keys used to read state (qpos) from obs — matches eval_policy.py
QPOS_KEYS = [
    "arm_left_position_raw",
    "end_effector_left_position_raw",
    "arm_right_position_raw",
    "end_effector_right_position_raw",
]


def build_act_model(
    chunk_size: int = 50,
    camera_names: list[str] | None = None,
    backbone: str = "resnet18",
    hidden_dim: int = 512,
    dim_feedforward: int = 3200,
    enc_layers: int = 4,
    dec_layers: int = 7,
    nheads: int = 8,
    action_dim: int = 26,
    state_dim: int = 26,
    kl_weight: int = 10,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    device: str = "cuda",
    use_depth_image: bool = False,
) -> "ACTModelWrapper":
    """Build and return an ACTModelWrapper (nn.Module with configure_optimizers)."""
    if camera_names is None:
        camera_names = ["camera_head"]

    args = {
        "lr": lr,
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "num_queries": chunk_size,
        "chunk_size": chunk_size,
        "camera_names": camera_names,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "kl_weight": kl_weight,
        # unused in ACT (set defaults)
        "use_vq": False,
        "vq_class": None,
        "vq_dim": None,
        "no_encoder": False,
        "use_depth_image": use_depth_image,
        "no_sepe_backbone": False,
        "use_lang": False,
        "weight_decay": 1e-4,
        "position_embedding": "sine",
        "masks": False,
        "dilation": False,
        "dropout": 0.1,
        "pre_norm": False,
        "device": device,
    }
    return ACTModelWrapper(args)


class ACTModelWrapper(nn.Module):
    """Thin wrapper around the DETR-VAE model matching the training interface."""

    def __init__(self, args: dict) -> None:
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args["kl_weight"]
        self.num_queries = args["num_queries"]

    def __call__(self, qpos, image, depth_image=None, actions=None, is_pad=None,
                 vq_sample=None, language_distilbert=None, logger=None):
        env_state = None
        image = _IMAGENET_NORMALIZE(image)
        if depth_image is not None:
            depth_image = depth_image.float()

        if actions is not None:   # training
            actions  = actions[:, :self.model.num_queries]
            is_pad   = is_pad[:, :self.model.num_queries]
            a_hat, _, (mu, logvar), probs, binaries = self.model(
                qpos, image, depth_image, env_state, actions, is_pad, vq_sample,
                lang_embed=language_distilbert)
            total_kld, _, _ = _kl_divergence(mu, logvar)
            all_l1 = torch.nn.functional.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return {"l1": l1, "kl": total_kld[0], "loss": l1 + total_kld[0] * self.kl_weight}
        else:   # inference
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, depth_image, env_state, vq_sample=vq_sample,
                lang_embed=language_distilbert)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def _kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu     = mu.view(mu.size(0), mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total   = klds.sum(1).mean(0, True)
    dim_wise = klds.mean(0)
    mean    = klds.mean(1).mean(0, True)
    return total, dim_wise, mean


# ---------------------------------------------------------------------------
# Inference policy — used by tools/policies/runner.py
# ---------------------------------------------------------------------------

class ACTPolicy(BasePolicy):
    """
    ACT inference policy.

    Loads checkpoint + dataset_stats.pkl from the checkpoint directory.
    Implements temporal aggregation (exp-weighted average over chunk predictions).

    Args (passed via model_path):
      model_path: path to .ckpt file  (dataset_stats.pkl must be in the same dir)
    """

    # Configurable at load time via extra kwargs passed from runner
    CHUNK_SIZE: int = 50
    TEMPORAL_AGG: bool = True
    ACTION_HORIZON: int | None = None
    TEMPORAL_DECAY: float = 0.01
    TEMPORAL_PRIORITY: str = "oldest"
    TEMPORAL_PRIORITIES = ("oldest", "newest", "uniform")
    EPISODE_LEN: int = 2000
    CAM_NAME: str = "camera_head"
    # Inference image resize (width, height). Default 320x240. Overridden by
    # runner.py per policy name (act → 320x240, act_v1 → 640x480) and/or the
    # --img-w/--img-h flags. Must match the resolution the checkpoint was trained at.
    IMG_W: int = 320
    IMG_H: int = 240

    # Debug log file (written alongside the checkpoint)
    _dbg_file = None

    # ACT uses a synthetic hand state because simulator measurements may be
    # unreliable at reset. The validated default is the historical seed;
    # task-specific and measured variants remain available for diagnostics.
    _DEFAULT_L_HAND_HOME = np.array([1.570001, 0.031568, 0.041977, 0.042064, 0.042116, 0.042173], dtype=np.float32)
    _DEFAULT_R_HAND_HOME = np.array([1.570001, 0.031575, 0.041989, 0.042076, 0.042128, 0.042185], dtype=np.float32)
    _LEGACY_L_HAND_HOME = np.array([1.316, 0.204, 0.209, 0.261, 0.320, 0.312], dtype=np.float32)
    _LEGACY_R_HAND_HOME = np.array([1.316, 0.204, 0.209, 0.261, 0.320, 0.312], dtype=np.float32)
    HAND_STATE_MODES = ("task_home", "legacy_home", "measured")
    HAND_FEEDBACK_MODES = ("commanded", "measured")
    IMAGE_COLOR_ORDERS = ("rgb", "bgr")
    _TASK_HAND_HOME = {
        "ind_task_01": (
            np.array([1.570001, 0.031568, 0.041977, 0.042064, 0.042116, 0.042173], dtype=np.float32),
            np.array([1.570001, 0.031575, 0.041989, 0.042076, 0.042128, 0.042185], dtype=np.float32),
        ),
        "ind_task_02": (
            np.array([1.570001, 0.016734, 0.022164, 0.022263, 0.022337, 0.022375], dtype=np.float32),
            np.array([1.570001, 0.014872, 0.019653, 0.019761, 0.019847, 0.019879], dtype=np.float32),
        ),
        "ind_task_03": (
            np.array([1.570001, 0.019321, 0.025642, 0.025728, 0.025783, 0.025834], dtype=np.float32),
            np.array([1.570001, 0.015213, 0.020087, 0.020203, 0.020295, 0.020327], dtype=np.float32),
        ),
        "lab_task_01": (
            np.array([0.973687, 0.179566, 0.007574, 0.017032, 0.020709, 0.052926], dtype=np.float32),
            np.array([0.973766, 0.179574, 0.007638, 0.017131, 0.020844, 0.052949], dtype=np.float32),
        ),
        "lab_task_03": (
            np.array([0.918868, 0.234877, 0.008913, 0.017129, 0.019561, 0.036912], dtype=np.float32),
            np.array([0.918958, 0.235073, 0.008919, 0.017224, 0.019582, 0.036947], dtype=np.float32),
        ),
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        task_name: str = "",
        hand_state_mode: str = "legacy_home",
        hand_feedback: str | None = None,
        image_color_order: str = "rgb",
        debug_every: int = 50,
        chunk_size: int | None = None,
        temporal_agg: bool | None = None,
        action_horizon: int | None = None,
        temporal_decay: float | None = None,
        temporal_priority: str | None = None,
    ):
        if hand_state_mode not in self.HAND_STATE_MODES:
            raise ValueError(
                f"Unknown hand_state_mode {hand_state_mode!r}; "
                f"expected one of {self.HAND_STATE_MODES}"
            )
        if hand_feedback is None:
            hand_feedback = "measured" if hand_state_mode == "measured" else "commanded"
        if hand_feedback not in self.HAND_FEEDBACK_MODES:
            raise ValueError(
                f"Unknown hand_feedback {hand_feedback!r}; "
                f"expected one of {self.HAND_FEEDBACK_MODES}"
            )
        if image_color_order not in self.IMAGE_COLOR_ORDERS:
            raise ValueError(
                f"Unknown image_color_order {image_color_order!r}; "
                f"expected one of {self.IMAGE_COLOR_ORDERS}"
            )
        self.model_chunk_size = int(self.CHUNK_SIZE if chunk_size is None else chunk_size)
        if self.model_chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.temporal_agg = self.TEMPORAL_AGG if temporal_agg is None else temporal_agg
        requested_horizon = self.ACTION_HORIZON if action_horizon is None else action_horizon
        self.action_horizon = self.model_chunk_size if requested_horizon is None else int(requested_horizon)
        if not 1 <= self.action_horizon <= self.model_chunk_size:
            raise ValueError(
                f"action_horizon must be in [1, {self.model_chunk_size}], "
                f"got {self.action_horizon}"
            )
        self.temporal_decay = self.TEMPORAL_DECAY if temporal_decay is None else float(temporal_decay)
        if self.temporal_decay < 0:
            raise ValueError("temporal_decay must be non-negative")
        self.temporal_priority = self.TEMPORAL_PRIORITY if temporal_priority is None else temporal_priority
        if self.temporal_priority not in self.TEMPORAL_PRIORITIES:
            raise ValueError(
                f"Unknown temporal_priority {self.temporal_priority!r}; "
                f"expected one of {self.TEMPORAL_PRIORITIES}"
            )
        self.task_name = task_name or self._infer_task_name(model_path)
        self.hand_state_mode = hand_state_mode
        self.hand_feedback = hand_feedback
        self.image_color_order = image_color_order
        self.debug_every = max(1, int(debug_every))
        super().__init__(model_path, device)

    @classmethod
    def _infer_task_name(cls, model_path: str) -> str:
        path = os.path.abspath(model_path or "")
        for task_name in cls._TASK_HAND_HOME:
            if task_name in path:
                return task_name
        return ""

    def _select_hand_home(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hand_state_mode == "legacy_home":
            return self._LEGACY_L_HAND_HOME.copy(), self._LEGACY_R_HAND_HOME.copy()
        homes = self._TASK_HAND_HOME.get(self.task_name)
        if homes is not None:
            return homes[0].copy(), homes[1].copy()
        if hasattr(self, "norm_stats"):
            qpos_mean = np.asarray(self.norm_stats["qpos_mean"], dtype=np.float32)
            return qpos_mean[7:13].copy(), qpos_mean[20:26].copy()
        return self._DEFAULT_L_HAND_HOME.copy(), self._DEFAULT_R_HAND_HOME.copy()

    def _open_debug_log(self):
        log_path = os.path.join(os.path.dirname(self.model_path), "act_debug.log")
        self._dbg_file = open(log_path, "w", buffering=1)
        print(f"[ACTPolicy] Debug log → {log_path}")

    def _dbg(self, msg: str):
        print(msg)
        if self._dbg_file:
            self._dbg_file.write(msg + "\n")

    def _load_model(self):
        ckpt_dir = os.path.dirname(self.model_path)
        self._open_debug_log()

        # --- load norm stats ---
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found in {ckpt_dir}")
        with open(stats_path, "rb") as f:
            self.norm_stats = pickle.load(f)

        # --- build model ---
        self.chunk_size = self.model_chunk_size
        policy = build_act_model(chunk_size=self.chunk_size, device=self.device)
        ckpt = torch.load(self.model_path, map_location="cpu")
        policy.deserialize(ckpt["nets"])
        policy.eval()
        policy.to(self.device)
        print(f"[ACTPolicy] Loaded from {self.model_path} (step {ckpt.get('step', '?')})")
        print(f"[ACTPolicy] task={self.task_name or 'unknown'}")

        # temporal aggregation buffer (reset on each episode)
        self._init_buffers()
        return policy

    def _init_buffers(self):
        self.t = -1
        # Seed commanded hand positions; measured-feedback mode may override them.
        self._last_l_hand, self._last_r_hand = self._select_hand_home()
        self._last_action: np.ndarray | None = None
        if self.temporal_agg:
            from collections import deque
            # Rolling buffer: stores (base_t, chunk_pred) for last chunk_size predictions
            # No fixed episode length limit — works for any task duration
            self._pred_history: deque = deque()
        else:
            self._chunk_actions = None

    def reset(self):
        self._init_buffers()

    # --- obs pre-processing ---

    @staticmethod
    def _read_measured_hand(obs: dict, key: str, fallback: np.ndarray) -> np.ndarray:
        """Return a finite six-joint hand state, falling back per invalid joint."""
        try:
            measured = np.asarray(obs["puppet"][key]["data"], dtype=np.float32).ravel()
        except (KeyError, TypeError, ValueError):
            return fallback.copy()
        if measured.size != fallback.size:
            return fallback.copy()
        return np.where(np.isfinite(measured), measured, fallback).astype(np.float32)

    def _input_hands(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        if self.hand_feedback == "commanded":
            return self._last_l_hand.copy(), self._last_r_hand.copy()
        return (
            self._read_measured_hand(
                obs, "end_effector_left_position_raw", self._last_l_hand
            ),
            self._read_measured_hand(
                obs, "end_effector_right_position_raw", self._last_r_hand
            ),
        )

    def _get_qpos(self, obs: dict) -> np.ndarray:
        l_arm = np.array(obs["puppet"]["arm_left_position_raw"]["data"]).ravel().astype(np.float32)
        r_arm = np.array(obs["puppet"]["arm_right_position_raw"]["data"]).ravel().astype(np.float32)
        l_arm = np.nan_to_num(l_arm, nan=0.0)
        r_arm = np.nan_to_num(r_arm, nan=0.0)
        l_hand, r_hand = self._input_hands(obs)
        return np.concatenate([l_arm, l_hand, r_arm, r_hand])  # (26,)

    def _log_telemetry(self, obs: dict, qpos: np.ndarray, action: np.ndarray) -> None:
        if self.t % self.debug_every != 0:
            return
        measured_l = self._read_measured_hand(
            obs, "end_effector_left_position_raw", self._last_l_hand
        )
        measured_r = self._read_measured_hand(
            obs, "end_effector_right_position_raw", self._last_r_hand
        )
        delta = (
            np.zeros_like(action)
            if self._last_action is None
            else action - self._last_action
        )
        self._dbg(
            f"[ACT TELEMETRY t={self.t} mode={self.hand_state_mode} "
            f"feedback={self.hand_feedback}] "
            f"image_order={self.image_color_order} "
            f"measured_l_hand={np.round(measured_l, 3).tolist()} "
            f"input_l_hand={np.round(qpos[7:13], 3).tolist()} "
            f"measured_r_hand={np.round(measured_r, 3).tolist()} "
            f"input_r_hand={np.round(qpos[20:26], 3).tolist()}"
        )
        self._dbg(
            f"[ACT ACTION t={self.t}] "
            f"left_arm={np.round(action[:7], 4).tolist()} "
            f"left_hand={np.round(action[7:13], 4).tolist()} "
            f"right_arm={np.round(action[13:20], 4).tolist()} "
            f"right_hand={np.round(action[20:26], 4).tolist()} "
            f"delta_l2={float(np.linalg.norm(delta)):.5f}"
        )

    def _get_image(self, obs: dict) -> torch.Tensor:
        img_rgb = obs["camera_observations"]["color_images"][self.CAM_NAME]  # (H, W, 3) RGB
        img_rgb = cv2.resize(img_rgb, (self.IMG_W, self.IMG_H))
        if self.image_color_order == "bgr":
            img_rgb = img_rgb[:, :, ::-1].copy()
        img_f = (img_rgb / 255.0).astype(np.float32)
        t = torch.from_numpy(img_f).permute(2, 0, 1)                         # (3, H, W)
        return t.unsqueeze(0).unsqueeze(0)                                    # (1, 1, 3, H, W)

    def _get_depth(self, obs: dict) -> torch.Tensor | None:
        """Return an optional depth tensor for RGB-D subclasses.

        The original ACT checkpoint is RGB-only. Keeping the default at None
        preserves its architecture and checkpoint compatibility while allowing
        a separately trained policy to opt into the existing depth branch.
        """
        del obs
        return None

    def _normalize_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        q = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return torch.from_numpy(q).float().unsqueeze(0)                       # (1, 26)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.norm_stats["action_std"] + self.norm_stats["action_mean"]

    @staticmethod
    def temporal_weights(length: int, decay: float, priority: str) -> np.ndarray:
        """Return normalized weights for predictions ordered oldest to newest."""
        if length <= 0:
            raise ValueError("length must be positive")
        if priority == "uniform":
            weights = np.ones(length, dtype=np.float64)
        elif priority == "oldest":
            weights = np.exp(-decay * np.arange(length, dtype=np.float64))
        elif priority == "newest":
            weights = np.exp(-decay * np.arange(length - 1, -1, -1, dtype=np.float64))
        else:
            raise ValueError(f"Unknown temporal priority: {priority}")
        return weights / weights.sum()

    # --- main inference ---

    def infer(self, obs: dict) -> np.ndarray:
        self.t += 1
        t = self.t

        # DEBUG: 第一步打印 obs 结构，确认 key 和维度
        if t == 0:
            try:
                self._dbg("[ACT DEBUG] obs puppet keys:")
                for k in obs['puppet']:
                    d = np.array(obs['puppet'][k]['data'])
                    self._dbg(f"  puppet[{k}]: shape={d.shape}, val={np.round(d.ravel()[:7],3).tolist()}")
                qpos_raw = self._get_qpos(obs)
                self._dbg(f"[ACT DEBUG] qpos(26): {np.round(qpos_raw,3).tolist()}")
                self._dbg(
                    f"[ACT DEBUG] hand_state_mode={self.hand_state_mode} "
                    f"hand_feedback={self.hand_feedback} "
                    f"image_color_order={self.image_color_order} "
                    f"temporal_agg={self.temporal_agg} "
                    f"action_horizon={self.action_horizon} "
                    f"temporal_priority={self.temporal_priority} "
                    f"temporal_decay={self.temporal_decay} "
                    f"debug_every={self.debug_every}"
                )
            except Exception as e:
                self._dbg(f"[ACT DEBUG] obs inspect error: {e}")

        query_frequency = 1 if self.temporal_agg else self.action_horizon

        with torch.inference_mode():
            if t % query_frequency == 0:
                qpos = self._get_qpos(obs)
                qpos_t  = self._normalize_qpos(qpos).to(self.device)     # (1, 26)
                image_t = self._get_image(obs).to(self.device)            # (1, 1, 3, H, W)
                depth_t = self._get_depth(obs)
                if depth_t is not None:
                    depth_t = depth_t.to(self.device)

                # model expects image shape (B, num_cams, C, H, W)
                all_actions = self.model(qpos_t, image_t, depth_t)        # (1, chunk, 26)
                all_actions = all_actions.cpu().numpy()
                self._chunk_actions = all_actions                          # cache for non-query steps
                if self.temporal_agg:
                    self._pred_history.append((t, all_actions[0]))
                    if len(self._pred_history) > self.chunk_size:
                        self._pred_history.popleft()
                # DEBUG: 每10步打印一次原始 chunk 首尾，看模型有没有预测运动
                if t % 10 == 0:
                    mid  = self.chunk_size // 2
                    r0   = np.round(all_actions[0,   0, 13:20], 3).tolist()
                    r_end= np.round(all_actions[0,  -1, 13:20], 3).tolist()
                    h0   = np.round(all_actions[0,   0, 20:26], 3).tolist()
                    h_mid= np.round(all_actions[0, mid, 20:26], 3).tolist()
                    h_end= np.round(all_actions[0,  -1, 20:26], 3).tolist()
                    self._dbg(f"[CHUNK t={t}] r_arm@0={r0} @{self.chunk_size-1}={r_end}")
                    self._dbg(f"[CHUNK t={t}] r_hand@0={h0} @{mid}={h_mid} @{self.chunk_size-1}={h_end}")

            if self.temporal_agg:
                chunk = self.chunk_size
                # Collect all historical predictions that cover step t
                actions_for_t = []
                for (base_t, chunk_pred) in self._pred_history:
                    offset = t - base_t
                    if 0 <= offset < chunk:
                        actions_for_t.append(chunk_pred[offset])
                # DEBUG: 每50步打印一次 populated 数量
                if t % 50 == 0:
                    self._dbg(f"[ACT DEBUG t={t}] populated={len(actions_for_t)}/{min(t+1, chunk)}")
                if len(actions_for_t) == 0:
                    raw = all_actions[0, 0]
                else:
                    actions_for_t = np.array(actions_for_t)
                    weights = self.temporal_weights(
                        len(actions_for_t), self.temporal_decay, self.temporal_priority
                    )
                    raw = (actions_for_t * weights[:, np.newaxis]).sum(0)  # (26,)
            else:
                raw = self._chunk_actions[0, t % query_frequency]         # (26,)

        action = self._denormalize_action(raw)
        qpos_for_telemetry = self._get_qpos(obs)
        self._log_telemetry(obs, qpos_for_telemetry, action)
        # Track commanded EE so next step's qpos obs stays self-consistent
        self._last_l_hand = action[7:13].copy()
        self._last_r_hand = action[20:26].copy()
        self._last_action = action.copy()
        return action
