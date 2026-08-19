"""
Conditional Flow Matching policy for 26-dim humanoid action chunks.

This file contains both:
- the trainable action generator used by tools/train/train_flow.py
- the inference policy used by tools/policies/runner.py

The action order matches ACT and the simulator contract:
  [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.policies.base_policy import BasePolicy


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float().view(-1, 1)
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        emb = t * freq.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FlowActionModel(nn.Module):
    """Rectified-flow action generator conditioned on image + robot state."""

    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 26,
        state_dim: int = 26,
        hidden_dim: int = 256,
        num_layers: int = 4,
        nheads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.qpos_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def _encode_condition(self, qpos: torch.Tensor, image: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            image = image[:, 0]
        image_feat = self.image_encoder(image)
        qpos_feat = self.qpos_encoder(qpos)
        time_feat = self.time_encoder(t)
        return image_feat + qpos_feat + time_feat

    def predict_velocity(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._encode_condition(qpos, image, t).unsqueeze(1)
        h = self.action_proj(noisy_action) + self.pos_embed[:, : noisy_action.shape[1]] + cond
        h = self.transformer(h)
        return self.out_proj(h)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = actions.shape[0]
        t = torch.rand(bsz, device=actions.device)
        x0 = torch.randn_like(actions)
        x1 = actions
        xt = (1.0 - t.view(bsz, 1, 1)) * x0 + t.view(bsz, 1, 1) * x1
        target_v = x1 - x0
        pred_v = self.predict_velocity(qpos, image, xt, t)

        valid = (~is_pad).unsqueeze(-1).float()
        denom = valid.sum().clamp_min(1.0) * actions.shape[-1]
        flow_mse = ((pred_v - target_v).pow(2) * valid).sum() / denom
        return {"loss": flow_mse, "flow_mse": flow_mse}

    @torch.inference_mode()
    def sample_actions(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        sample_steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(
            qpos.shape[0],
            self.chunk_size,
            self.action_dim,
            device=qpos.device,
            generator=generator,
        )
        dt = 1.0 / float(sample_steps)
        for step in range(sample_steps):
            t = torch.full((qpos.shape[0],), step / float(sample_steps), device=qpos.device)
            v = self.predict_velocity(qpos, image, x, t)
            x = x + dt * v
        return x

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def build_flow_model(
    chunk_size: int = 16,
    action_dim: int = 26,
    state_dim: int = 26,
    hidden_dim: int = 256,
    num_layers: int = 4,
    nheads: int = 8,
    dropout: float = 0.1,
) -> FlowActionModel:
    return FlowActionModel(
        chunk_size=chunk_size,
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nheads=nheads,
        dropout=dropout,
    )


class FlowPolicy(BasePolicy):
    CHUNK_SIZE: int = 16
    SAMPLE_STEPS: int = 8
    ACTION_HORIZON: int = 4
    CAM_NAME: str = "camera_head"
    IMG_W: int = 320
    IMG_H: int = 240

    _DEFAULT_L_HAND_HOME = np.array([1.570001, 0.031568, 0.041977, 0.042064, 0.042116, 0.042173], dtype=np.float32)
    _DEFAULT_R_HAND_HOME = np.array([1.570001, 0.031575, 0.041989, 0.042076, 0.042128, 0.042185], dtype=np.float32)
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

    def __init__(self, model_path: str, device: str = "cuda", task_name: str = "", seed: int = 1):
        self.task_name = task_name or self._infer_task_name(model_path)
        self.seed = seed
        self._reset_count = 0
        super().__init__(model_path, device)

    @classmethod
    def _infer_task_name(cls, model_path: str) -> str:
        path = os.path.abspath(model_path or "")
        for task_name in cls._TASK_HAND_HOME:
            if task_name in path:
                return task_name
        return ""

    def _select_hand_home(self) -> tuple[np.ndarray, np.ndarray]:
        homes = self._TASK_HAND_HOME.get(self.task_name)
        if homes is not None:
            return homes[0].copy(), homes[1].copy()
        if hasattr(self, "norm_stats"):
            qpos_mean = np.asarray(self.norm_stats["qpos_mean"], dtype=np.float32)
            return qpos_mean[7:13].copy(), qpos_mean[20:26].copy()
        return self._DEFAULT_L_HAND_HOME.copy(), self._DEFAULT_R_HAND_HOME.copy()

    def _load_model(self):
        ckpt_dir = os.path.dirname(self.model_path)
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found in {ckpt_dir}")
        with open(stats_path, "rb") as f:
            self.norm_stats = pickle.load(f)

        ckpt = torch.load(self.model_path, map_location="cpu")
        config = ckpt.get("config", {})
        self.CHUNK_SIZE = int(config.get("chunk_size", self.CHUNK_SIZE))
        self.IMG_W = int(config.get("img_w", self.IMG_W))
        self.IMG_H = int(config.get("img_h", self.IMG_H))
        self.chunk_size = self.CHUNK_SIZE
        model = build_flow_model(
            chunk_size=self.CHUNK_SIZE,
            action_dim=int(config.get("action_dim", 26)),
            state_dim=int(config.get("state_dim", 26)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            nheads=int(config.get("nheads", 8)),
            dropout=float(config.get("dropout", 0.1)),
        )
        model.deserialize(ckpt["nets"])
        model.eval()
        model.to(self.device)
        print(f"[FlowPolicy] Loaded from {self.model_path} (step {ckpt.get('step', '?')})")
        print(f"[FlowPolicy] task={self.task_name or 'unknown'} chunk={self.CHUNK_SIZE} sample_steps={self.SAMPLE_STEPS}")
        self.reset()
        return model

    def reset(self):
        self.t = -1
        self._last_l_hand, self._last_r_hand = self._select_hand_home()
        self._queue = None
        self._queue_idx = 0
        self._reset_count += 1
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.seed + self._reset_count)

    def _get_qpos(self, obs: dict) -> np.ndarray:
        l_arm = np.array(obs["puppet"]["arm_left_position_raw"]["data"]).ravel().astype(np.float32)
        r_arm = np.array(obs["puppet"]["arm_right_position_raw"]["data"]).ravel().astype(np.float32)
        l_arm = np.nan_to_num(l_arm, nan=0.0)
        r_arm = np.nan_to_num(r_arm, nan=0.0)
        return np.concatenate([l_arm, self._last_l_hand, r_arm, self._last_r_hand])

    def _get_image(self, obs: dict) -> torch.Tensor:
        img_rgb = obs["camera_observations"]["color_images"][self.CAM_NAME]
        img_rgb = cv2.resize(img_rgb, (self.IMG_W, self.IMG_H))
        img_f = (img_rgb / 255.0).astype(np.float32)
        t = torch.from_numpy(img_f).permute(2, 0, 1)
        return t.unsqueeze(0).unsqueeze(0)

    def _normalize_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        q = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return torch.from_numpy(q).float().unsqueeze(0)

    def _denormalize_action_chunk(self, action: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.norm_stats["action_mean"], dtype=np.float32)
        action_min = np.asarray(self.norm_stats["action_min"], dtype=np.float32)
        action_max = np.asarray(self.norm_stats["action_max"], dtype=np.float32)
        denormalized = action * self.norm_stats["action_std"] + mean
        # Flow sampling starts from Gaussian noise, so guard the simulator against
        # rare non-finite or out-of-distribution joint targets.
        denormalized = np.where(np.isfinite(denormalized), denormalized, mean)
        return np.clip(denormalized, action_min, action_max).astype(np.float32)

    def _sample_chunk(self, obs: dict) -> np.ndarray:
        qpos = self._get_qpos(obs)
        qpos_t = self._normalize_qpos(qpos).to(self.device)
        image_t = self._get_image(obs).to(self.device)
        with torch.inference_mode():
            chunk = self.model.sample_actions(
                qpos_t,
                image_t,
                sample_steps=self.SAMPLE_STEPS,
                generator=self._rng,
            )
        chunk_np = chunk[0].cpu().numpy()
        return self._denormalize_action_chunk(chunk_np)

    def infer(self, obs: dict) -> np.ndarray:
        self.t += 1
        if self._queue is None or self._queue_idx >= min(self.ACTION_HORIZON, len(self._queue)):
            self._queue = self._sample_chunk(obs)
            self._queue_idx = 0
        action = self._queue[self._queue_idx].astype(np.float32)
        self._queue_idx += 1
        self._last_l_hand = action[7:13].copy()
        self._last_r_hand = action[20:26].copy()
        return action
