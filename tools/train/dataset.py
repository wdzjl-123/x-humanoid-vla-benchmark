"""
PyTorch dataset for ACT training on RoboMIND2.0 data.

Adapted from act_dp_ref/dataset_load/dataset_multi_robot_v2.py.

Directory layout expected by load_data() — flat release layout:
  <task_dir>/train/<ep_id>.hdf5
  <task_dir>/val/<ep_id>.hdf5        (val may be empty → no validation)

See README §2 for how to download the data and lay it out.
"""
from __future__ import annotations
import glob
import os
import pickle
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.train.h5_reader import H5Reader, TIENKUNG_CONTROLS, EE_DIMS

# Image size used for training and inference
IMG_W, IMG_H = 320, 240   # (width, height) for cv2.resize


def _collect_hdf5(split_dir: str, reader: H5Reader) -> list[str]:
    """Collect all valid .hdf5 files directly under split_dir (flat layout)."""
    files: list[str] = []
    if not os.path.isdir(split_dir):
        return files
    for fpath in sorted(glob.glob(os.path.join(split_dir, "*.hdf5"))):
        try:
            reader.episode_length(fpath)   # validate file is readable
            files.append(fpath)
        except Exception as e:
            print(f"[dataset] skip {fpath}: {e}")
    return files


def _concat_qpos(ctrl_puppet: dict) -> np.ndarray:
    """Concatenate puppet controls into qpos array.

    arm_* is kept as-is (7 dims each).
    end_effector_* is 6-dim in the current challenge data and used as-is; if a
    file is 12-dim (old format) it is sliced to EE_DIMS (6 dims). Either way the
    resulting state_dim=26 matches what the benchmark ZMQ sends at inference time.
    """
    parts = []
    for c in TIENKUNG_CONTROLS:
        if c not in ctrl_puppet:
            continue
        d = ctrl_puppet[c]
        if "end_effector" in c and d.shape[-1] > 6:
            d = d[..., EE_DIMS]   # 12-dim → 6-dim; current 6-dim data skips this
        parts.append(d)
    return np.concatenate(parts, axis=-1)   # (..., 26)


def get_norm_stats(
    train_files: list[str],
    val_files: list[str],
    reader: H5Reader,
) -> tuple[dict, list[int], list[int]]:
    """Compute z-score normalization stats from all episodes.

    Returns:
        stats: dict with action/qpos mean, std, min, max as numpy arrays
        train_ep_len: list of episode lengths for train set
        val_ep_len:   list of episode lengths for val set
    """
    all_qpos: list[torch.Tensor] = []
    all_action: list[torch.Tensor] = []
    train_ep_len: list[int] = []
    val_ep_len: list[int] = []

    for split_files, ep_len_list in [(train_files, train_ep_len), (val_files, val_ep_len)]:
        for fpath in split_files:
            try:
                _, ctrl = reader.read(fpath, camera_frame=0)
                qpos   = _concat_qpos(ctrl["puppet"])   # (T, 26)
                action = _concat_qpos(ctrl["puppet"])   # (T, 26) — puppet joint angles as action targets
                all_qpos.append(torch.from_numpy(qpos))
                all_action.append(torch.from_numpy(action))
                ep_len_list.append(len(action))
            except Exception as e:
                print(f"[norm_stats] skip {fpath}: {e}")

    all_qpos_t = torch.cat(all_qpos, dim=0).float()
    all_action_t = torch.cat(all_action, dim=0).float()

    eps = 1e-4
    stats = {
        "qpos_mean": all_qpos_t.mean(0).numpy(),
        "qpos_std":  torch.clamp(all_qpos_t.std(0), 1e-2).numpy(),
        "qpos_min":  all_qpos_t.min(0).values.numpy() - eps,
        "qpos_max":  all_qpos_t.max(0).values.numpy() + eps,
        "action_mean": all_action_t.mean(0).numpy(),
        "action_std":  torch.clamp(all_action_t.std(0), 1e-2).numpy(),
        "action_min":  all_action_t.min(0).values.numpy() - eps,
        "action_max":  all_action_t.max(0).values.numpy() + eps,
    }
    return stats, train_ep_len, val_ep_len


class EpisodicDataset(torch.utils.data.Dataset):
    """Dataset that samples random (episode, timestep) pairs.

    Args:
        file_list: flat list of .hdf5 episode paths
        ep_ids: index into file_list for each position in the cumulative sum
        ep_len: episode length for each entry in ep_ids
        norm_stats: output of get_norm_stats()
        chunk_size: number of action steps per sample
        reader: H5Reader instance
        use_aug: apply random crop+rotate+color jitter
    """

    def __init__(
        self,
        file_list: list[str],
        ep_ids: list[int],
        ep_len: list[int],
        norm_stats: dict,
        chunk_size: int,
        reader: H5Reader,
        use_aug: bool = False,
        img_w: int = IMG_W,
        img_h: int = IMG_H,
        action_norm: str = 'zscore',
        obs_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.file_list = file_list
        self.ep_ids = ep_ids
        self.ep_len = ep_len
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.reader = reader
        self.use_aug = use_aug
        self.img_w = img_w
        self.img_h = img_h
        self.action_norm = action_norm
        self.obs_horizon = obs_horizon
        self.cumlen = np.cumsum(ep_len)

        if use_aug:
            import torchvision.transforms as T
            ratio = 0.95
            self.aug_transforms = [
                T.RandomCrop([int(img_h * ratio), int(img_w * ratio)]),
                T.Resize((img_h, img_w), antialias=True),
                T.RandomRotation(degrees=[-5.0, 5.0]),
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
        else:
            self.aug_transforms = []

    def __len__(self) -> int:
        if len(self.cumlen) == 0:
            return 0
        return 1000 * int(self.cumlen[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        index = index % int(self.cumlen[-1])
        ep_idx = int(np.argmax(self.cumlen > index))
        start_ts = index - int(self.cumlen[ep_idx] - self.ep_len[ep_idx])
        return self.ep_ids[ep_idx], start_ts

    def __getitem__(self, index: int):
        ep_id, start_ts = self._locate(index)
        fpath = self.file_list[ep_id]

        # Read with chunk_size so control data spans [start_ts : start_ts + chunk_size]
        # → ctrl["puppet"][key].shape == (chunk_size_or_less, D)
        # → qpos[0] == state at start_ts, action == targets from start_ts
        image_dict, ctrl = self.reader.read(fpath, camera_frame=start_ts,
                                            chunk_size=self.chunk_size)

        # --- qpos (state at start_ts) ---
        qpos = _concat_qpos(ctrl["puppet"])   # (chunk_size, 26)
        qpos = qpos[0]                        # state at start_ts → (26,)

        # --- action (puppet joint angles, chunk_size steps from start_ts) ---
        # Using puppet angles (not master grasp %) so inference output is directly
        # applicable as joint position targets in joint_callback (physically correct).
        action = _concat_qpos(ctrl["puppet"])   # (T', 26) where T' <= chunk_size
        episode_len = len(action)

        # Pad / truncate to chunk_size
        if episode_len < self.chunk_size:
            padded = np.zeros((self.chunk_size, action.shape[-1]), dtype=np.float32)
            padded[:episode_len] = action
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[episode_len:] = True
        else:
            padded = action[:self.chunk_size]
            is_pad = np.zeros(self.chunk_size, dtype=bool)

        # --- image ---
        cam_name = self.reader.camera_names[0]

        def _decode_rgb(bgr):                              # cv2.imdecode returns BGR
            bgr = cv2.resize(bgr, (self.img_w, self.img_h))
            rgb = bgr[:, :, ::-1].copy()                  # BGR -> RGB, matching sim obs at inference
            return (rgb / 255.0).astype(np.float32)

        # --- image tensor (obs_horizon frames: oldest→newest, slot0=t-obs_horizon+1) ---
        if self.obs_horizon == 1:
            img_f = _decode_rgb(image_dict["color_images"][cam_name])
            image_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        else:
            frames = []
            for t_back in reversed(range(self.obs_horizon)):
                ts = max(0, start_ts - t_back)
                img_d, _ = self.reader.read(fpath, camera_frame=ts, chunk_size=1)
                f = _decode_rgb(img_d["color_images"][cam_name])
                frames.append(torch.from_numpy(f).permute(2, 0, 1))
            image_t = torch.stack(frames, dim=0)   # (obs_horizon, 3, H, W)

        # --- normalize ---
        stats = self.norm_stats
        if self.action_norm == 'minmax':
            action_range = stats["action_max"] - stats["action_min"]
            action_norm = (padded - stats["action_min"]) / action_range * 2.0 - 1.0
        else:
            action_norm = (padded - stats["action_mean"]) / stats["action_std"]
        qpos_norm = (qpos - stats["qpos_mean"]) / stats["qpos_std"]

        # --- to tensors ---
        # image_t already built above
        action_t = torch.from_numpy(action_norm).float()                   # (chunk, 26)
        qpos_t   = torch.from_numpy(qpos_norm).float()                     # (26,)
        is_pad_t = torch.from_numpy(is_pad).bool()                         # (chunk,)
        depth_t  = torch.zeros(1, self.img_h, self.img_w)                  # dummy depth

        if self.use_aug and self.aug_transforms:
            for t in self.aug_transforms:
                image_t = t(image_t)

        # Return order matches forward_pass() in train_act.py:
        # image, depth, qpos, action, is_pad
        return image_t, depth_t, qpos_t, action_t, is_pad_t


def load_data(
    task_dir: str,
    robot_infor: dict,
    batch_size_train: int,
    batch_size_val: int,
    chunk_size: int,
    use_aug: bool = False,
    num_workers: int = 4,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    action_norm: str = 'zscore',
    obs_horizon: int = 1,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/val DataLoaders and compute norm stats.

    Args:
        task_dir: path to the task directory (contains train/ and val/)
        robot_infor: dict with camera_names key
        batch_size_train / batch_size_val: batch sizes
        chunk_size: action chunk length
        use_aug: use data augmentation
        num_workers: DataLoader workers
        img_w / img_h: target image size for cv2.resize (default 320×240 for ACT;
                       pass 640×480 for DroidDiffusion)

    Returns:
        train_loader, val_loader, norm_stats
    """
    camera_names = robot_infor["camera_names"]
    reader = H5Reader(camera_names=camera_names)

    # Flat release layout: <task>/train and <task>/val (see README §2).
    train_files = _collect_hdf5(os.path.join(task_dir, "train"), reader)
    val_files   = _collect_hdf5(os.path.join(task_dir, "val"),   reader)

    print(f"[dataset] train={len(train_files)}  val={len(val_files)}")

    norm_stats, train_ep_len, val_ep_len = get_norm_stats(train_files, val_files, reader)

    n_train = len(train_files)
    n_val   = len(val_files)
    train_ep_ids = list(range(n_train))
    val_ep_ids   = list(range(n_val))

    train_ds = EpisodicDataset(train_files, train_ep_ids, train_ep_len, norm_stats,
                               chunk_size, reader, use_aug, img_w=img_w, img_h=img_h,
                               action_norm=action_norm, obs_horizon=obs_horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    # No val/ episodes (e.g. the unified "200 episodes, all for training" release)
    # → return val_loader=None so callers skip validation instead of crashing.
    if not val_files:
        return train_loader, None, norm_stats

    val_ds   = EpisodicDataset(val_files,   val_ep_ids,   val_ep_len,   norm_stats,
                               chunk_size, reader, False,  img_w=img_w, img_h=img_h,
                               action_norm=action_norm, obs_horizon=obs_horizon)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size_val,   shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, norm_stats
