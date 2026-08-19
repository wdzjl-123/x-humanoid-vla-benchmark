"""
Minimal HDF5 reader for RoboMIND2.0 data.

HDF5 structure (per episode file):
  camera_observations/color_images/<cam_name>   shape=(T,), dtype=object  (JPEG bytes)
  puppet/<control>/data                          shape=(T, D)
  master/<control>/data                          shape=(T, D)

Adapted from act_dp_ref/dataset_load/read_h5_v2.py.
"""
from __future__ import annotations
import os
from collections import defaultdict

import cv2
import h5py
import numpy as np


# Controls used for TienKung dual-arm dex-hand robot
TIENKUNG_CONTROLS = [
    "arm_left_position_align",
    "end_effector_left_position_align",
    "arm_right_position_align",
    "end_effector_right_position_align",
]

# HDF5 end_effector data may be 12-dim (all hand joints) or already 6-dim.
# When 12-dim, EE_DIMS selects the 6 joints the benchmark ZMQ uses:
# thumb_metacarpal(0), thumb_proximal(1), index_proximal(5),
# middle_proximal(7), ring_proximal(9), pinky_proximal(11) — keeping training
# and inference aligned. The current challenge data is already 6-dim, so the
# slice is skipped (see the `> 6` guard in dataset._concat_qpos).
EE_DIMS = [0, 1, 5, 7, 9, 11]


def _decode_jpeg(buf: np.ndarray) -> np.ndarray:
    """Decode a JPEG-encoded byte buffer into an HxWx3 BGR uint8 array."""
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class H5Reader:
    """Read a single RoboMIND2.0 HDF5 episode file.

    Args:
        camera_names: list of camera names to load (e.g. ['camera_head'])
        controls: list of control keys to load (TIENKUNG_CONTROLS by default)
    """

    def __init__(
        self,
        camera_names: list[str] | None = None,
        controls: list[str] | None = None,
    ) -> None:
        self.camera_names = camera_names or ["camera_head"]
        self.controls = controls or TIENKUNG_CONTROLS

    def read(
        self,
        file_path: str,
        camera_frame: int | None = None,
        chunk_size: int | None = None,
    ) -> tuple[dict, dict]:
        """Read one HDF5 episode.

        Args:
            file_path: path to the .hdf5 file
            camera_frame: start timestep; if None, reads frame 0
            chunk_size: number of control timesteps to read (for action chunking)

        Returns:
            image_dict: {'color_images': {cam_name: np.ndarray(H, W, 3) RGB}}
            control_dict: {'puppet': {ctrl: np.ndarray(T, D)},
                           'master':  {ctrl: np.ndarray(T, D)}}
        """
        t = camera_frame if camera_frame is not None else 0

        image_dict: dict = {"color_images": {}}
        control_dict: dict = {"puppet": {}, "master": {}}

        with h5py.File(file_path, "r", libver="latest") as root:
            # --- images ---
            for cam in self.camera_names:
                img_path = f"camera_observations/color_images/{cam}"
                encoded = root[img_path][t]          # bytes / object
                image_dict["color_images"][cam] = _decode_jpeg(encoded)

            # --- control ---
            for arm in ("puppet", "master"):
                for ctrl in self.controls:
                    key = f"{arm}/{ctrl}/data"
                    if key not in root:
                        continue
                    if chunk_size is not None:
                        # Slice [t : t+chunk_size] — qpos[0] == state at timestep t
                        data = root[key][t: t + chunk_size]
                    else:
                        data = root[key][:]              # (T, D) full episode
                    control_dict[arm][ctrl] = data

        return image_dict, control_dict

    def episode_length(self, file_path: str) -> int:
        """Return the number of timesteps in an episode."""
        with h5py.File(file_path, "r", libver="latest") as root:
            ctrl_key = f"puppet/{self.controls[0]}/data"
            return root[ctrl_key].shape[0]
