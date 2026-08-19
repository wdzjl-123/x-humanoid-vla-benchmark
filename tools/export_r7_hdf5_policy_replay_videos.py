#!/usr/bin/env python3
"""Export provenance-labeled R7 policy replays from recorded HDF5 observations.

The input action traces were produced by the latest R7 checkpoint on recorded
HDF5 observations. This tool renders those observations plus action diagnostics;
it does not launch a physics simulator or infer task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("ind_task_01", "ind_task_02", "ind_task_03", "lab_task_01", "lab_task_03")
ACTION_DIM = 26
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30
RGB_WIDTH = 960
RGB_HEIGHT = 540
HEADER_HEIGHT = 48
FOOTER_TOP = HEADER_HEIGHT + RGB_HEIGHT
ACTION_GROUPS = (
    ("left_arm", slice(0, 7), (53, 167, 156)),
    ("left_hand", slice(7, 13), (75, 134, 180)),
    ("right_arm", slice(13, 20), (226, 137, 56)),
    ("right_hand", slice(20, 26), (191, 85, 110)),
)
SOURCE_RUN_DEFAULT = ROOT / "reports/local_rollouts/r7_recorded_observation_replay_20260810T092919Z/run_summary.json"


class ExportError(RuntimeError):
    """Raised when replay evidence cannot be exported faithfully."""


@dataclass(frozen=True)
class SourceTiming:
    median_interval_ms: float
    source_hz: float
    output_fps: int


@dataclass(frozen=True)
class ReplayRecord:
    task: str
    model: dict[str, Any]
    episode: dict[str, Any]
    execution: dict[str, Any]
    action_artifact: Path


@dataclass(frozen=True)
class ActionTrace:
    actions: np.ndarray
    targets: np.ndarray
    inference_seconds: np.ndarray
    absolute_error: np.ndarray
    frame_mae: np.ndarray
    group_mae: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export five R7 recorded-observation policy replay videos from public HDF5 data."
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=SOURCE_RUN_DEFAULT,
        help="Existing R7 recorded-observation run_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory for videos and metadata. Defaults to a UTC timestamped reports directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_source_timing(timestamps: np.ndarray) -> SourceTiming:
    values = np.asarray(timestamps, dtype=np.float64).ravel()
    if values.size < 2:
        raise ExportError("HDF5 observation timestamps must contain at least two frames")
    intervals = np.diff(values)
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        raise ExportError("HDF5 observation timestamps have no positive interval")
    median_interval = float(np.median(intervals))
    source_hz = 1000.0 / median_interval
    if not 25.0 <= source_hz <= 35.0:
        raise ExportError(
            f"Unsupported observation cadence: median interval={median_interval:.3f} ms ({source_hz:.3f} Hz)"
        )
    return SourceTiming(median_interval_ms=median_interval, source_hz=source_hz, output_fps=OUTPUT_FPS)


def validate_action_arrays(actions: np.ndarray, targets: np.ndarray, inference_seconds: np.ndarray) -> ActionTrace:
    actions = np.asarray(actions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    inference_seconds = np.asarray(inference_seconds, dtype=np.float64).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ExportError(f"Expected predicted actions shaped (T, {ACTION_DIM}), got {actions.shape}")
    if targets.shape != actions.shape:
        raise ExportError(f"Demonstrator target shape {targets.shape} does not match predictions {actions.shape}")
    if inference_seconds.shape != (actions.shape[0],):
        raise ExportError(
            f"Inference timing shape {inference_seconds.shape} does not match action length {actions.shape[0]}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(targets).all() or not np.isfinite(inference_seconds).all():
        raise ExportError("Action trace contains non-finite predictions, targets, or inference timings")

    absolute_error = np.abs(actions - targets)
    group_mae = {
        name: absolute_error[:, action_slice].mean(axis=1)
        for name, action_slice, _ in ACTION_GROUPS
    }
    return ActionTrace(
        actions=actions,
        targets=targets,
        inference_seconds=inference_seconds,
        absolute_error=absolute_error,
        frame_mae=absolute_error.mean(axis=1),
        group_mae=group_mae,
    )


def load_replay_records(path: Path) -> tuple[dict[str, Any], list[ReplayRecord]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Unable to read recorded-observation run manifest {path}: {exc}") from exc

    raw_records = raw.get("records")
    if not isinstance(raw_records, list):
        raise ExportError("Recorded-observation run manifest has no records list")
    by_task: dict[str, ReplayRecord] = {}
    for item in raw_records:
        if not isinstance(item, dict):
            raise ExportError("Recorded-observation run manifest contains a non-object record")
        task = item.get("task")
        model = item.get("model")
        episode = item.get("episode")
        execution = item.get("execution")
        action_artifact = item.get("action_artifact")
        if not isinstance(task, str) or not isinstance(model, dict) or not isinstance(episode, dict):
            raise ExportError("Recorded-observation run manifest has an incomplete record")
        if not isinstance(execution, dict) or not isinstance(action_artifact, str):
            raise ExportError(f"Recorded-observation run manifest lacks execution data for {task}")
        if task in by_task:
            raise ExportError(f"Recorded-observation run manifest contains duplicate task {task}")
        if execution.get("physical_simulator") is not False:
            raise ExportError(f"Expected recorded-observation evidence for {task}, not a physics rollout")
        if execution.get("mode") != "recorded_observation_policy_replay":
            raise ExportError(f"Unexpected replay mode for {task}: {execution.get('mode')!r}")
        by_task[task] = ReplayRecord(
            task=task,
            model=model,
            episode=episode,
            execution=execution,
            action_artifact=Path(action_artifact),
        )

    unexpected = set(by_task) - set(TASKS)
    missing = set(TASKS) - set(by_task)
    if unexpected or missing:
        raise ExportError(f"Replay task set mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    records = [by_task[task] for task in TASKS]

    checkpoint_paths = {record.model.get("path") for record in records}
    checkpoint_hashes = {record.model.get("sha256") for record in records}
    if len(checkpoint_paths) != 1 or len(checkpoint_hashes) != 1:
        raise ExportError("All replay records must use the same R7 checkpoint and SHA-256")
    checkpoint_path = Path(str(next(iter(checkpoint_paths))))
    expected_sha = str(next(iter(checkpoint_hashes)))
    if not checkpoint_path.is_file():
        raise ExportError(f"R7 checkpoint does not exist: {checkpoint_path}")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_sha:
        raise ExportError(
            f"R7 checkpoint SHA-256 mismatch: manifest={expected_sha}, current={actual_sha}"
        )
    return raw, records


def decode_rgb(encoded: Any) -> np.ndarray:
    buffer = np.asarray(encoded, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ExportError("Unable to decode an HDF5 RGB JPEG frame")
    # This is the same explicit BGR-to-RGB conversion used by R7 training.
    return bgr[:, :, ::-1].copy()


def decode_depth(encoded: Any) -> np.ndarray:
    buffer = np.asarray(encoded, dtype=np.uint8)
    depth = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ExportError("Unable to decode an HDF5 depth PNG frame")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ExportError(f"Expected a two-dimensional depth frame, got shape {depth.shape}")
    return np.asarray(depth, dtype=np.float32)


def depth_display_range(depths: h5py.Dataset) -> tuple[float, float]:
    sample_count = min(32, len(depths))
    sample_indices = np.linspace(0, len(depths) - 1, sample_count, dtype=int)
    samples: list[np.ndarray] = []
    for index in np.unique(sample_indices):
        depth = decode_depth(depths[int(index)])
        valid = depth[depth > 0]
        if valid.size:
            stride = max(1, valid.size // 100_000)
            samples.append(valid[::stride])
    if not samples:
        return 0.0, 1.0
    values = np.concatenate(samples)
    lo, hi = np.percentile(values, (2, 98))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return 0.0, 1.0
    return float(lo), float(hi)


def colorize_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((depth - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return bgr[:, :, ::-1].copy()


_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def font(size: int) -> ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            loaded = ImageFont.truetype(candidate, size=size)
            _FONT_CACHE[size] = loaded
            return loaded
    loaded = ImageFont.load_default()
    _FONT_CACHE[size] = loaded
    return loaded


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, color: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font(size), fill=color)


def draw_action_panel(
    draw: ImageDraw.ImageDraw,
    trace: ActionTrace,
    frame_index: int,
) -> None:
    x0, y0, width, height = RGB_WIDTH, HEADER_HEIGHT + 180, OUTPUT_WIDTH - RGB_WIDTH, RGB_HEIGHT - 180
    draw.rectangle((x0, y0, x0 + width, y0 + height), fill=(22, 31, 39))
    draw.rectangle((x0, y0, x0 + width, y0 + 2), fill=(53, 167, 156))
    draw_text(draw, (x0 + 14, y0 + 12), "R7 ACTION DIAGNOSTICS", 18, (241, 246, 248))
    draw_text(draw, (x0 + 14, y0 + 40), "MAE vs recorded demonstrator target", 12, (190, 205, 211))

    value_scale = max(0.03, float(np.quantile(trace.frame_mae, 0.95)))
    row_y = y0 + 72
    for name, _, color in ACTION_GROUPS:
        value = float(trace.group_mae[name][frame_index])
        draw_text(draw, (x0 + 14, row_y), name.replace("_", " "), 14, (225, 233, 236))
        bar_left, bar_top, bar_width, bar_height = x0 + 14, row_y + 22, width - 28, 10
        draw.rectangle((bar_left, bar_top, bar_left + bar_width, bar_top + bar_height), fill=(49, 62, 70))
        bar_fill = int(min(1.0, value / value_scale) * bar_width)
        draw.rectangle((bar_left, bar_top, bar_left + bar_fill, bar_top + bar_height), fill=color)
        draw_text(draw, (x0 + width - 72, row_y), f"{value:.4f}", 13, color)
        row_y += 54

    running_mae = float(trace.frame_mae[: frame_index + 1].mean())
    inference_ms = float(trace.inference_seconds[frame_index] * 1000.0)
    draw_text(draw, (x0 + 14, y0 + height - 70), f"frame MAE: {trace.frame_mae[frame_index]:.4f}", 15, (241, 246, 248))
    draw_text(draw, (x0 + 14, y0 + height - 46), f"running MAE: {running_mae:.4f}", 15, (241, 246, 248))
    draw_text(draw, (x0 + 14, y0 + height - 22), f"R7 latency: {inference_ms:.1f} ms", 15, (241, 246, 248))


def render_frame(
    record: ReplayRecord,
    trace: ActionTrace,
    frame_index: int,
    timestamp_ms: float,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_lo: float,
    depth_hi: float,
) -> Image.Image:
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (16, 24, 31))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, OUTPUT_WIDTH, HEADER_HEIGHT), fill=(10, 18, 24))
    checkpoint_step = record.model.get("checkpoint_step", "unknown")
    checkpoint_sha = str(record.model.get("sha256", ""))[:12]
    draw_text(
        draw,
        (16, 11),
        f"{record.task} | R7 RGB-D ACT | frame {frame_index + 1}/{len(trace.actions)} | source t={timestamp_ms / 1000.0:.2f}s",
        18,
        (242, 247, 249),
    )
    draw_text(draw, (16, 31), f"checkpoint step {checkpoint_step} | sha256 {checkpoint_sha}", 12, (178, 199, 207))

    rgb_image = Image.fromarray(rgb, mode="RGB").resize((RGB_WIDTH, RGB_HEIGHT), Image.Resampling.LANCZOS)
    canvas.paste(rgb_image, (0, HEADER_HEIGHT))

    depth_image = Image.fromarray(colorize_depth(depth, depth_lo, depth_hi), mode="RGB")
    depth_image = depth_image.resize((OUTPUT_WIDTH - RGB_WIDTH, 180), Image.Resampling.NEAREST)
    canvas.paste(depth_image, (RGB_WIDTH, HEADER_HEIGHT))
    draw.rectangle((RGB_WIDTH, HEADER_HEIGHT, OUTPUT_WIDTH, HEADER_HEIGHT + 28), fill=(10, 18, 24))
    draw_text(draw, (RGB_WIDTH + 12, HEADER_HEIGHT + 6), "DEPTH INPUT (HDF5)", 14, (242, 247, 249))

    draw_action_panel(draw, trace, frame_index)

    draw.rectangle((0, FOOTER_TOP, OUTPUT_WIDTH, OUTPUT_HEIGHT), fill=(10, 18, 24))
    draw.rectangle((0, FOOTER_TOP, OUTPUT_WIDTH, FOOTER_TOP + 2), fill=(226, 137, 56))
    episode_name = Path(str(record.episode["path"])).name
    draw_text(draw, (16, FOOTER_TOP + 16), f"HDF5 episode: {episode_name}", 15, (226, 235, 238))
    draw_text(
        draw,
        (16, FOOTER_TOP + 43),
        "Recorded simulation observation + R7 action prediction. Predicted actions were not executed in physics.",
        14,
        (226, 235, 238),
    )
    draw_text(draw, (16, FOOTER_TOP + 67), "No success/failure result is asserted by this video.", 14, (241, 189, 109))
    return canvas


def encode_video(
    output_path: Path,
    frames: Any,
    frame_count: int,
) -> None:
    partial_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    if partial_path.exists():
        raise ExportError(f"Refusing to overwrite incomplete export: {partial_path}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-r",
        str(OUTPUT_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=16 * 1024 * 1024)
    try:
        assert process.stdin is not None
        for index, image in enumerate(frames, start=1):
            payload = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)).tobytes()
            process.stdin.write(payload)
            if index % 100 == 0 or index == frame_count:
                print(f"[export] {output_path.name}: rendered {index}/{frame_count} frames", flush=True)
        process.stdin.close()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        process.wait()
        raise ExportError(f"ffmpeg stopped while writing {output_path}: {stderr.strip()}") from exc
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise ExportError(f"ffmpeg failed for {output_path}: {stderr.strip()}")
    partial_path.replace(output_path)


def parse_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def validate_video(path: Path, expected_frames: int) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ExportError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ExportError(f"ffprobe returned no video stream for {path}") from exc

    frame_count = int(stream.get("nb_read_frames", "0"))
    fps = parse_rate(str(stream.get("avg_frame_rate", "0/1")))
    if stream.get("codec_name") != "h264":
        raise ExportError(f"Unexpected codec for {path}: {stream.get('codec_name')}")
    if stream.get("pix_fmt") != "yuv420p":
        raise ExportError(f"Unexpected pixel format for {path}: {stream.get('pix_fmt')}")
    if (int(stream.get("width", 0)), int(stream.get("height", 0))) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise ExportError(f"Unexpected video dimensions for {path}: {stream.get('width')}x{stream.get('height')}")
    if frame_count != expected_frames:
        raise ExportError(f"Video frame count mismatch for {path}: {frame_count} != {expected_frames}")
    if abs(fps - OUTPUT_FPS) > 0.01:
        raise ExportError(f"Unexpected video FPS for {path}: {fps}")

    capture = cv2.VideoCapture(str(path))
    sample_indices = sorted(set((0, expected_frames // 2, expected_frames - 1)))
    samples: list[dict[str, float | int]] = []
    try:
        if not capture.isOpened():
            raise ExportError(f"OpenCV cannot decode video {path}")
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ExportError(f"Unable to decode frame {index} from {path}")
            mean = float(frame.mean())
            std = float(frame.std())
            if mean <= 1.0 or std <= 1.0:
                raise ExportError(f"Decoded frame {index} from {path} appears blank")
            samples.append({"frame_index": index, "mean": round(mean, 4), "std": round(std, 4)})
    finally:
        capture.release()

    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 6),
        "decoded_samples": samples,
    }


def export_record(record: ReplayRecord, output_dir: Path) -> dict[str, Any]:
    episode_path = Path(str(record.episode.get("path", "")))
    if not episode_path.is_file():
        raise ExportError(f"HDF5 episode does not exist for {record.task}: {episode_path}")
    if not record.action_artifact.is_file():
        raise ExportError(f"R7 action artifact does not exist for {record.task}: {record.action_artifact}")

    with np.load(record.action_artifact) as data:
        required = ("actions", "demonstrator_targets", "inference_seconds")
        missing = [name for name in required if name not in data]
        if missing:
            raise ExportError(f"R7 action artifact for {record.task} is missing {missing}")
        trace = validate_action_arrays(data["actions"], data["demonstrator_targets"], data["inference_seconds"])

    with h5py.File(episode_path, "r") as h5:
        color_path = "camera_observations/color_images/camera_head"
        depth_path = "camera_observations/depth_images/camera_head"
        timestamp_path = "camera_observations/timestamp"
        missing_paths = [path for path in (color_path, depth_path, timestamp_path) if path not in h5]
        if missing_paths:
            raise ExportError(f"HDF5 episode for {record.task} is missing {missing_paths}")
        colors = h5[color_path]
        depths = h5[depth_path]
        timestamps = np.asarray(h5[timestamp_path], dtype=np.float64)
        expected_frames = len(trace.actions)
        if len(colors) != expected_frames or len(depths) != expected_frames or timestamps.size != expected_frames:
            raise ExportError(
                f"Timeline mismatch for {record.task}: colors={len(colors)}, depths={len(depths)}, "
                f"timestamps={timestamps.size}, actions={expected_frames}"
            )
        timing = derive_source_timing(timestamps)
        depth_lo, depth_hi = depth_display_range(depths)

        def rendered_frames():
            for index in range(expected_frames):
                yield render_frame(
                    record=record,
                    trace=trace,
                    frame_index=index,
                    timestamp_ms=float(timestamps[index]),
                    rgb=decode_rgb(colors[index]),
                    depth=decode_depth(depths[index]),
                    depth_lo=depth_lo,
                    depth_hi=depth_hi,
                )

        video_path = output_dir / f"{record.task}_r7_hdf5_policy_replay.mp4"
        encode_video(video_path, rendered_frames(), expected_frames)

    validation = validate_video(video_path, expected_frames)
    group_summary = {name: float(values.mean()) for name, values in trace.group_mae.items()}
    inference_ms = trace.inference_seconds * 1000.0
    sidecar = {
        "schema_version": 1,
        "task": record.task,
        "video": str(video_path.resolve()),
        "source": {
            "hdf5_episode": str(episode_path.resolve()),
            "episode_basename": episode_path.name,
            "data_type": "public_hdf5_simulated_trajectory",
            "frame_count": expected_frames,
            "source_timestamp_median_interval_ms": timing.median_interval_ms,
            "source_nominal_hz": timing.source_hz,
        },
        "model": {
            "name": record.model.get("name"),
            "checkpoint": str(Path(str(record.model["path"])).resolve()),
            "sha256": record.model["sha256"],
            "checkpoint_step": record.model.get("checkpoint_step"),
            "policy": record.model.get("policy"),
        },
        "execution": {
            "mode": "recorded_observation_policy_replay_video",
            "physical_simulator": False,
            "observation_source": "recorded public HDF5 RGB-D observations",
            "prediction_execution": "not sent to a physics simulator",
        },
        "metrics": {
            "mean_action_mae": float(trace.frame_mae.mean()),
            "per_group_mae": group_summary,
            "inference_ms_mean": float(inference_ms.mean()),
            "inference_ms_p95": float(np.percentile(inference_ms, 95)),
            "inference_ms_max": float(inference_ms.max()),
        },
        "result": {
            "state": "not_applicable",
            "success": None,
            "failure": None,
            "reason": "Recorded-observation policy replay does not execute actions in physics and cannot establish task success or failure.",
        },
        "validation": validation,
    }
    sidecar_path = output_dir / f"{record.task}_r7_hdf5_policy_replay.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "reports/local_rollouts" / f"r7_hdf5_policy_replay_videos_{timestamp}"


def export_all(source_run: Path, output_dir: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    if not source_run.is_file():
        raise ExportError(f"Recorded-observation run manifest does not exist: {source_run}")
    if output_dir.exists():
        raise ExportError(f"Refusing to overwrite existing output directory: {output_dir}")
    source_manifest, records = load_replay_records(source_run)
    output_dir.mkdir(parents=True, exist_ok=False)
    exported: list[dict[str, Any]] = []
    try:
        for record in records:
            print(f"[export] Starting {record.task}", flush=True)
            exported.append(export_record(record, output_dir))
    except Exception:
        print(f"[export] Failed. Incomplete files, if any, remain in {output_dir} for diagnosis.", file=sys.stderr)
        raise

    run_manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_recorded_observation_run": str(source_run),
        "source_run_id": source_manifest.get("run_id"),
        "evidence_boundary": "All videos replay recorded HDF5 observations with R7 action predictions. They are not physics rollouts and contain no success/failure result.",
        "model": exported[0]["model"],
        "exports": [
            {
                "task": item["task"],
                "video": item["video"],
                "sidecar": str((output_dir / f"{item['task']}_r7_hdf5_policy_replay.json").resolve()),
                "validation": item["validation"],
            }
            for item in exported
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[export] Complete: {output_dir}", flush=True)
    return run_manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir()
    export_all(args.source_run, output_dir)


if __name__ == "__main__":
    try:
        main()
    except ExportError as exc:
        print(f"[export] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
