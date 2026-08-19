"""Evaluate a multi-task RGB-D ACT checkpoint on one manifest partition."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.multitask import ACTION_DIM, MULTITASK_STATE_DIM, TASK_NAMES
from tools.policies.act_policy import build_act_model
from tools.train.multitask_dataset import MultiTaskEpisodicDataset, load_episode_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multi-task RGB-D ACT checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_policy(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its serialized model config")
    if int(config.get("state_dim", -1)) != MULTITASK_STATE_DIM:
        raise ValueError("Checkpoint is not compatible with the unified 45D multi-task state")
    if not config.get("use_depth_image", False):
        raise ValueError("Checkpoint is not a depth-enabled multi-task ACT model")

    policy = build_act_model(
        chunk_size=int(config["chunk_size"]),
        backbone=str(config.get("backbone", "resnet18")),
        hidden_dim=int(config.get("hidden_dim", 512)),
        dim_feedforward=int(config.get("dim_feedforward", 3200)),
        enc_layers=int(config.get("enc_layers", 4)),
        dec_layers=int(config.get("dec_layers", 7)),
        nheads=int(config.get("nheads", 8)),
        action_dim=int(config.get("action_dim", ACTION_DIM)),
        state_dim=MULTITASK_STATE_DIM,
        kl_weight=int(config.get("kl_weight", 10)),
        lr=float(config.get("lr", 1e-4)),
        lr_backbone=float(config.get("lr_backbone", 1e-5)),
        device=str(device),
        use_depth_image=True,
    )
    policy.deserialize(checkpoint["nets"])
    policy.to(device)
    policy.eval()
    return policy, config, int(checkpoint.get("step", -1))


def evaluate_task(
    policy,
    refs,
    norm_stats: dict,
    config: dict,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    device: torch.device,
) -> dict:
    dataset = MultiTaskEpisodicDataset(
        refs,
        norm_stats,
        chunk_size=int(config["chunk_size"]),
        image_width=int(config["img_w"]),
        image_height=int(config["img_h"]),
        max_depth_meters=float(config.get("max_depth_meters", 5.0)),
        use_aug=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
    samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image, depth, state, actions, is_pad = batch
            metrics = policy(
                state.to(device, non_blocking=True),
                image.to(device, non_blocking=True),
                depth.to(device, non_blocking=True),
                actions.to(device, non_blocking=True),
                is_pad.to(device, non_blocking=True),
            )
            count = int(state.shape[0])
            samples += count
            for name in totals:
                totals[name] += float(metrics[name].mean()) * count
            if max_batches and batch_index + 1 >= max_batches:
                break
    if samples == 0:
        raise RuntimeError("Evaluation produced no samples")
    return {
        "episodes": len(refs),
        "samples": samples,
        "loss": totals["loss"] / samples,
        "normalized_action_l1": totals["l1"] / samples,
        "kl": totals["kl"] / samples,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_batches_per_task < 0:
        raise ValueError("batch-size must be positive; worker and batch limits must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = Path(args.model_path).resolve().parent
    stats_path = checkpoint_dir / "dataset_stats.pkl"
    if not stats_path.is_file():
        raise FileNotFoundError(f"dataset_stats.pkl not found next to checkpoint: {stats_path}")
    with stats_path.open("rb") as handle:
        norm_stats = pickle.load(handle)
    if len(norm_stats["qpos_mean"]) != MULTITASK_STATE_DIM:
        raise ValueError("Checkpoint statistics are incompatible with the unified state")

    policy, config, checkpoint_step = load_policy(args.model_path, device)
    split_refs = load_episode_split_manifest(args.data_root, args.split_manifest, TASK_NAMES)
    selected_refs = split_refs[args.split]
    by_task = {
        task_name: [ref for ref in selected_refs if ref.task_name == task_name]
        for task_name in TASK_NAMES
    }
    per_task = {
        task_name: evaluate_task(
            policy,
            refs,
            norm_stats,
            config,
            args.batch_size,
            args.num_workers,
            args.max_batches_per_task,
            device,
        )
        for task_name, refs in by_task.items()
    }
    total_samples = sum(result["samples"] for result in per_task.values())
    summary = {
        name: sum(result[name] * result["samples"] for result in per_task.values()) / total_samples
        for name in ("loss", "normalized_action_l1", "kl")
    }
    summary["episodes"] = sum(result["episodes"] for result in per_task.values())
    summary["samples"] = total_samples
    result = {
        "model_path": str(Path(args.model_path)),
        "checkpoint_step": checkpoint_step,
        "split_manifest": str(Path(args.split_manifest)),
        "split": args.split,
        "summary": summary,
        "per_task": per_task,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
