"""Train one RGB-D, pose, task-conditioned ACT checkpoint for all VLA tasks."""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.multitask import ACTION_DIM, MULTITASK_STATE_DIM, TASK_NAMES
from tools.policies.act_policy import build_act_model
from tools.train.multitask_dataset import MultiTaskEpisodicDataset, RobustAugmentationConfig, build_multitask_dataloaders
from tools.train.utils import compute_dict_mean


def setup_logger(ckpt_dir: str) -> logging.Logger:
    logger = logging.getLogger("train_multitask_act")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(levelname)-6s  %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(ckpt_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def repeater(loader, total_steps: int):
    completed = 0
    while completed < total_steps:
        for batch in loader:
            yield batch
            completed += 1
            if completed >= total_steps:
                return


def forward_pass(batch, policy, device: torch.device) -> dict[str, torch.Tensor]:
    image, depth, state, actions, is_pad = batch
    return policy(
        state.to(device, non_blocking=True),
        image.to(device, non_blocking=True),
        depth.to(device, non_blocking=True),
        actions.to(device, non_blocking=True),
        is_pad.to(device, non_blocking=True),
    )


def evaluate_validation_loader(
    policy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    metrics = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            metrics.append(forward_pass(batch, policy, device))
            if max_batches > 0 and batch_index >= max_batches - 1:
                break
    if not metrics:
        raise RuntimeError("Validation loader produced no batches")
    return compute_dict_mean(metrics)


def validate_by_task(
    policy,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, dict[str, torch.Tensor]]:
    dataset = val_loader.dataset
    if not isinstance(dataset, MultiTaskEpisodicDataset):
        raise TypeError("Expected a MultiTaskEpisodicDataset validation loader")

    per_task = {}
    for task_name in dataset.task_names:
        task_dataset = MultiTaskEpisodicDataset(
            dataset.by_task[task_name],
            dataset.norm_stats,
            dataset.chunk_size,
            dataset.image_width,
            dataset.image_height,
            dataset.max_depth_meters,
            use_aug=False,
        )
        task_loader = DataLoader(
            task_dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=True,
        )
        per_task[task_name] = evaluate_validation_loader(
            policy, task_loader, device, max_batches
        )
    return per_task


def build_config(args: argparse.Namespace) -> dict:
    return {
        "task_names": tuple(args.tasks),
        "chunk_size": args.chunk_size,
        "action_dim": ACTION_DIM,
        "state_dim": MULTITASK_STATE_DIM,
        "backbone": args.backbone,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "kl_weight": args.kl_weight,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "img_w": args.img_w,
        "img_h": args.img_h,
        "max_depth_meters": args.max_depth_meters,
        "use_depth_image": True,
        "split_manifest": args.split_manifest,
        "task_sampling_weights": (
            None if args.task_sampling_weights is None
            else dict(zip(args.tasks, args.task_sampling_weights))
        ),
        "augmentation": {
            "enabled": args.use_aug,
            "translate_fraction": args.aug_translate_fraction,
            "scale_jitter": args.aug_scale_jitter,
            "depth_noise_std": args.depth_noise_std,
            "depth_dropout_probability": args.depth_dropout_probability,
            "normalized_state_noise_std": args.normalized_state_noise_std,
            "motion_sampling_alpha": args.motion_sampling_alpha,
            "motion_sampling_max_ratio": args.motion_sampling_max_ratio,
        },
    }


def save_checkpoint(path: str, policy, step: int, config: dict, loss: float | None = None) -> None:
    checkpoint = {"step": step, "nets": policy.serialize(), "config": config}
    if loss is not None:
        checkpoint["loss"] = float(loss)
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if not 0 <= args.aug_translate_fraction < 0.5:
        raise ValueError("aug-translate-fraction must be in [0, 0.5)")
    if not 0 <= args.aug_scale_jitter < 0.5:
        raise ValueError("aug-scale-jitter must be in [0, 0.5)")
    if args.depth_noise_std < 0 or not 0 <= args.depth_dropout_probability < 1:
        raise ValueError("depth augmentation values must be non-negative with dropout below one")
    if args.max_val_batches < 0:
        raise ValueError("max-val-batches must be non-negative")
    if args.task_sampling_weights is not None:
        if len(args.task_sampling_weights) != len(args.tasks):
            raise ValueError("task-sampling-weights must have one positive integer per selected task")
        if any(weight <= 0 for weight in args.task_sampling_weights):
            raise ValueError("task-sampling-weights must contain only positive integers")
        task_sampling_weights = dict(zip(args.tasks, args.task_sampling_weights))
    else:
        task_sampling_weights = None
    if args.normalized_state_noise_std < 0:
        raise ValueError("normalized-state-noise-std must be non-negative")
    if args.motion_sampling_alpha < 0 or args.motion_sampling_max_ratio < 1:
        raise ValueError("motion sampling requires non-negative alpha and max ratio at least one")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = setup_logger(args.ckpt_dir)
    device = torch.device(args.device)
    config = build_config(args)

    logger.info("Loading unified RGB-D dataset for tasks: %s", ", ".join(args.tasks))
    augmentation_config = RobustAugmentationConfig(
        translate_fraction=args.aug_translate_fraction,
        scale_jitter=args.aug_scale_jitter,
        depth_noise_std=args.depth_noise_std,
        depth_dropout_probability=args.depth_dropout_probability,
        normalized_state_noise_std=args.normalized_state_noise_std,
    )
    train_loader, val_loader, norm_stats = build_multitask_dataloaders(
        data_root=args.data_root,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        chunk_size=args.chunk_size,
        image_width=args.img_w,
        image_height=args.img_h,
        max_depth_meters=args.max_depth_meters,
        task_names=tuple(args.tasks),
        val_ratio=args.val_ratio,
        task_sampling_weights=task_sampling_weights,
        split_seed=args.split_seed,
        use_aug=args.use_aug,
        augmentation_config=augmentation_config,
        motion_sampling_alpha=args.motion_sampling_alpha,
        motion_sampling_max_ratio=args.motion_sampling_max_ratio,
        num_workers=args.num_workers,
        max_episodes_per_task=args.max_episodes_per_task,
        split_manifest=args.split_manifest,
    )
    with open(os.path.join(args.ckpt_dir, "dataset_stats.pkl"), "wb") as handle:
        pickle.dump(norm_stats, handle)

    policy = build_act_model(
        chunk_size=args.chunk_size,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        nheads=args.nheads,
        action_dim=ACTION_DIM,
        state_dim=MULTITASK_STATE_DIM,
        kl_weight=args.kl_weight,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        device=str(device),
        use_depth_image=True,
    )
    policy.to(device)
    optimizer = policy.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        if checkpoint_config.get("state_dim", MULTITASK_STATE_DIM) != MULTITASK_STATE_DIM:
            raise ValueError("Resume checkpoint is not a compatible unified multi-task ACT model")
        policy.deserialize(checkpoint["nets"])
        start_step = int(checkpoint.get("step", 0))
        logger.info("Resumed weights from %s at step %d", args.resume, start_step)

    best_loss = float("inf")
    train_iter = repeater(train_loader, args.num_steps - start_step)
    logger.info("Training unified ACT for %d steps (start=%d)", args.num_steps, start_step)
    for step in tqdm(range(start_step, args.num_steps)):
        if val_loader is not None and step % args.validate_every == 0:
            policy.eval()
            per_task_summary = validate_by_task(
                policy, val_loader, device, args.max_val_batches
            )
            val_loss = sum(
                float(summary["loss"]) for summary in per_task_summary.values()
            ) / len(per_task_summary)
            logger.info("step %6d  val_macro_loss=%.6f", step, val_loss)
            for task_name in args.tasks:
                summary = per_task_summary[task_name]
                logger.info(
                    "step %6d  val_%s_loss=%.6f  l1=%.6f",
                    step,
                    task_name,
                    float(summary["loss"]),
                    float(summary["l1"]),
                )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    os.path.join(args.ckpt_dir, "agent_best.ckpt"),
                    policy,
                    step,
                    config,
                    val_loss,
                )

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        losses = forward_pass(next(train_iter), policy, device)
        loss = losses["loss"].mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            logger.info(
                "step %6d  loss=%.6f  l1=%.6f  kl=%.6f  lr=%.3e",
                step,
                loss.item(),
                losses["l1"].item(),
                losses["kl"].item(),
                optimizer.param_groups[0]["lr"],
            )
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.ckpt_dir, f"policy_step_{step}.ckpt"),
                policy,
                step + 1,
                config,
                loss.item(),
            )

    save_checkpoint(
        os.path.join(args.ckpt_dir, "policy_last.ckpt"),
        policy,
        args.num_steps,
        config,
    )
    logger.info("Saved unified checkpoint -> %s", os.path.join(args.ckpt_dir, "policy_last.ckpt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one task-conditioned RGB-D ACT checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--img-w", type=int, default=320)
    parser.add_argument("--img-h", type=int, default=240)
    parser.add_argument("--max-depth-meters", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Episode-level train/val/test manifest; when set, --val-ratio is ignored",
    )
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument(
        "--max-val-batches", type=int, default=0,
        help="Maximum validation batches; 0 evaluates the complete validation split",
    )
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-aug", action="store_true")
    parser.add_argument("--aug-translate-fraction", type=float, default=0.03)
    parser.add_argument("--aug-scale-jitter", type=float, default=0.03)
    parser.add_argument("--depth-noise-std", type=float, default=0.01)
    parser.add_argument("--depth-dropout-probability", type=float, default=0.01)
    parser.add_argument("--normalized-state-noise-std", type=float, default=0.01)
    parser.add_argument("--motion-sampling-alpha", type=float, default=2.0)
    parser.add_argument("--motion-sampling-max-ratio", type=float, default=5.0)
    parser.add_argument(
        "--task-sampling-weights", nargs="+", type=int, default=None,
        help="Positive integer sampling weights in --tasks order; default keeps tasks uniform",
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
