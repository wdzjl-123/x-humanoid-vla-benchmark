"""
Train a Conditional Flow Matching action generator on challenge HDF5 data.

The saved checkpoint is served by:
  python3 tools/policy_infer.py --policy flow --model-path checkpoints/<task>_flow/policy_last.ckpt
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.policies.flow_policy import build_flow_model
from tools.train.dataset import load_data
from tools.train.utils import compute_dict_mean, detach_dict, plot_history, set_seed_everywhere


def setup_logger(ckpt_dir: str) -> logging.Logger:
    logger = logging.getLogger("train_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-6s  %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(ckpt_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def repeater(loader, total_steps):
    step = 0
    while step < total_steps:
        for batch in loader:
            yield batch
            step += 1
            if step >= total_steps:
                return


def forward_pass(batch, policy, device: torch.device):
    image_data, _depth_data, qpos_data, action_data, is_pad = batch
    image_data = image_data.to(device, non_blocking=True)
    qpos_data = qpos_data.to(device, non_blocking=True)
    action_data = action_data.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def build_config(args) -> dict:
    return {
        "chunk_size": args.chunk_size,
        "action_dim": 26,
        "state_dim": 26,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "nheads": args.nheads,
        "dropout": args.dropout,
        "img_w": args.img_w,
        "img_h": args.img_h,
    }


def build_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
    step: int,
    loss: float | None = None,
) -> dict:
    checkpoint = {
        "step": step,
        "nets": policy.serialize(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if loss is not None:
        checkpoint["loss"] = loss
    return checkpoint


def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = setup_logger(args.ckpt_dir)
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    robot_infor = {
        "camera_names": ["camera_head"],
        "camera_sensors": ["color_images"],
        "arms": ["puppet", "master"],
        "controls": [
            "arm_left_position_align",
            "end_effector_left_position_align",
            "arm_right_position_align",
            "end_effector_right_position_align",
        ],
    }

    logger.info("Loading dataset ...")
    train_loader, val_loader, norm_stats = load_data(
        task_dir=args.task_dir,
        robot_infor=robot_infor,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        chunk_size=args.chunk_size,
        use_aug=args.use_aug,
        num_workers=args.num_workers,
        img_w=args.img_w,
        img_h=args.img_h,
    )

    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)
    logger.info(f"Saved norm stats -> {stats_path}")

    config = build_config(args)
    policy = build_flow_model(**{k: config[k] for k in (
        "chunk_size", "action_dim", "state_dim", "hidden_dim", "num_layers", "nheads", "dropout"
    )})
    policy.to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.deserialize(ckpt["nets"])
        start_step = int(ckpt.get("step", 0))
        if "optimizer" in ckpt and "scheduler" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info("Restored optimizer and scheduler state")
        else:
            logger.warning(
                "Resume checkpoint has no optimizer/scheduler state; "
                "continuing with freshly initialized optimizer state"
            )
            if start_step > 0:
                scheduler.step(start_step)
                logger.info("Advanced fresh scheduler to resumed step %d", start_step)
        logger.info(f"Resumed from {args.resume} at step {start_step}")

    train_history = []
    val_history = []
    val_steps = []
    min_val_loss = float("inf")
    best_ckpt_info = None
    best_step = start_step

    train_iter = repeater(train_loader, args.num_steps)
    logger.info(f"Training FlowPolicy for {args.num_steps} steps (start={start_step})")

    for step in tqdm(range(start_step, args.num_steps)):
        if val_loader is not None and step % args.validate_every == 0:
            policy.eval()
            val_dicts = []
            with torch.inference_mode():
                for i, batch in enumerate(val_loader):
                    val_dicts.append(forward_pass(batch, policy, device))
                    if i >= 50:
                        break
            val_summary = compute_dict_mean(val_dicts)
            val_history.append(val_summary)
            val_steps.append(step)
            val_loss = val_summary["loss"].item()
            logger.info(f"step {step:6d}  val_loss={val_loss:.5f}")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_step = step
                best_ckpt_info = build_checkpoint(
                    policy, optimizer, scheduler, config, step + 1, val_loss
                )
                best_ckpt_info["min_val_loss"] = min_val_loss

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        batch = next(train_iter)
        fwd = forward_pass(batch, policy, device)
        loss = fwd["loss"].mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        train_history.append(detach_dict(fwd))
        if step % args.log_every == 0:
            logger.info(
                "step %6d  loss=%.6f  flow_mse=%.6f  lr=%.3e",
                step,
                loss.item(),
                fwd["flow_mse"].item(),
                optimizer.param_groups[0]["lr"],
            )

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"policy_step_{step}.ckpt")
            torch.save(
                build_checkpoint(policy, optimizer, scheduler, config, step + 1, loss.item()),
                ckpt_path,
            )
            logger.info(f"Saved checkpoint -> {ckpt_path}")
            if train_history and val_history:
                plot_history(
                    train_history,
                    val_history,
                    args.num_steps,
                    args.ckpt_dir,
                    args.seed,
                    val_steps=val_steps,
                    train_steps=range(start_step, start_step + len(train_history)),
                )

    last_path = os.path.join(args.ckpt_dir, "policy_last.ckpt")
    torch.save(build_checkpoint(policy, optimizer, scheduler, config, args.num_steps), last_path)
    logger.info(f"Saved last checkpoint -> {last_path}")

    if best_ckpt_info is not None:
        best_path = os.path.join(args.ckpt_dir, "agent_best.ckpt")
        torch.save(best_ckpt_info, best_path)
        logger.info(f"Best checkpoint (step {best_step}, val_loss={min_val_loss:.5f}) -> {best_path}")

    if train_history and val_history:
        plot_history(
            train_history,
            val_history,
            args.num_steps,
            args.ckpt_dir,
            args.seed,
            val_steps=val_steps,
            train_steps=range(start_step, start_step + len(train_history)),
        )

    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional Flow Matching policy")
    p.add_argument("--task-dir", required=True, help="Task directory containing train/ and optional val/")
    p.add_argument("--ckpt-dir", required=True, help="Output directory for checkpoints")

    p.add_argument("--num-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--use-aug", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--img-w", type=int, default=320)
    p.add_argument("--img-h", type=int, default=240)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
