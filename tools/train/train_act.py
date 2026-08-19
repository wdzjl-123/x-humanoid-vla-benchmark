"""
ACT training entry point.

Adapted from act_dp_ref/train_algo_h5_v2.py — single-GPU, no wandb, no DDP complexity.

Usage:
  python3 tools/train/train_act.py \\
      --task-dir data/ind_task_01 \\
      --ckpt-dir ./checkpoints/task01_act \\
      --num-steps 50000 \\
      --batch-size 8 \\
      --chunk-size 50

  # 640×480 high-resolution mode:
  python3 tools/train/train_act.py ... --img-w 640 --img-h 480

The script saves:
  <ckpt_dir>/dataset_stats.pkl    — normalization stats (required by inference)
  <ckpt_dir>/agent_best.ckpt      — best val-loss checkpoint (only if val/ is non-empty)
  <ckpt_dir>/policy_last.ckpt     — final checkpoint
  <ckpt_dir>/policy_step_N.ckpt   — periodic checkpoints
"""
from __future__ import annotations
import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from tools.train.dataset import load_data
from tools.train.utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
from tools.policies.act_policy import build_act_model


def setup_logger(ckpt_dir: str) -> logging.Logger:
    logger = logging.getLogger("train_act")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-6s  %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(ckpt_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def forward_pass(batch, policy):
    image_data, depth_data, qpos_data, action_data, is_pad = batch
    image_data = image_data.cuda()
    qpos_data  = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad     = is_pad.cuda()
    depth_data = None   # not used by ACT

    return policy(qpos_data, image_data, depth_data, action_data, is_pad)


def repeater(loader, total_steps):
    step = 0
    while step < total_steps:
        for batch in loader:
            yield batch
            step += 1
            if step >= total_steps:
                return


def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = setup_logger(args.ckpt_dir)
    set_seed_everywhere(args.seed)

    # --- robot config ---
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

    # --- data ---
    logger.info("Loading dataset …")
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
    logger.info(f"Saved norm stats → {stats_path}")

    # --- model ---
    policy = build_act_model(
        chunk_size=args.chunk_size,
        camera_names=["camera_head"],
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        nheads=args.nheads,
        action_dim=26,
        state_dim=26,
        kl_weight=args.kl_weight,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
    )
    policy.cuda()

    optimizer = policy.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    # --- resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        policy.deserialize(ckpt["nets"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from {args.resume} at step {start_step}")

    # --- training loop ---
    train_history = []
    val_history   = []
    val_steps     = []   # actual training step of each validation point (for correct x-axis)
    min_val_loss  = float("inf")
    best_ckpt_info = None
    best_step = start_step

    train_iter = repeater(train_loader, args.num_steps)

    logger.info(f"Training for {args.num_steps} steps (start={start_step})")
    for step in tqdm(range(start_step, args.num_steps)):

        # validation (skipped when val/ is empty — e.g. the unified all-for-training release)
        if val_loader is not None and step % args.validate_every == 0:
            policy.eval()
            val_dicts = []
            with torch.inference_mode():
                for i, batch in enumerate(val_loader):
                    d = forward_pass(batch, policy)
                    val_dicts.append(d)
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
                best_ckpt_info = {
                    "step": step + 1,
                    "nets": policy.serialize(),
                    "loss": val_loss,
                    "min_val_loss": min_val_loss,
                }

        # train step
        policy.train()
        optimizer.zero_grad()
        batch = next(train_iter)
        fwd = forward_pass(batch, policy)
        loss = fwd["loss"].mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_history.append(detach_dict(fwd))

        # periodic checkpoint
        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"policy_step_{step}.ckpt")
            torch.save({"step": step + 1, "nets": policy.serialize(), "loss": loss.item()}, ckpt_path)
            logger.info(f"Saved checkpoint → {ckpt_path}")
            if len(train_history) > 0 and len(val_history) > 0:
                plot_history(train_history, val_history, args.num_steps, args.ckpt_dir, args.seed,
                             val_steps=val_steps,
                             train_steps=range(start_step, start_step + len(train_history)))

    # --- save final + best ---
    last_path = os.path.join(args.ckpt_dir, "policy_last.ckpt")
    torch.save({"step": args.num_steps, "nets": policy.serialize()}, last_path)
    logger.info(f"Saved last checkpoint → {last_path}")

    if best_ckpt_info is not None:
        best_path = os.path.join(args.ckpt_dir, "agent_best.ckpt")
        torch.save(best_ckpt_info, best_path)
        logger.info(f"Best checkpoint (step {best_step}, val_loss={min_val_loss:.5f}) → {best_path}")

    if len(train_history) > 0 and len(val_history) > 0:
        plot_history(train_history, val_history, args.num_steps, args.ckpt_dir, args.seed,
                     val_steps=val_steps,
                     train_steps=range(start_step, start_step + len(train_history)))

    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Train ACT on RoboMIND2.0 data")
    p.add_argument("--task-dir", required=True, help="Task directory containing train/ and val/ (see README §2)")
    p.add_argument("--ckpt-dir", required=True, help="Output directory for checkpoints")

    # training
    p.add_argument("--num-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--use-aug", action="store_true", default=False)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    # model
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dim-feedforward", type=int, default=3200)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=7)
    p.add_argument("--nheads", type=int, default=8)

    # image resolution (320×240 for original ACT; 640×480 for v1)
    p.add_argument("--img-w", type=int, default=320)
    p.add_argument("--img-h", type=int, default=240)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
