"""Training utilities adapted from act_dp_ref/utils.py."""
import random
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互后端：仅 savefig 存 PNG，无需 GUI。
# 必须在 import pyplot 之前设置。否则默认 TkAgg 后端在训练中周期绘图后，
# Figure/Image/Variable 的 Tk 对象会被 DataLoader 子线程 GC，其 __del__ 在非主线程
# 调 Tcl 触发 "main thread is not in main loop" / "Tcl_AsyncDelete" → 进程 abort，
# 进而误报 "DataLoader worker killed by signal: Aborted"。
import matplotlib.pyplot as plt


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_dict_mean(epoch_dicts: list) -> dict:
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = sum(d[k] for d in epoch_dicts)
        result[k] = value_sum / num_items
    return result


def detach_dict(d: dict) -> dict:
    return {k: v.detach() for k, v in d.items()}


def _series_x(n_points, steps, num_steps):
    """X coordinates for a plotted series.

    If `steps` (the actual training-step of each point) is given, plot at those
    exact positions. Otherwise fall back to spreading n_points evenly over
    [0, num_steps-1] (legacy behaviour).
    """
    if steps is not None:
        return np.asarray(steps)
    if n_points <= 0:
        return np.asarray([])
    if n_points == 1:
        return np.asarray([0.0])
    return np.linspace(0, num_steps - 1, n_points)


def plot_history(train_history, validation_history, num_steps, ckpt_dir, seed,
                 val_steps=None, train_steps=None):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [s[key].item() for s in train_history]
        val_values = [s[key].item() for s in validation_history]
        train_x = _series_x(len(train_values), train_steps, num_steps)
        val_x   = _series_x(len(val_values),   val_steps,   num_steps)
        plt.plot(train_x, train_values, label="train")
        plt.plot(val_x, val_values, label="validation")
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")
