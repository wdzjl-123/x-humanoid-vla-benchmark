"""Create a reproducible episode-level train/val/test manifest for multi-task ACT."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.multitask import TASK_NAMES
from tools.train.multitask_dataset import create_episode_split_manifest, write_episode_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an episode-level multi-task split manifest")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_episode_split_manifest(
        data_root=args.data_root,
        task_names=TASK_NAMES,
        split_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    write_episode_split_manifest(manifest, args.output)
    for task_name in TASK_NAMES:
        task_splits = manifest["tasks"][task_name]
        print(
            f"{task_name}: train={len(task_splits['train'])} "
            f"val={len(task_splits['val'])} test={len(task_splits['test'])}"
        )
    print(f"Wrote split manifest: {args.output}")


if __name__ == "__main__":
    main()
