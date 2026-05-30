#!/usr/bin/env python3
"""Train fruit classification (species-only or multistate dual-head)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TASK_DIR / "configs" / "default.yaml"
MULTISTATE_CONFIG = TASK_DIR / "configs" / "multistate.yaml"


def config_task_type(config_path: Path) -> str:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("task_type:"):
            return line.split(":", 1)[1].strip()
    return "classification"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fruit classification.")
    parser.add_argument("--config", type=Path, default=MULTISTATE_CONFIG)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = args.model_config or args.config
    task_type = config_task_type(config)

    if task_type == "classification_multistate":
        cmd = [
            sys.executable,
            str(TASK_DIR / "scripts" / "train_multistate.py"),
            "--config",
            str(config),
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        raise SystemExit(subprocess.call(cmd))

    print(f"[fruit_classification] config={config} task_type={task_type}")
    print("Species-only ImageFolder training: use legacy train_efficientnet_lite0.py with data/fruit_classification/")


if __name__ == "__main__":
    main()
