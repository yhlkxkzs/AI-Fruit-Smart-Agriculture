#!/usr/bin/env python3
"""Evaluate fruit classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TASK_DIR / "configs" / "default.yaml"
DEFAULT_RUNS = TASK_DIR / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fruit classification.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = parser.parse_args()
    print(f"[fruit_classification] eval split={args.split} config={args.config}")
    print("TODO: wire to shared/eval when implemented.")


if __name__ == "__main__":
    main()
