#!/usr/bin/env python3
"""Prepare ImageFolder dataset for fruit classification."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "fruit_classification"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fruit classification dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA,
        help="Target ImageFolder root (train/val/test).",
    )
    args = parser.parse_args()
    root = args.data_root
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
    print(f"[fruit_classification] data root: {root}")
    print("Expected layout: {root}/{train,val,test}/<class_name>/*.jpg")
    print("Legacy source: task1_fruit_classification/dataset_detection/")
    print("See docs/MIGRATION.md for migration steps.")


if __name__ == "__main__":
    main()
