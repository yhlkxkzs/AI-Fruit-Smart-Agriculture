#!/usr/bin/env python3
"""Emit unique incoming image paths changed in HEAD vs HEAD^ (images + sidecar targets)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_diff_paths(spec: str) -> list[str]:
    r = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^", "HEAD", "--", spec],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def main() -> None:
    paths: list[str] = []
    paths.extend(git_diff_paths("incoming/**/*.jpg"))
    paths.extend(git_diff_paths("incoming/**/*.jpeg"))
    paths.extend(git_diff_paths("incoming/**/*.png"))
    paths.extend(git_diff_paths("incoming/**/*.webp"))
    for sc in git_diff_paths("incoming/**/*.afsa.json"):
        p = REPO_ROOT / sc
        if not p.is_file():
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        img = meta.get("image_path")
        if img:
            paths.append(img)
    seen: set[str] = set()
    for p in sorted(paths):
        if p and p not in seen:
            seen.add(p)
            print(p)


if __name__ == "__main__":
    main()
