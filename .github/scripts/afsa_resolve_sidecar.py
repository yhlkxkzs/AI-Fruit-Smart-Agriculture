#!/usr/bin/env python3
"""Resolve sidecar JSON for each changed incoming image path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def sidecar_path(image_path: str) -> Path:
    p = Path(image_path)
    return p.with_suffix(".afsa.json")


def route_from_sidecar(sc: Path, fallback_image: str) -> dict | None:
    if not sc.is_file():
        return None
    meta = json.loads(sc.read_text(encoding="utf-8"))
    image_path = meta.get("image_path") or fallback_image
    return {
        "image_path": image_path,
        "task_type": meta.get("task_type", "fruit_category"),
        "github_model_target_id": meta.get("github_model_target_id", "category_all"),
        "afsa_detection_id": meta.get("afsa_detection_id"),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: afsa_resolve_sidecar.py /tmp/images.txt", file=sys.stderr)
        sys.exit(2)
    images_file = Path(sys.argv[1])
    routes: list[dict] = []
    seen: set[str] = set()
    for line in images_file.read_text(encoding="utf-8").splitlines():
        image_path = line.strip()
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        sc = REPO_ROOT / sidecar_path(image_path)
        row = route_from_sidecar(sc, image_path)
        if row:
            routes.append(row)
            continue
        # App 可能先 push 图片、后 push sidecar；无 sidecar 时仍推理（无 detection_id）
        print(f"[warn] missing sidecar: {sc} — fallback route without afsa_detection_id")
        routes.append(
            {
                "image_path": image_path,
                "task_type": "fruit_category",
                "github_model_target_id": "category_all",
                "afsa_detection_id": None,
            }
        )
    out = {"routes": routes}
    Path("/tmp/afsa_routes.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] routes={len(routes)}")
    if not routes:
        sys.exit(1)


if __name__ == "__main__":
    main()
