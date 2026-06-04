#!/usr/bin/env python3
"""Resolve sidecar JSON for each changed incoming image path."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def sidecar_path(image_path: str) -> Path:
    p = Path(image_path)
    return p.with_suffix(".afsa.json")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: afsa_resolve_sidecar.py /tmp/images.txt", file=sys.stderr)
        sys.exit(2)
    images_file = Path(sys.argv[1])
    routes: list[dict] = []
    for line in images_file.read_text(encoding="utf-8").splitlines():
        image_path = line.strip()
        if not image_path:
            continue
        sc = sidecar_path(image_path)
        if not sc.is_file():
            print(f"[warn] missing sidecar: {sc}")
            continue
        meta = json.loads(sc.read_text(encoding="utf-8"))
        routes.append(
            {
                "image_path": meta.get("image_path") or image_path,
                "task_type": meta.get("task_type", "fruit_category"),
                "github_model_target_id": meta.get("github_model_target_id", "category_all"),
                "afsa_detection_id": meta.get("afsa_detection_id"),
            }
        )
    out = {"routes": routes}
    Path("/tmp/afsa_routes.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] routes={len(routes)}")


if __name__ == "__main__":
    main()
