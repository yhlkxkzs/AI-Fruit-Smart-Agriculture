#!/usr/bin/env python3
"""Append prediction rows to out/predictions.json."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("out/predictions.json")


def ensure_parent() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.is_file():
        OUT.write_text(json.dumps({"predictions": []}, indent=2) + "\n", encoding="utf-8")


def append_row(row: dict) -> None:
    ensure_parent()
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data.setdefault("predictions", []).append(row)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
