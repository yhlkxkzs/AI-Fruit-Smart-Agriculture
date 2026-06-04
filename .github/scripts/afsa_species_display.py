#!/usr/bin/env python3
"""Resolve model class id → canonical species + bilingual display labels."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_PATH = SCRIPT_DIR / "species_display_map.json"


@lru_cache(maxsize=1)
def load_map() -> dict:
    if not MAP_PATH.is_file():
        return {"species_aliases": {}, "species": {}, "states": {}}
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def canonical_species(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "other"
    data = load_map()
    aliases = data.get("species_aliases") or {}
    return aliases.get(raw, aliases.get(raw.lower(), raw))


def species_display(raw: str) -> dict[str, str]:
    """Return canonical + en/zh for a model output class name."""
    key = canonical_species(raw)
    data = load_map()
    info = (data.get("species") or {}).get(key) or {}
    en = info.get("en") or key.replace("_", " ").title()
    zh = info.get("zh") or en
    return {
        "predicted_species_key": key,
        "predicted_class_en": en,
        "predicted_class_zh": zh,
    }


def state_display(raw: str) -> dict[str, str]:
    key = (raw or "unknown").strip()
    data = load_map()
    info = (data.get("states") or {}).get(key) or {}
    en = info.get("en") or key.replace("_", " ").title()
    zh = info.get("zh") or en
    return {
        "predicted_state_key": key,
        "predicted_state_en": en,
        "predicted_state_zh": zh,
    }
