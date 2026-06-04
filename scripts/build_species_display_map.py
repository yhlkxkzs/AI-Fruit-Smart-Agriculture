#!/usr/bin/env python3
"""Build species_display_map.json (canonical + en/zh) for GitHub inference & App."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECIES_MAP = Path("/home/yuhanlin/Database/datasets/temp/species_mapping.json")
LABELMAP = Path(
    "/home/yuhanlin/APP/AFSA/data/fruit_classification_multistate/labelmap_species.json"
)
SPECIES_ZH_FILE = ROOT / "data" / "species_zh_full.json"
OUT = ROOT / ".github/scripts/species_display_map.json"

STATE_ZH: dict[str, str] = {
    "healthy": "健康",
    "immature": "未成熟",
    "mature": "成熟",
    "aged": "过熟/衰老",
    "diseased": "病害",
    "pest": "虫害",
    "unknown": "未知",
}


def title_en(key: str) -> str:
    return key.replace("_", " ").strip().title()


def load_species_zh() -> dict[str, str]:
    if SPECIES_ZH_FILE.is_file():
        return json.loads(SPECIES_ZH_FILE.read_text(encoding="utf-8"))
    # 回退：运行生成脚本
    gen = ROOT / "scripts" / "generate_species_zh_full.py"
    if gen.is_file():
        import subprocess
        subprocess.run(["python3", str(gen)], check=True)
        return json.loads(SPECIES_ZH_FILE.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    species_list: list[str] = []
    aliases: dict[str, str] = {}
    if SPECIES_MAP.is_file():
        data = json.loads(SPECIES_MAP.read_text(encoding="utf-8"))
        species_list = data.get("species") or []
        aliases = data.get("species_aliases") or {}
    if LABELMAP.is_file():
        lm = json.loads(LABELMAP.read_text(encoding="utf-8"))
        for row in lm:
            n = row.get("object_name")
            if n and n not in species_list:
                species_list.append(n)

    species_zh = load_species_zh()
    species_out: dict[str, dict[str, str]] = {}
    for key in species_list:
        en = title_en(key)
        zh = species_zh.get(key, en)
        species_out[key] = {"en": en, "zh": zh}

    states_out = {
        k: {"en": title_en(k), "zh": STATE_ZH.get(k, title_en(k))}
        for k in STATE_ZH
    }

    payload = {
        "version": 2,
        "description": "Map model species_classes / aliases to display names (en + zh); 275 species",
        "species_aliases": aliases,
        "species": species_out,
        "states": states_out,
    }
    payload["species_aliases"]["Rotten"] = "other"
    payload["species_aliases"]["rotten"] = "other"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    zh_ok = sum(1 for v in species_out.values() if v["zh"] != v["en"])
    print(f"wrote {OUT} ({len(species_out)} species, {len(aliases)} aliases, {zh_ok} with zh!=en)")


if __name__ == "__main__":
    main()
