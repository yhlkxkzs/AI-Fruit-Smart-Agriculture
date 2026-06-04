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
OUT = ROOT / ".github/scripts/species_display_map.json"

# 常见物种中文名（可逐步扩充）
SPECIES_ZH: dict[str, str] = {
    "apple": "苹果",
    "pear": "梨",
    "peach": "桃",
    "grape": "葡萄",
    "orange": "橙",
    "citrus": "柑橘",
    "lemon": "柠檬",
    "lime": "青柠",
    "banana": "香蕉",
    "strawberry": "草莓",
    "watermelon": "西瓜",
    "melon": "甜瓜",
    "mango": "芒果",
    "pineapple": "菠萝",
    "coconut": "椰子",
    "guava": "番石榴",
    "papaya": "木瓜",
    "pomegranate": "石榴",
    "litchi": "荔枝",
    "longan": "龙眼",
    "avocado": "牛油果",
    "cherry": "樱桃",
    "blueberry": "蓝莓",
    "blackberry": "黑莓",
    "raspberry": "树莓",
    "cranberry": "蔓越莓",
    "tomato": "番茄",
    "cucumber": "黄瓜",
    "eggplant": "茄子",
    "jackfruit": "菠萝蜜",
    "durian": "榴莲",
    "rambutan": "红毛丹",
    "mangosteen": "山竹",
    "yali_pear": "鸭梨",
    "starfruit": "杨桃",
    "dragonfruit": "火龙果",
    "pitaya": "火龙果",
    "passionfruit": "百香果",
    "kiwi": "猕猴桃",
    "fig": "无花果",
    "date": "椰枣",
    "plum": "李子",
    "apricot": "杏",
    "pistachio": "开心果",
    "chestnut": "栗子",
    "walnut": "核桃",
    "tamarind": "罗望子",
    "pomelo": "柚子",
    "grapefruit": "葡萄柚",
    "lychee": "荔枝",
    "loquat": "枇杷",
    "persimmon": "柿子",
    "hawthorn": "山楂",
    "jujube": "枣",
    "gooseberry": "醋栗",
    "currant": "醋栗",
    "black_currant": "黑醋栗",
    "red_currant": "红醋栗",
    "mulberry": "桑葚",
    "white_mulberry": "桑椹",
    "black_mulberry": "黑桑",
    "olive": "橄榄",
    "coffee": "咖啡",
    "cocoa_bean": "可可",
    "bean": "豆",
    "soybean": "大豆",
    "broccoli": "西兰花",
    "corn_kernel": "玉米",
    "zucchini": "西葫芦",
    "bitter_melon": "苦瓜",
    "wax_gourd": "冬瓜",
    "hogplum": "李榄",
    "custard_apple": "番荔枝",
    "soursop": "刺果番荔枝",
    "cherimoya": "番荔枝",
    "feijoa": "费约果",
    "acerola": "针叶樱桃",
    "camu_camu": "卡姆果",
    "african_plum": "非洲李",
    "other": "其他",
}


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

    species_out: dict[str, dict[str, str]] = {}
    for key in species_list:
        en = title_en(key)
        zh = SPECIES_ZH.get(key, en)
        species_out[key] = {"en": en, "zh": zh}

    states_out = {
        k: {"en": title_en(k), "zh": STATE_ZH.get(k, title_en(k))}
        for k in STATE_ZH
    }

    payload = {
        "version": 1,
        "description": "Map model species_classes / aliases to display names (en + zh)",
        "species_aliases": aliases,
        "species": species_out,
        "states": states_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(species_out)} species, {len(aliases)} aliases)")


if __name__ == "__main__":
    main()
