#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为刚合入 dataset_detection 的 Strawberry（Roboflow 熟度数据）补全成熟度标注。

我们在合并时按 combined_instances_{split}.json 的 images 顺序，依次写入到：
  train: strawberry_002706.. (共 172)
  val:   strawberry_000580.. (共 49)
  test:  strawberry_000581.. (共 26)

本脚本会读取源 COCO 的 categories/annotations，将每张图的 maturity（如 red/green/...）
写入目标 JSON 的 images[0].maturity 字段（保留原结构与 bbox 不变）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


SRC = Path("/home/yuhanlin/Database/local/database/Strawberry/annotations")
DST = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection")

RANGES = {
    "train": (2706, 172),
    "val": (580, 49),
    "test": (581, 26),
}


def maturity_list_for_split(split: str) -> list[str]:
    coco = json.loads((SRC / f"combined_instances_{split}.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco.get("categories") or []}
    img_ids_in_order = [im["id"] for im in coco.get("images") or []]

    ann_by_img = defaultdict(list)
    for a in coco.get("annotations") or []:
        ann_by_img[a["image_id"]].append(a["category_id"])

    out = []
    for iid in img_ids_in_order:
        cat_ids = ann_by_img.get(iid, [])
        if not cat_ids:
            out.append("unknown")
            continue
        # 一张图多框：取众数
        cid = Counter(cat_ids).most_common(1)[0][0]
        out.append(cats.get(cid, "unknown"))
    return out


def patch_one_json(json_path: Path, maturity: str) -> bool:
    if not json_path.is_file():
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data.get("images"):
        return False
    data["images"][0]["maturity"] = maturity
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main():
    total = 0
    for split, (start, n) in RANGES.items():
        maturities = maturity_list_for_split(split)
        if len(maturities) != n:
            raise RuntimeError(f"{split}: expected {n} maturities, got {len(maturities)}")
        ok = 0
        for i in range(n):
            idx = start + i
            jp = DST / split / "strawberry" / "json" / f"strawberry_{idx:06d}.json"
            if patch_one_json(jp, maturities[i]):
                ok += 1
        print(f"{split}: patched {ok}/{n}")
        total += ok
    print("Done. Patched total:", total)


if __name__ == "__main__":
    main()

