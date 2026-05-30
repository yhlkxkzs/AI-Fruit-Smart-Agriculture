#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将标准化后的 /home/yuhanlin/Database/local/database/Strawberry/strawberries 数据集
合并进 task1 的 dataset_detection 的 strawberry 类别中（按原 COCO split：train/val/test）。

规则：
- 只把图片当作「strawberry」果实类合入，不保留熟度子类。
- 写入到 dataset_detection/<split>/strawberry/images 与 json。
- 目标命名：strawberry_{index:06d}{ext}，index 在各 split 内按现有最大编号顺延。
- JSON：沿用 dataset_detection 的单图 JSON 结构，bbox 用整图框 [0,0,w,h]。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from PIL import Image


SRC_ROOT = Path("/home/yuhanlin/Database/local/database/Strawberry")
SRC_IMAGES_ROOT = SRC_ROOT / "strawberries"
SRC_ANN = SRC_ROOT / "annotations"

DST_ROOT = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection")


def _max_idx(split: str) -> int:
    img_dir = DST_ROOT / split / "strawberry" / "images"
    if not img_dir.is_dir():
        return -1
    mmax = -1
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        m = re.match(r"strawberry_(\d+)", p.stem, re.I)
        if m:
            mmax = max(mmax, int(m.group(1)))
    return mmax


def _build_image_map() -> dict[str, Path]:
    """file_name -> existing path in any subclass images/"""
    mp: dict[str, Path] = {}
    for cls_dir in SRC_IMAGES_ROOT.iterdir():
        images_dir = cls_dir / "color" / "images"
        if not images_dir.is_dir():
            continue
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                mp.setdefault(p.name, p)
    return mp


def _write_json(dst_json: Path, dst_img: Path, split: str):
    with Image.open(dst_img) as im:
        w, h = im.size
    size_bytes = dst_img.stat().st_size
    stem = dst_img.stem
    image_id = abs(hash(f"{split}:{stem}")) % (10**9) + 10**9
    ann_id = image_id + 10**9
    payload = {
        "info": {
            "description": "data",
            "version": "1.0",
            "year": 2025,
            "contributor": "roboflow",
            "source": "Strawberry_standardized",
            "license": {
                "name": "Creative Commons Attribution 4.0 International",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
        },
        "images": [
            {
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": dst_img.name,
                "size": size_bytes,
                "format": "JPEG" if dst_img.suffix.lower() in (".jpg", ".jpeg") else "PNG",
                "url": "",
                "hash": "",
                "status": "success",
                "source_dataset": "Strawberry",
            }
        ],
        "annotations": [
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [],
                "area": float(w * h),
                "bbox": [0, 0, float(w), float(h)],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "STRAWBERRY", "supercategory": "strawberry"}
        ],
    }
    dst_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    image_map = _build_image_map()
    if not image_map:
        raise RuntimeError(f"未在 {SRC_IMAGES_ROOT} 下找到图片")

    splits = {"train": "train", "val": "val", "test": "test"}
    next_idx = {s: _max_idx(s) + 1 for s in splits}
    print("Next indices:", next_idx)

    merged_counts = {s: 0 for s in splits}

    for split in ("train", "val", "test"):
        coco_path = SRC_ANN / f"combined_instances_{split}.json"
        if not coco_path.exists():
            print("Skip (no coco):", coco_path)
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        imgs = coco.get("images") or []
        if not imgs:
            continue

        dst_img_dir = DST_ROOT / split / "strawberry" / "images"
        dst_json_dir = DST_ROOT / split / "strawberry" / "json"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_json_dir.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            fname = img.get("file_name")
            if not fname:
                continue
            src_img = image_map.get(fname)
            if not src_img or not src_img.exists():
                continue

            idx = next_idx[split]
            next_idx[split] += 1
            new_name = f"strawberry_{idx:06d}{src_img.suffix.lower()}"
            dst_img = dst_img_dir / new_name
            dst_json = dst_json_dir / f"strawberry_{idx:06d}.json"

            shutil.copy2(src_img, dst_img)
            _write_json(dst_json, dst_img, split)
            merged_counts[split] += 1

    print("Merged counts:", merged_counts)


if __name__ == "__main__":
    main()

