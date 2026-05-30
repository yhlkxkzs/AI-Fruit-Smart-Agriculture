#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 datasetlocal 和 datasetnew 的 COCO 标注（combined_instances_<split>.json）里，
按「图片」查找每张图对应的标注，并导出为 dataset_cls 下每张图同名的 .json。

逻辑：
  - dataset_cls 里图片名为 <src_name>_<split>_<i>.<ext>（如 datasetlocal_cls_train_12.jpg）
  - 通过 src_name 和 index 在 datasetlocal_cls / datasetnew_cls 里找到源图路径，得到文件名（如 train_00002177.jpg）
  - 在 datasetlocal/annotations 或 datasetnew/annotations 的 combined_instances_<split>.json 里，
    按 file_name 匹配该文件名（支持 basename 或含路径的 file_name），取出该 image 的 annotations，写出为
    dataset_cls/<split>/<class>/<src_name>_<split>_<i>.json

用法：
  python copy_json_from_coco_to_dataset_cls.py --dataset-cls dataset_cls \\
    --sources datasetnew_cls datasetlocal_cls \\
    --coco-roots datasetnew:datasetnew datasetlocal:datasetlocal
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# 源名 -> 实际 COCO 根目录（该目录下要有 annotations/combined_instances_<split>.json）
SOURCE_TO_COCO_ROOT = {
    "datasetnew_cls": "datasetnew",
    "datasetlocal_cls": "datasetlocal",
}


def collect_class_images_sorted(root: Path, split: str) -> dict[str, list[Path]]:
    """与 merge_cls_datasets 一致：按类别、按路径名排序。"""
    out = {}
    split_dir = root / split
    if not split_dir.exists():
        return out
    for class_dir in sorted(split_dir.iterdir(), key=lambda x: x.name):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT]
        if files:
            out[class_dir.name] = sorted(files, key=lambda p: p.name)
    return out


def load_coco_and_index(coco_json: Path):
    """加载 COCO JSON，返回 (images list, id->image, basename->image_id, annotations by image_id)。"""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    id_to_img = {im["id"]: im for im in images}
    # 用 basename 和 完整 file_name 做索引，便于用“文件名”查找
    basename_to_ids = defaultdict(list)
    for im in images:
        fn = im.get("file_name", "")
        basename = Path(fn).name
        basename_to_ids[basename].append(im["id"])
        # 也存一下去掉扩展名匹配（避免 .jpg vs .png）
        stem = Path(fn).stem
        if stem != basename:
            basename_to_ids[stem + ".jpg"].append(im["id"])
            basename_to_ids[stem + ".png"].append(im["id"])
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann["image_id"]].append(ann)
    return images, id_to_img, basename_to_ids, ann_by_img, data.get("categories", [])


def main():
    ap = argparse.ArgumentParser(description="从 datasetlocal/datasetnew 的 COCO 标注中为 dataset_cls 每张图导出对应 JSON")
    ap.add_argument("--dataset-cls", type=str, default="dataset_cls", help="合并后的分类数据集根目录")
    ap.add_argument("--sources", nargs="+", default=["datasetnew_cls", "datasetlocal_cls"], help="源目录名列表")
    ap.add_argument(
        "--coco-roots",
        nargs="*",
        metavar="SOURCE:PATH",
        default=[],
        help="源名:COCO根路径，如 datasetlocal_cls:datasetlocal datasetnew_cls:datasetnew；不写则用内置默认",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将要写入的 JSON 数量，不写文件")
    args = ap.parse_args()

    ds_cls = Path(args.dataset_cls)
    if not ds_cls.is_absolute():
        ds_cls = PROJECT_ROOT / ds_cls
    if not ds_cls.exists():
        print(f"目录不存在: {ds_cls}")
        return

    # 解析 coco-roots：源名 -> COCO 根目录（含 annotations/）
    coco_roots = dict(SOURCE_TO_COCO_ROOT)
    for pair in args.coco_roots or []:
        if ":" in pair:
            name, path = pair.split(":", 1)
            p = Path(path.strip())
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                coco_roots[name.strip()] = str(p)

    # 源名 -> 源根路径（用于取“第 i 张图”）
    source_roots = {}
    for name in args.sources:
        p = PROJECT_ROOT / name
        if p.exists():
            source_roots[name] = p
        else:
            print(f"跳过不存在的源: {p}")
    if not source_roots:
        print("没有有效源目录")
        return

    pattern = re.compile(r"^(.+)_(train|val|test)_(\d+)\.[a-zA-Z]+$")
    cache = {}

    def get_source_paths(src_name: str, split_name: str, cls: str) -> list[Path]:
        key = (src_name, split_name, cls)
        if key not in cache:
            root = source_roots.get(src_name)
            if root is None:
                cache[key] = []
            else:
                by_cls = collect_class_images_sorted(root, split_name)
                cache[key] = by_cls.get(cls, [])
        return cache[key]

    written = 0
    skipped = 0
    no_coco = 0
    no_match = 0
    # 按 split 加载 COCO，避免重复加载
    coco_cache = {}  # (coco_root, split) -> (id_to_img, basename_to_ids, ann_by_img, categories)

    for split in ["train", "val", "test"]:
        split_dir = ds_cls / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir(), key=lambda x: x.name):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            cls = class_dir.name
            for f in class_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXT:
                    continue
                m = pattern.match(f.name)
                if not m:
                    continue
                src_name, split_name, idx_str = m.group(1), m.group(2), m.group(3)
                idx = int(idx_str)
                out_json = class_dir / f"{src_name}_{split_name}_{idx}.json"
                if out_json.exists():
                    skipped += 1
                    continue
                coco_root_name = coco_roots.get(src_name)
                if not coco_root_name:
                    no_coco += 1
                    continue
                coco_root = PROJECT_ROOT / coco_root_name if not Path(coco_root_name).is_absolute() else Path(coco_root_name)
                coco_json = coco_root / "annotations" / f"combined_instances_{split_name}.json"
                if not coco_json.exists():
                    no_coco += 1
                    continue
                cache_key = (str(coco_root), split_name)
                if cache_key not in coco_cache:
                    try:
                        _, id_to_img, basename_to_ids, ann_by_img, categories = load_coco_and_index(coco_json)
                        coco_cache[cache_key] = (id_to_img, basename_to_ids, ann_by_img, categories)
                    except Exception as e:
                        print(f"加载 {coco_json} 失败: {e}")
                        no_coco += 1
                        continue
                id_to_img, basename_to_ids, ann_by_img, categories = coco_cache[cache_key]
                paths = get_source_paths(src_name, split_name, cls)
                if idx >= len(paths):
                    no_match += 1
                    continue
                source_image = paths[idx]
                basename = source_image.name
                stem = source_image.stem
                # 尝试用 basename 或 stem 匹配
                image_ids = basename_to_ids.get(basename) or basename_to_ids.get(stem + ".jpg") or basename_to_ids.get(stem + ".png")
                if not image_ids:
                    no_match += 1
                    continue
                image_id = image_ids[0]
                img_obj = id_to_img.get(image_id)
                anns = ann_by_img.get(image_id, [])
                if not img_obj:
                    no_match += 1
                    continue
                out_data = {
                    "image": img_obj,
                    "annotations": anns,
                    "categories": categories,
                }
                if not args.dry_run:
                    with open(out_json, "w", encoding="utf-8") as fp:
                        json.dump(out_data, fp, ensure_ascii=False, indent=2)
                written += 1

    print(f"从 COCO 写出 JSON: {written}, 已存在跳过: {skipped}, 未配置/无 COCO: {no_coco}, 未匹配到图片: {no_match}")
    if args.dry_run:
        print("(dry-run，未写入文件)")


if __name__ == "__main__":
    main()
