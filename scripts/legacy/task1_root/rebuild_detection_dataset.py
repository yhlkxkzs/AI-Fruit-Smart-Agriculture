#!/usr/bin/env python3
"""
从 Database/github/database 与 Database/local/database 重新生成「识别水果种类」用 dataset，
仅采用「有果实」的数据源（叶/花/杂草/作物等已排除），见 docs/data_source_inventory.md。
保证每张图都有对应 JSON，目录结构与 apple_detection_drone_brazil 一致（images/ + json/，1:1 对应），
并做严谨性校验与报告。

用法:
  cd /home/yuhanlin/APP/AFSA/task1_fruit_classification
  python scripts/rebuild_detection_dataset.py [--output dataset_detection]   # 默认仅果实数据源
  python scripts/rebuild_detection_dataset.py --allow-all [--output ...]    # 使用全部 FRUIT_MAPPING 数据源
  python scripts/rebuild_detection_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_dataset import (
    FRUIT_MAPPING,
    get_image_files,
    is_image_file,
    validate_image,
    IMAGE_EXTENSIONS,
)
from PIL import Image

GITHUB_DB = Path("/home/yuhanlin/Database/github/database")
LOCAL_DB = Path("/home/yuhanlin/Database/local/database")
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset_detection"

# 仅「有果实」的数据集用于识别水果种类；叶/花/杂草/作物等不纳入。见 docs/data_source_inventory.md
FRUIT_ONLY_DATASETS = frozenset({
    # local
    "apple_detection_drone_brazil",
    "apple_segmentation_minnesota",
    "citrus_leaves",
    "embrapa_wgisd_grape_detection",
    "ghai_strawberry_fruit_detection",
    "mango_detection_australia",
    "riseholme_strawberry_classification_2021",
    "tomato_ripeness_detection",
    # github
    "acfr-multifruit-2016",
    "AppleBBCH81",
    "apple_minnesota",
    "CherryBBCH72",
    "CherryBBCH81",
    "deepfruits",
    "embrapa_add_256",
    "fruit-salad",
    "lemon-datase",
    "merged_fruit_detection",
    "Pear640",
    "Pistachio",
    "RawRipe",
    "tomato_plant",
})
SKIP_DATASETS = {"scripts"}


def _fruit_folder_name(fruit: str) -> str:
    """apple -> apples, grape -> grapes 等"""
    if fruit.endswith("s"):
        return fruit
    if fruit in ("orange", "apple", "grape", "peach", "pear", "cherry", "strawberry", "blueberry", "raspberry", "blackberry", "coconut", "avocado", "mango", "tomato", "squash", "lychee", "fig", "guava", "almond", "pistachio", "broccoli", "soybean", "wheat", "carrot", "cassava", "corn", "pepper", "potato", "papaya", "pomegranate", "apricot", "plum", "kiwi", "pineapple", "lemon", "lime", "banana", "watermelon", "cantaloupe", "rockmelon", "longan", "pomelo"):
        return fruit + "s" if not fruit.endswith("s") else fruit
    return fruit + "s"


def _discover_per_image_json_folders(dataset_path: Path, fruit_name: str) -> list[tuple[Path, Path]]:
    """
    若数据集存在 <fruit_plural>/images 与 <fruit_plural>/json，且按 stem 一一对应，返回 [(img_path, json_path), ...]。
    """
    pairs = []
    folder_name = _fruit_folder_name(fruit_name)
    images_dir = dataset_path / folder_name / "images"
    json_dir = dataset_path / folder_name / "json"
    if not images_dir.is_dir() or not json_dir.is_dir():
        return pairs
    for img_path in images_dir.iterdir():
        if not img_path.is_file() or not is_image_file(str(img_path)):
            continue
        stem = img_path.stem
        json_path = json_dir / f"{stem}.json"
        if json_path.is_file():
            pairs.append((img_path, json_path))
    return pairs


def _load_and_validate_single_json(json_path: Path, image_path: Path, fruit_name: str) -> dict | None:
    """加载单图 JSON，校验必要字段并修正 bbox/尺寸；若无效返回 None。"""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    images = data.get("images") or []
    if len(images) != 1:
        return None
    img_info = images[0]
    w_json = img_info.get("width")
    h_json = img_info.get("height")
    try:
        with Image.open(image_path) as im:
            w_real, h_real = im.size
    except Exception:
        return None
    # 以实际图片尺寸为准
    w, h = w_real, h_real
    if w_json is not None and h_json is not None and (int(w_json) != w or int(h_json) != h):
        img_info["width"] = w
        img_info["height"] = h
    img_info["file_name"] = image_path.name

    anns = data.get("annotations") or []
    categories = data.get("categories") or []
    cat_ids = {c["id"] for c in categories}
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = [float(bbox[i]) for i in range(4)]
        # 裁剪到图像内
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        bw = max(0, min(bw, w - x))
        bh = max(0, min(bh, h - y))
        if bw <= 0 or bh <= 0:
            continue
        ann["bbox"] = [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]
        ann["area"] = bw * bh
        if ann.get("category_id") not in cat_ids:
            ann["category_id"] = categories[0]["id"] if categories else 1
    data["annotations"] = [a for a in anns if len(a.get("bbox", [])) == 4 and a["bbox"][2] > 0 and a["bbox"][3] > 0]
    if not data["annotations"] and categories:
        # 无有效 bbox 时给整图一个默认框
        data["annotations"] = [{"id": 1, "image_id": img_info.get("id", 1), "category_id": categories[0]["id"], "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0}]
    return data


def _coco_to_per_image_jsons(dataset_path: Path, fruit_name: str) -> list[tuple[Path, dict]]:
    """
    若存在 annotations/combined_instances_*.json (COCO)，按 image_id 拆成单图 JSON，返回 [(image_path, json_dict), ...]。
    """
    ann_dir = dataset_path / "annotations"
    if not ann_dir.is_dir():
        return []
    coco_files = list(ann_dir.glob("combined_instances_*.json"))
    if not coco_files:
        return []
    all_images = []
    all_annotations = []
    all_categories = []
    id_to_img = {}
    for fp in coco_files:
        try:
            with open(fp, encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            continue
        for img in c.get("images") or []:
            id_to_img[img["id"]] = img
        all_annotations.extend(c.get("annotations") or [])
        if not all_categories and c.get("categories"):
            all_categories = c.get("categories")
    if not id_to_img:
        return []
    # 按 image_id 分组 annotations
    ann_by_img = defaultdict(list)
    for a in all_annotations:
        ann_by_img[a["image_id"]].append(a)
    result = []
    for img_id, img_info in id_to_img.items():
        fname = img_info.get("file_name")
        if not fname:
            continue
        img_path = dataset_path / fname
        if not img_path.is_file():
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        anns = ann_by_img.get(img_id, [])
        categories = all_categories or [{"id": 1, "name": fruit_name.upper(), "supercategory": "FRUIT"}]
        cat_ids = {c["id"] for c in categories}
        valid_anns = []
        for a in anns:
            bbox = a.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = [float(bbox[i]) for i in range(4)]
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            bw = max(0, min(bw, w - x))
            bh = max(0, min(bh, h - y))
            if bw <= 0 or bh <= 0:
                continue
            if a.get("category_id") not in cat_ids:
                a["category_id"] = categories[0]["id"]
            a["bbox"] = [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]
            a["area"] = bw * bh
            valid_anns.append(a)
        if not valid_anns:
            valid_anns = [{"id": 1, "image_id": img_id, "category_id": categories[0]["id"], "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0}]
        single = {
            "info": {"description": "From COCO", "version": "1.0", "source_dataset": dataset_path.name},
            "images": [{"id": img_id, "width": w, "height": h, "file_name": img_path.name, **{k: v for k, v in img_info.items() if k not in ("id", "width", "height", "file_name")}}],
            "annotations": valid_anns,
            "categories": categories,
        }
        result.append((img_path, single))
    return result


def _minimal_json_for_image(image_path: Path, fruit_name: str, dataset_name: str) -> dict:
    """无标注时生成整图 bbox 的单图 JSON。"""
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 0, 0
    return {
        "info": {"description": "No annotation; full-image bbox", "version": "1.0", "source_dataset": dataset_name},
        "images": [{"id": 1, "width": w, "height": h, "file_name": image_path.name, "source_dataset": dataset_name}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, w, h], "area": w * h, "iscrowd": 0}],
        "categories": [{"id": 1, "name": fruit_name.upper(), "supercategory": "FRUIT"}],
    }


def _discover_citrus_fruits_only(dataset_path: Path, dataset_name: str) -> list[tuple[Path, dict]]:
    """citrus_leaves 仅取 fruits/* 下的果实图，不取 leaves。"""
    out: list[tuple[Path, dict]] = []
    fruits_root = dataset_path / "fruits"
    if not fruits_root.is_dir():
        return out
    for subdir in fruits_root.iterdir():
        if not subdir.is_dir():
            continue
        images_dir = subdir / "images"
        json_dir = subdir / "json"
        if not images_dir.is_dir() or not json_dir.is_dir():
            continue
        for img_path in images_dir.iterdir():
            if not img_path.is_file() or not is_image_file(str(img_path)):
                continue
            json_path = json_dir / f"{img_path.stem}.json"
            if not json_path.is_file():
                continue
            if not validate_image(str(img_path)):
                continue
            data = _load_and_validate_single_json(json_path, img_path, "orange")
            if data is not None:
                data["images"][0]["source_dataset"] = dataset_name
                out.append((img_path, data))
    return out


def collect_pairs_from_dataset(dataset_path: Path, dataset_name: str, fruit_name: str) -> list[tuple[Path, dict]]:
    """
    从单个数据集中收集 (image_path, json_dict) 列表。
    json_dict 为单图 JSON（file_name 会在写入时被改为新文件名）。
    """
    out: list[tuple[Path, dict]] = []

    # 0) citrus_leaves 仅用 fruits/ 下的果实
    if dataset_name == "citrus_leaves":
        out = _discover_citrus_fruits_only(dataset_path, dataset_name)
        if out:
            return out

    # 1) 优先：已有 per-image json（如 apple_detection_drone_brazil/apples）
    pairs = _discover_per_image_json_folders(dataset_path, fruit_name)
    if pairs:
        for img_path, json_path in pairs:
            if not validate_image(str(img_path)):
                continue
            data = _load_and_validate_single_json(json_path, img_path, fruit_name)
            if data is not None:
                data["images"][0]["source_dataset"] = dataset_name
                out.append((img_path, data))
        if out:
            return out

    # 2) COCO 标注
    coco_pairs = _coco_to_per_image_jsons(dataset_path, fruit_name)
    if coco_pairs:
        for img_path, data in coco_pairs:
            if validate_image(str(img_path)):
                out.append((img_path, data))
        if out:
            return out

    # 3) 仅图片：递归收集，生成 minimal JSON
    skip_dirs = {"__pycache__", ".git", "scripts", "labels", "annotations", "data", "docs", "sets", "origin"}
    image_files = get_image_files(dataset_path)
    for img_path in image_files:
        path = Path(img_path)
        data = _minimal_json_for_image(path, fruit_name, dataset_name)
        out.append((path, data))
    return out


def main():
    ap = argparse.ArgumentParser(description="Rebuild detection dataset with 1:1 image-json and validation.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset root")
    ap.add_argument("--dry-run", action="store_true", help="Only collect and print stats, do not write")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--local-only", action="store_true", help="Only use LOCAL_DB (faster test)")
    ap.add_argument("--allow-all", action="store_true", help="Use all FRUIT_MAPPING datasets (default: only FRUIT_ONLY_DATASETS, 果实数据)")
    ap.add_argument("--max-per-fruit", type=int, default=0, help="Max pairs per fruit (0=no limit, for quick test use e.g. 500)")
    args = ap.parse_args()
    random.seed(args.seed)

    output_root = args.output.resolve()
    print("Output:", output_root)
    print("Sources: ", GITHUB_DB, LOCAL_DB)

    # 按水果汇总 (image_path, json_dict)，并记录来源
    by_fruit: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    dbs = [(LOCAL_DB, "local")]
    if not args.local_only:
        dbs.append((GITHUB_DB, "github"))
    for db_path, db_name in dbs:
        if not db_path.is_dir():
            print("Skip (not dir):", db_path)
            continue
        print(f"\nScanning {db_name}: {db_path}")
        for name in sorted(db_path.iterdir()):
            if not name.is_dir() or name.name in SKIP_DATASETS:
                continue
            fruit_name = None
            for key, value in FRUIT_MAPPING.items():
                if value and key.lower() in name.name.lower():
                    fruit_name = value
                    break
            if not fruit_name:
                continue
            if not args.allow_all and name.name not in FRUIT_ONLY_DATASETS:
                continue  # 只采用「有果实」的数据源
            print(f"  Processing {name.name} -> {fruit_name} ...", end=" ", flush=True)
            pairs = collect_pairs_from_dataset(name, name.name, fruit_name)
            print(len(pairs), "pairs", flush=True)
            if pairs:
                by_fruit[fruit_name].extend(pairs)

    # 去重：同一绝对路径只保留一次（以第一次来源为准）
    for fruit_name in list(by_fruit.keys()):
        seen = set()
        unique = []
        for img_path, j in by_fruit[fruit_name]:
            key = str(Path(img_path).resolve())
            if key not in seen:
                seen.add(key)
                unique.append((img_path, j))
        by_fruit[fruit_name] = unique

    # 可选：每类最多保留 N 对（便于快速试跑）
    max_per_fruit = getattr(args, "max_per_fruit", 0)
    if max_per_fruit > 0:
        for fruit_name in list(by_fruit.keys()):
            lst = by_fruit[fruit_name]
            if len(lst) > max_per_fruit:
                random.shuffle(lst)
                by_fruit[fruit_name] = lst[:max_per_fruit]
        print(f"\nCapped to max {max_per_fruit} per fruit.")

    # 划分 70 / 15 / 15
    splits = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_lists: dict[str, dict[str, list[tuple[Path, dict]]]] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for fruit_name, pairs in by_fruit.items():
        random.shuffle(pairs)
        n = len(pairs)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)
        split_lists["train"][fruit_name] = pairs[:t1]
        split_lists["val"][fruit_name] = pairs[t1:t2]
        split_lists["test"][fruit_name] = pairs[t2:]

    if args.dry_run:
        print("\n[DRY RUN] Would write:")
        for split in ("train", "val", "test"):
            for fruit_name, list_pairs in split_lists[split].items():
                print(f"  {output_root}/{split}/{fruit_name}/ images+json: {len(list_pairs)}")
        return

    # 重建前清空旧数据，避免残留叶/花/杂草等旧类别（如 blueberry、broccoli 等）
    for sub in ("train", "val", "test", "_report"):
        d = output_root / sub
        if d.exists():
            shutil.rmtree(d)
            print(f"Cleaned existing {d}")
    output_root.mkdir(parents=True, exist_ok=True)

    # 写入：每个 split/fruit 下 images/ 与 json/，命名 {fruit}_{idx:06d}.ext / .json
    report_dir = output_root / "_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    invalid_bbox = []
    for split in ("train", "val", "test"):
        for fruit_name, list_pairs in split_lists[split].items():
            img_dir = output_root / split / fruit_name / "images"
            json_dir = output_root / split / fruit_name / "json"
            img_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            for idx, (img_path, data) in enumerate(list_pairs):
                ext = Path(img_path).suffix
                stem = f"{fruit_name}_{idx:06d}"
                new_img_name = stem + ext
                new_json_name = stem + ".json"
                dst_img = img_dir / new_img_name
                dst_json = json_dir / new_json_name
                try:
                    shutil.copy2(img_path, dst_img)
                except Exception as e:
                    missing.append(f"{img_path} -> {dst_img}: {e}")
                    continue
                data["images"][0]["file_name"] = new_img_name
                try:
                    with Image.open(dst_img) as im:
                        data["images"][0]["width"], data["images"][0]["height"] = im.size
                except Exception:
                    pass
                for ann in data.get("annotations") or []:
                    bbox = ann.get("bbox", [])
                    w, h = data["images"][0]["width"], data["images"][0]["height"]
                    if len(bbox) == 4:
                        x, y, bw, bh = bbox
                        if x + bw > w or y + bh > h or x < 0 or y < 0:
                            invalid_bbox.append(f"{dst_json}: bbox {bbox} vs size {w}x{h}")
                with open(dst_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

    # 校验 1:1
    for split in ("train", "val", "test"):
        split_path = output_root / split
        if not split_path.is_dir():
            continue
        for fruit_dir in split_path.iterdir():
            if not fruit_dir.is_dir() or fruit_dir.name.startswith("_"):
                continue
            img_dir = fruit_dir / "images"
            json_dir = fruit_dir / "json"
            if not img_dir.is_dir() or not json_dir.is_dir():
                continue
            img_stems = {p.stem for p in img_dir.iterdir() if p.is_file() and is_image_file(str(p))}
            json_stems = {p.stem for p in json_dir.iterdir() if p.is_file() and p.suffix == ".json"}
            only_img = img_stems - json_stems
            only_json = json_stems - img_stems
            if only_img:
                missing.extend(f"only image: {fruit_dir}/images/{s}.*" for s in only_img)
            if only_json:
                missing.extend(f"only json: {fruit_dir}/json/{s}.json" for s in only_json)

    # 报告
    with open(report_dir / "missing_or_errors.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(missing) if missing else "None\n")
    with open(report_dir / "invalid_bbox.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(invalid_bbox) if invalid_bbox else "None\n")
    summary = {
        "by_fruit": {f: len(by_fruit[f]) for f in sorted(by_fruit)},
        "by_split": {},
        "missing_count": len(missing),
        "invalid_bbox_count": len(invalid_bbox),
    }
    for split in ("train", "val", "test"):
        summary["by_split"][split] = sum(len(split_lists[split][f]) for f in split_lists[split])
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nDone. Report in", report_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
