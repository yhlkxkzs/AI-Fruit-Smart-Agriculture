#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为已有的 dataset_cls 中的每张图片找到对应的 JSON 并复制进来（与 merge_cls_datasets 的命名一致，且使用相同的排序顺序）。

约定：dataset_cls 中图片命名为 <src_name>_<split>_<i>.<ext>，其中 i 为源目录中「按路径名排序」后的下标。
本脚本会到源目录（或可选的 json 根目录）中按同一顺序找到同 stem 的 .json，复制为 <src_name>_<split>_<i>.json。

用法：
  # 从各源目录自身找 json（源目录下需有 class/json/<stem>.json 或 class/<stem>.json）
  python copy_json_to_dataset_cls.py --dataset-cls dataset_cls --sources datasetnew_cls datasetlocal_cls

  # 指定某源的 JSON 来自另一路径（例如 datasetlocal_cls 的 json 在 dataset 里）
  python copy_json_to_dataset_cls.py --dataset-cls dataset_cls --sources datasetnew_cls datasetlocal_cls \\
    --json-sources datasetlocal_cls:dataset
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


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


def find_json_for_stem(class_dir: Path, stem: str, json_root: Path | None, split: str, cls: str) -> Path | None:
    """在 class_dir 或 json_root 下查找 <stem>.json。"""
    for candidate in (
        class_dir / f"{stem}.json",
        class_dir / "json" / f"{stem}.json",
    ):
        if candidate.is_file():
            return candidate
    if json_root is not None:
        for sub in (json_root / split / cls / "json", json_root / split / cls):
            candidate = sub / f"{stem}.json"
            if candidate.is_file():
                return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="为 dataset_cls 中的图片补全对应 JSON")
    ap.add_argument("--dataset-cls", type=str, default="dataset_cls", help="当前合并后的分类数据集根目录")
    ap.add_argument("--sources", nargs="+", required=True, help="与合并时一致的源目录名，如 datasetnew_cls datasetlocal_cls")
    ap.add_argument(
        "--json-sources",
        nargs="*",
        default=[],
        metavar="SOURCE:PATH",
        help="源名:JSON根路径，如 datasetlocal_cls:dataset",
    )
    ap.add_argument("--symlink", action="store_true", help="使用符号链接代替复制")
    args = ap.parse_args()

    ds_cls = Path(args.dataset_cls)
    if not ds_cls.is_absolute():
        ds_cls = PROJECT_ROOT / ds_cls
    if not ds_cls.exists():
        print(f"目录不存在: {ds_cls}")
        return

    # 解析 --sources 为路径（与 merge 一致，在项目根下找）
    source_roots: dict[str, Path] = {}
    for name in args.sources:
        p = PROJECT_ROOT / name
        if p.exists():
            source_roots[name] = p
        else:
            print(f"跳过不存在的源: {p}")
    if not source_roots:
        print("没有有效源目录")
        return

    json_sources: dict[str, Path] = {}
    for pair in args.json_sources or []:
        if ":" in pair:
            name, path = pair.split(":", 1)
            p = Path(path.strip())
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                json_sources[name.strip()] = p
                print(f"  JSON 源: {name} -> {p}")

    # 图片文件名正则：<src_name>_<split>_<i>.<ext>
    pattern = re.compile(r"^(.+)_(train|val|test)_(\d+)\.[a-zA-Z]+$")
    copied = 0
    skipped = 0
    missing = 0

    # 按 (src_name, split, cls) 缓存源路径列表，避免重复遍历
    cache: dict[tuple[str, str, str], list[Path]] = {}
    def get_source_paths(src_name: str, split_name: str, cls: str) -> list[Path]:
        key = (src_name, split_name, cls)
        if key not in cache:
            src_root = source_roots.get(src_name)
            if src_root is None:
                cache[key] = []
            else:
                by_cls = collect_class_images_sorted(src_root, split_name)
                cache[key] = by_cls.get(cls, [])
        return cache[key]

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
                paths = get_source_paths(src_name, split_name, cls)
                if idx >= len(paths):
                    missing += 1
                    continue
                source_image = paths[idx]
                stem = source_image.stem
                src_root = source_roots.get(src_name)
                src_class_dir = src_root / split_name / cls if src_root else None
                if src_class_dir is None:
                    missing += 1
                    continue
                json_root = json_sources.get(src_name)
                json_path = find_json_for_stem(src_class_dir, stem, json_root, split_name, cls)
                if json_path is None:
                    missing += 1
                    continue
                if args.symlink:
                    out_json.symlink_to(json_path.resolve())
                else:
                    shutil.copy2(json_path, out_json)
                copied += 1

    print(f"已复制 JSON: {copied}, 已存在跳过: {skipped}, 未找到 JSON: {missing}")


if __name__ == "__main__":
    main()
