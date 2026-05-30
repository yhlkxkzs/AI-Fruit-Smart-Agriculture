#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把多个「分类数据集」合并成一个，供后续统一训练。以后新加的数据集也加入列表再跑一次即可。

每个输入目录结构需为：
  <source>/
    train/<class>/图片...
    val/<class>/图片...
    test/<class>/图片...  （可选）

输出（默认 dataset_cls）：
  dataset_cls/
    train/<class>/...
    val/<class>/...
    test/<class>/...

用法：
  # 合并 datasetnew_cls 和 datasetlocal_cls 到 dataset_cls
  python merge_cls_datasets.py --sources datasetnew_cls datasetlocal_cls --output dataset_cls

  # 以后新加了 new_data_cls，再合并一次（会覆盖或追加到 dataset_cls）
  python merge_cls_datasets.py --sources datasetnew_cls datasetlocal_cls new_data_cls --output dataset_cls
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def collect_class_images(root: Path, split: str) -> dict[str, list[Path]]:
    """root/split/class/ -> {class: [path, ...]}。要求 split 下是「类别子文件夹」，不是直接放图片。列表按路径名排序以保证顺序稳定。"""
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


def check_source_has_class_subdirs(root: Path) -> tuple[bool, str]:
    """
    检查 root 是否是「分类结构」：train/val/test 下应有类别子文件夹（如 apple/、grape/），
    而不是直接在 train/ 下放图片。返回 (是否合格, 说明)。
    """
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        subdirs = [x for x in split_dir.iterdir() if x.is_dir() and not x.name.startswith(".")]
        files_direct = [x for x in split_dir.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXT]
        if files_direct and not subdirs:
            return False, f"{root.name}/{split}/ 下直接是图片文件，没有按类别分的子文件夹（需要 train/苹果名/图片 这种结构）"
        if subdirs:
            # 至少有一个 split 有类别子目录即认为结构正确
            return True, ""
    return False, f"{root.name} 下 train/val/test 中没有任何类别子文件夹"


def _find_json_for_image(class_dir: Path, stem: str, json_root: Path | None) -> Path | None:
    """在 class_dir 或 json_root 下查找与图片同 stem 的 .json。优先 class_dir 同目录、再 class_dir/json/、再 json_root/split/class/json/ 或 json_root/split/class/。"""
    for candidate in (
        class_dir / f"{stem}.json",
        class_dir / "json" / f"{stem}.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def merge_into(
    out_root: Path,
    sources: list[tuple[str, Path]],
    use_symlink: bool = False,
    copy_json: bool = False,
    json_sources: dict[str, Path] | None = None,
) -> None:
    """把多个 source 的 train/val/test 按类别合并到 out_root。若 copy_json 为 True，会为每张图复制同 stem 的 .json 到输出目录。"""
    json_sources = json_sources or {}
    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)
    all_classes = set()
    for name, src in sources:
        for split in ["train", "val", "test"]:
            for cls, paths in collect_class_images(src, split).items():
                all_classes.add(cls)
    for cls in sorted(all_classes):
        for split in ["train", "val", "test"]:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    for src_id, (src_name, src_root) in enumerate(sources):
        for split in ["train", "val", "test"]:
            for cls, paths in collect_class_images(src_root, split).items():
                out_dir = out_root / split / cls
                class_dir = src_root / split / cls
                json_root = json_sources.get(src_name)
                for i, p in enumerate(paths):
                    suf = p.suffix.lower()
                    if not suf:
                        suf = ".jpg"
                    new_name = f"{src_name}_{split}_{i}{suf}"
                    out_path = out_dir / new_name
                    if not out_path.exists():
                        if use_symlink:
                            out_path.symlink_to(p.resolve())
                        else:
                            shutil.copy2(p, out_path)

                    if copy_json:
                        stem = p.stem
                        json_path = _find_json_for_image(class_dir, stem, None)
                        if json_path is None and json_root is not None:
                            for sub in (json_root / split / cls / "json", json_root / split / cls):
                                candidate = sub / f"{stem}.json"
                                if candidate.is_file():
                                    json_path = candidate
                                    break
                        new_json_name = f"{src_name}_{split}_{i}.json"
                        out_json = out_dir / new_json_name
                        if json_path is not None and not out_json.exists():
                            if use_symlink:
                                out_json.symlink_to(json_path.resolve())
                            else:
                                shutil.copy2(json_path, out_json)
    return


def main():
    ap = argparse.ArgumentParser(description="合并多个分类数据集为一个")
    ap.add_argument("--sources", nargs="+", required=True, help="多个数据集根目录，如 datasetnew_cls datasetlocal_cls")
    ap.add_argument("--output", type=str, default="dataset_cls", help="合并后的输出目录，默认 dataset_cls")
    ap.add_argument("--symlink", action="store_true", help="用符号链接代替复制，省空间")
    ap.add_argument("--clear", action="store_true", help="合并前清空输出目录（否则在现有基础上追加）")
    ap.add_argument("--copy-json", action="store_true", help="同时复制每张图对应的同 stem 的 .json 到输出目录")
    ap.add_argument(
        "--json-sources",
        nargs="*",
        default=[],
        metavar="SOURCE:PATH",
        help="可选：源名:JSON根路径，如 datasetlocal_cls:/path/to/dataset。用于从该路径下 split/class/json/<stem>.json 或 split/class/<stem>.json 取 JSON",
    )
    args = ap.parse_args()

    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    if args.clear and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = []
    for s in args.sources:
        p = Path(s)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"跳过不存在的源: {p}")
            continue
        ok, msg = check_source_has_class_subdirs(p)
        if not ok:
            print(f"跳过（结构不符）: {p}")
            print(f"  原因: {msg}")
            continue
        name = p.name
        sources.append((name, p))
        print(f"  源: {p}")

    if not sources:
        print("没有有效源目录")
        return
    json_sources = {}
    for pair in getattr(args, "json_sources", []) or []:
        if ":" in pair:
            name, path = pair.split(":", 1)
            p = Path(path.strip())
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                json_sources[name.strip()] = p
                print(f"  JSON 源: {name} -> {p}")
    print(f"合并到: {out_root}  (symlink={args.symlink}, copy_json={getattr(args, 'copy_json', False)})")
    merge_into(
        out_root,
        sources,
        use_symlink=args.symlink,
        copy_json=getattr(args, "copy_json", False),
        json_sources=json_sources if json_sources else None,
    )

    for split in ["train", "val", "test"]:
        total = 0
        for c in (out_root / split).iterdir():
            if (out_root / split / c).is_dir():
                total += sum(1 for f in (out_root / split / c).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT)
        print(f"  {split}: {total} 张")


if __name__ == "__main__":
    main()
