#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从「YOLO 检测数据集」生成「按水果种类子文件夹」的分类数据集（datasetnew_cls 等），便于合并和训练。

YOLO 结构：<dataset>/images/train, labels/train, images/val, ...
每张图对应一个 .txt 标签，每行: class_id x_center y_center w h（归一化）。
按 class_id 把图片归入对应类别子文件夹。

用法：
  # 从 datasetnew 生成 datasetnew_cls（按水果种类子文件夹归纳）
  python build_cls_from_yolo.py --yolo-dir datasetnew --output datasetnew_cls --names "almond,apple,grape,lemon,mango,tomato"
  # 或使用同目录下的 run_build_datasetnew_cls.sh
"""
from __future__ import annotations

import argparse
import shutil
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def load_names_from_yaml(yaml_path: Path) -> list[str]:
    """从 YOLO data.yaml 读取 names（类别名列表，顺序即 class_id）。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    return []


def main():
    ap = argparse.ArgumentParser(description="从 YOLO 检测集生成 分类集（train/val/test 下按水果种类子文件夹）")
    ap.add_argument("--yolo-dir", type=str, required=True, help="YOLO 数据集根目录，含 images/train, labels/train 等")
    ap.add_argument("--output", type=str, required=True, help="输出的分类集根目录，如 datasetnew_cls")
    ap.add_argument("--names", type=str, default="", help="类别名，逗号分隔，顺序与 class_id 对应；不填则用 --yaml 读取")
    ap.add_argument("--yaml", type=str, default="", help="YOLO data.yaml 路径（含 names）；若提供则可省略 --names")
    ap.add_argument("--symlink", action="store_true", help="用符号链接代替复制")
    args = ap.parse_args()

    yolo_root = Path(args.yolo_dir)
    if not yolo_root.is_absolute():
        yolo_root = PROJECT_ROOT / yolo_root
    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    if args.names:
        names = [x.strip() for x in args.names.split(",") if x.strip()]
    elif args.yaml:
        yaml_path = Path(args.yaml)
        if not yaml_path.is_absolute():
            yaml_path = PROJECT_ROOT / yaml_path
        names = load_names_from_yaml(yaml_path)
    else:
        raise ValueError("请提供 --names 或 --yaml")
    if not names:
        raise ValueError("类别列表为空")

    img_dir = yolo_root / "images"
    lbl_dir = yolo_root / "labels"
    if not img_dir.exists():
        img_dir = yolo_root
        lbl_dir = yolo_root / "labels"
    if not lbl_dir.exists():
        raise FileNotFoundError(f"未找到标签目录: {lbl_dir}")

    for split in ["train", "val", "test"]:
        imgs = img_dir / split
        lbls = lbl_dir / split
        if not imgs.exists():
            continue
        for cls in names:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)
        count = 0
        for f in imgs.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXT:
                continue
            stem = f.stem
            txt = lbls / f"{stem}.txt"
            if not txt.exists():
                continue
            lines = [x.strip().split() for x in txt.read_text().strip().splitlines() if x.strip()]
            if not lines:
                continue
            ids = [int(l[0]) for l in lines if len(l) >= 5 and l[0].isdigit()]
            if not ids:
                continue
            cid = ids[0]
            if cid < 0 or cid >= len(names):
                continue
            cls = names[cid]
            out_path = out_root / split / cls / f"{stem}{f.suffix}"
            if not out_path.exists():
                if args.symlink:
                    out_path.symlink_to(f.resolve())
                else:
                    shutil.copy2(f, out_path)
            count += 1
        print(f"  {split}: {count} 张 -> {out_root / split}")

    print(f"已生成: {out_root}")


if __name__ == "__main__":
    main()
