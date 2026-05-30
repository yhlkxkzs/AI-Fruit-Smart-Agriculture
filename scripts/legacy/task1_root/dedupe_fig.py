#!/usr/bin/env python3
"""
整理 fig 目录：去除图片命名重复，统一为每个 stem 只保留一个文件（fig_XXXXX.jpg 或 hash 名），
并更新 json、csv、sets。
"""
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "dataset" / "train" / "fig"
JSON_DIR = FIG_DIR / "json"
CSV_DIR = FIG_DIR / "csv"
SETS_DIR = FIG_DIR / "sets"
DATASET_SETS = BASE / "dataset" / "sets"

EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def base_stem(name: str) -> str:
    """fig_000000 -> fig_000000; fig_000000_duplicate_xxx -> fig_000000; fig_10295_6e0d41 -> fig_10295_6e0d41"""
    m = re.match(r"(fig_\d+)(?:_duplicate_[a-f0-9]+)*", name, re.I)
    return m.group(1) if m else name


def main():
    # 1) 收集所有图片，按 base_stem 分组
    all_files = []
    for f in FIG_DIR.iterdir():
        if not f.is_file() or f.suffix not in EXTS:
            continue
        stem = f.stem
        base = base_stem(stem)
        all_files.append((base, stem, f.name, f))

    groups = defaultdict(list)
    for base, stem, name, path in all_files:
        groups[base].append((stem, name, path))

    # 2) 为每组决定保留的文件
    keep_path = {}
    to_delete = []

    for base, items in groups.items():
        is_numeric = re.match(r"fig_\d+$", base) is not None
        canonical_name = base + ".jpg" if is_numeric else None

        def priority(item):
            stem, name, path = item
            ext = path.suffix.lower()
            has_dup = "_duplicate_" in name
            if ext == ".jpg":
                return (0, 1 if has_dup else 0, name)
            if ext == ".jpeg":
                return (1, 1 if has_dup else 0, name)
            if ext == ".png":
                return (2, 1 if has_dup else 0, name)
            return (3, 1 if has_dup else 0, name)

        items_sorted = sorted(items, key=priority)
        keep_stem, keep_name, keep_path_obj = items_sorted[0]
        if not is_numeric:
            canonical_name = keep_name
        keep_path[keep_path_obj] = canonical_name
        for stem, name, path in items_sorted[1:]:
            to_delete.append(path)
        if canonical_name and keep_name != canonical_name:
            target = keep_path_obj.parent / canonical_name
            if target != keep_path_obj:
                if target.exists():
                    to_delete.append(keep_path_obj)
                else:
                    shutil.move(str(keep_path_obj), str(target))
                    keep_path[target] = canonical_name

    for p in to_delete:
        try:
            p.unlink()
        except Exception as e:
            print("Delete failed:", p, e)

    def get_canonical(fn: str) -> str:
        stem = Path(fn).stem
        base = base_stem(stem)
        if re.match(r"fig_\d+$", base):
            return base + ".jpg"
        return fn

    # 3) 更新 JSON
    for jpath in JSON_DIR.glob("*.json"):
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not data.get("images"):
            continue
        fn = data["images"][0]["file_name"]
        new_fn = get_canonical(fn)
        if new_fn != fn:
            data["images"][0]["file_name"] = new_fn
            if "original_filename" in data["images"][0]:
                data["images"][0]["original_filename"] = new_fn
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # 4) 更新 CSV
    for cpath in CSV_DIR.glob("*.csv"):
        try:
            lines = open(cpath, encoding="utf-8").readlines()
        except Exception:
            continue
        if not lines:
            continue
        header = lines[0]
        if "file_name" not in header:
            continue
        idx = header.strip().split(",").index("file_name")
        out = [lines[0]]
        for line in lines[1:]:
            parts = line.split(",")
            if idx < len(parts):
                fn = parts[idx].strip()
                new_fn = get_canonical(fn)
                parts[idx] = new_fn
            out.append(",".join(parts))
        with open(cpath, "w", encoding="utf-8") as f:
            f.writelines(out)

    # 5) 重新生成 train/fig/sets/train.txt
    stems = set()
    for f in FIG_DIR.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stems.add(f.stem)
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETS_DIR / "train.txt", "w", encoding="utf-8") as f:
        for s in sorted(stems):
            f.write(s + "\n")

    # 6) 更新 dataset/sets/train.txt 中的 fig 行
    train_txt = DATASET_SETS / "train.txt"
    if train_txt.exists():
        with open(train_txt, "r", encoding="utf-8") as f:
            all_lines = [l.strip() for l in f if l.strip()]
        fig_prefix = "fig/"
        other = [l for l in all_lines if not l.startswith(fig_prefix)]
        fig_lines = ["fig/" + s for s in sorted(stems)]
        with open(train_txt, "w", encoding="utf-8") as f:
            for l in other:
                f.write(l + "\n")
            for l in fig_lines:
                f.write(l + "\n")

    print("Done. Kept", len(keep_path), "images; deleted", len(to_delete), "duplicates.")
    print("Sets train.txt:", len(stems), "entries.")


if __name__ == "__main__":
    main()
