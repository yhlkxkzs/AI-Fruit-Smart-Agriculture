#!/usr/bin/env python3
"""对单个 fruit 目录做去重+同步 JSON/CSV/sets。用法: python dedupe_one_fruit.py <fruit_name>"""
import json
import re
import shutil
import csv
import cv2
import sys
from pathlib import Path
from collections import defaultdict

EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")

def run(fruit_name: str, train_root: Path):
    # fruit_name 可能含空格如 "summer squash"
    fig_dir = train_root / fruit_name
    if not fig_dir.is_dir():
        print(f"Skip {fruit_name}: not a directory")
        return
    json_dir = fig_dir / "json"
    csv_dir = fig_dir / "csv"
    sets_dir = fig_dir / "sets"
    dataset_sets = train_root.parent / "sets"
    prefix_esc = re.escape(fruit_name)  # "summer squash" -> 字面匹配
    prefix_num = re.escape(fruit_name.replace(" ", "_"))  # 文件名可能是 summer_squash_000000
    def base_stem(name: str) -> str:
        m = re.match(r"(" + prefix_esc + r"_\d+)(?:_duplicate_[a-f0-9]+)*", name, re.I)
        if m: return m.group(1)
        m = re.match(r"(" + prefix_num + r"_\d+)(?:_duplicate_[a-f0-9]+)*", name, re.I)
        return m.group(1) if m else name

    def is_numeric_base(base: str) -> bool:
        return bool(re.match(re.escape(fruit_name) + r"_\d+$", base) or re.match(prefix_num + r"_\d+$", base))

    def canonical_name(base: str) -> str:
        return base + ".jpg" if is_numeric_base(base) else None

    # 1) 图片去重
    all_files = []
    for f in fig_dir.iterdir():
        if not f.is_file() or f.suffix not in EXTS:
            continue
        base = base_stem(f.stem)
        all_files.append((base, f.stem, f.name, f))
    groups = defaultdict(list)
    for base, stem, name, path in all_files:
        groups[base].append((stem, name, path))

    to_delete = []
    for base, items in groups.items():
        is_num = is_numeric_base(base)
        can = base + ".jpg" if is_num else None
        def pri(item):
            stem, name, path = item
            ext = path.suffix.lower()
            dup = 1 if "_duplicate_" in name else 0
            if ext == ".jpg": return (0, dup, name)
            if ext == ".jpeg": return (1, dup, name)
            if ext == ".png": return (2, dup, name)
            return (3, dup, name)
        items_sorted = sorted(items, key=pri)
        keep_stem, keep_name, keep_path = items_sorted[0]
        if not is_num:
            can = keep_name
        for _, _, path in items_sorted[1:]:
            to_delete.append(path)
        if can and keep_name != can:
            target = keep_path.parent / can
            if target != keep_path:
                if target.exists():
                    to_delete.append(keep_path)
                else:
                    shutil.move(str(keep_path), str(target))
    for p in to_delete:
        try: p.unlink()
        except: pass
    n_del = len(to_delete)

    # 2) 规范 file_name
    on_disk = {f.stem: f.name for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    def get_canonical(fn: str) -> str:
        stem = Path(fn).stem
        base = base_stem(stem)
        return (base + ".jpg") if is_numeric_base(base) else fn

    if json_dir.exists():
        for jp in json_dir.glob("*.json"):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except: continue
            if not data.get("images"): continue
            fn = data["images"][0]["file_name"]
            new_fn = get_canonical(fn)
            if new_fn != fn:
                data["images"][0]["file_name"] = new_fn
                if "original_filename" in data["images"][0]:
                    data["images"][0]["original_filename"] = new_fn
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
    # 3) 删除无效 JSON，每图保留一个
    if json_dir.exists():
        to_del = []
        for jp in json_dir.glob("*.json"):
            try:
                d = json.load(open(jp))
                fn = d.get("images", [{}])[0].get("file_name")
                stem = Path(fn).stem if fn else None
                if not stem or stem not in on_disk:
                    to_del.append(jp)
            except: to_del.append(jp)
        for jp in to_del:
            jp.unlink()
        stem_to_jsons = defaultdict(list)
        for jp in json_dir.glob("*.json"):
            try:
                d = json.load(open(jp))
                fn = d.get("images", [{}])[0].get("file_name")
                stem = Path(fn).stem
                if stem in on_disk:
                    stem_to_jsons[stem].append(jp)
            except: pass
        for stem, jlist in stem_to_jsons.items():
            if len(jlist) <= 1: continue
            target = json_dir / (stem + ".json")
            same = [j for j in jlist if j.stem == stem]
            keep = same[0] if same else jlist[0]
            with open(keep) as f:
                data = json.load(f)
            data["images"][0]["file_name"] = on_disk[stem]
            if "original_filename" in data["images"][0]:
                data["images"][0]["original_filename"] = on_disk[stem]
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            for jp in jlist:
                if jp != target:
                    jp.unlink()
        for jp in list(json_dir.glob("*.json")):
            if jp.stem not in on_disk:
                jp.unlink()
    # 4) 为无 JSON 的图片创建最小 JSON
    has_json = set(p.stem for p in json_dir.glob("*.json")) if json_dir.exists() else set()
    no_json = [s for s in on_disk if s not in has_json]
    if no_json and json_dir.exists():
        tpl_path = next(json_dir.glob("*.json"), None)
        if tpl_path:
            with open(tpl_path) as f:
                template = json.load(f)
            cat_id = template["categories"][0]["id"]
            for stem in no_json:
                fname = on_disk[stem]
                path = fig_dir / fname
                try:
                    img = cv2.imread(str(path))
                    h, w = (img.shape[0], img.shape[1]) if img is not None else (256, 256)
                except: h, w = 256, 256
                img_id = 2000000000 + hash(stem) % 1000000000
                if img_id < 0: img_id = -img_id
                ann_id = 3000000000 + hash(stem) % 1000000000
                if ann_id < 0: ann_id = -ann_id
                new_j = {
                    "info": template["info"],
                    "images": [{"id": img_id, "width": w, "height": h, "file_name": fname,
                        "size": w * h * 3, "format": "JPEG" if fname.lower().endswith((".jpg", ".jpeg")) else "PNG",
                        "url": "", "hash": "", "status": "success", "original_filename": fname,
                        "is_multi_fruit": False, "image_type": "single_fruit", "source_dataset": "unknown"}],
                    "annotations": [{"id": ann_id, "image_id": img_id, "category_id": cat_id, "segmentation": [],
                        "area": w * h, "bbox": [0, 0, w, h], "is_multi_fruit": False}],
                    "categories": template["categories"].copy()
                }
                with open(json_dir / (stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(new_j, f, indent=2, ensure_ascii=False)
    # 5) 重新生成 CSV
    if csv_dir.exists() and json_dir.exists():
        for jp in json_dir.glob("*.json"):
            try:
                d = json.load(open(jp))
            except: continue
            img = d["images"][0]
            ann = d["annotations"][0] if d.get("annotations") else {}
            cat = d["categories"][0] if d.get("categories") else {}
            row = {"image_id": img["id"], "file_name": img["file_name"], "width": img["width"], "height": img["height"],
                "category_id": ann.get("category_id", cat.get("id")), "category_name": cat.get("name", fruit_name),
                "bbox_x": ann.get("bbox", [0,0,0,0])[0], "bbox_y": ann.get("bbox", [0,0,0,0])[1],
                "bbox_width": ann.get("bbox", [0,0,0,0])[2], "bbox_height": ann.get("bbox", [0,0,0,0])[3],
                "area": ann.get("area", 0), "is_multi_fruit": ann.get("is_multi_fruit", False)}
            with open(csv_dir / (jp.stem + ".csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
        has_json = set(p.stem for p in json_dir.glob("*.json"))
        for cp in csv_dir.glob("*.csv"):
            if cp.stem not in has_json:
                cp.unlink()
    # 6) sets
    stems = set()
    for f in fig_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stems.add(f.stem)
    sets_dir.mkdir(parents=True, exist_ok=True)
    with open(sets_dir / "train.txt", "w", encoding="utf-8") as f:
        for s in sorted(stems):
            f.write(s + "\n")
    with open(sets_dir / "all.txt", "w", encoding="utf-8") as f:
        for s in sorted(stems):
            f.write(s + "\n")
    # 7) dataset/sets
    train_txt = dataset_sets / "train.txt"
    all_txt = dataset_sets / "all.txt"
    set_prefix = fruit_name + "/"
    for txt_path, out_path in [(train_txt, train_txt), (all_txt, all_txt)]:
        if not txt_path.exists(): continue
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        other = [l for l in lines if not l.startswith(set_prefix)]
        fruit_lines = [set_prefix + s for s in sorted(stems)]
        with open(out_path, "w", encoding="utf-8") as f:
            for l in other:
                f.write(l + "\n")
            for l in fruit_lines:
                f.write(l + "\n")
    n_img = len(on_disk)
    n_json = len(list(json_dir.glob("*.json"))) if json_dir.exists() else 0
    print(f"  {fruit_name}: deleted {n_del} dup images, {n_img} images, {n_json} JSON")
    return n_img

if __name__ == "__main__":
    train_root = Path(__file__).resolve().parent.parent / "dataset" / "train"
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name:
        run(name, train_root)
    else:
        for fruit in ["kiwi", "lemon", "lime", "mango", "margo", "orange", "peach", "pear", "pepper", "pineapple",
                      "plum", "pomegranate", "potato", "raspberry", "soybean", "squash", "strawberry", "summar",
                      "summer squash", "tomato"]:
            run(fruit, train_root)
