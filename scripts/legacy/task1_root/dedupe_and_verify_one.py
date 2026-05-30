#!/usr/bin/env python3
"""Usage: python dedupe_and_verify_one.py <fruit_name>
   Example: python dedupe_and_verify_one.py lime
   Does: image dedup, JSON sync, verify width/height/bbox vs image, CSV, sets."""
import json
import re
import shutil
import csv
import cv2
import sys
from pathlib import Path
from collections import defaultdict

TRAIN = Path(__file__).resolve().parent.parent / "dataset" / "train"
EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")

def main():
    name = sys.argv[1].strip()
    if not name:
        print("Usage: python dedupe_and_verify_one.py <fruit_name>")
        return
    fig_dir = TRAIN / name
    if not fig_dir.is_dir():
        print(f"Not a directory: {fig_dir}")
        return
    json_dir = fig_dir / "json"
    csv_dir = fig_dir / "csv"
    sets_dir = fig_dir / "sets"
    dataset_sets = TRAIN.parent / "sets"
    prefix = re.escape(name.replace(" ", "_"))
    prefix2 = re.escape(name)
    def base_stem(s):
        for p in [prefix2 + r"_\d+", prefix + r"_\d+"]:
            m = re.match(r"(" + p + r")(?:_duplicate_[a-f0-9]+)*", s, re.I)
            if m: return m.group(1)
        return s
    def is_num(b):
        return bool(re.match(prefix2 + r"_\d+$", b) or re.match(prefix + r"_\d+$", b))

    # 1) Dedup images
    all_files = []
    for f in fig_dir.iterdir():
        if not f.is_file() or f.suffix not in EXTS:
            continue
        base = base_stem(f.stem)
        all_files.append((base, f.stem, f.name, f))
    groups = defaultdict(list)
    for b, s, n, p in all_files:
        groups[b].append((s, n, p))
    to_del = []
    for base, items in groups.items():
        can = (base + ".jpg") if is_num(base) else None
        def pri(it):
            st, nm, pt = it
            ext = pt.suffix.lower()
            dup = 1 if "_duplicate_" in nm else 0
            return (0 if ext == ".jpg" else 1 if ext == ".jpeg" else 2 if ext == ".png" else 3, dup, nm)
        items_sorted = sorted(items, key=pri)
        keep_s, keep_n, keep_p = items_sorted[0]
        if not is_num(base):
            can = keep_n
        for _, _, path in items_sorted[1:]:
            to_del.append(path)
        if can and keep_n != can:
            tgt = keep_p.parent / can
            if tgt != keep_p:
                if tgt.exists():
                    to_del.append(keep_p)
                else:
                    shutil.move(str(keep_p), str(tgt))
    for p in to_del:
        try:
            p.unlink()
        except Exception:
            pass
    print(f"  Deleted {len(to_del)} duplicate images.")

    on_disk = {f.stem: f.name for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    def get_canonical(fn):
        b = base_stem(Path(fn).stem)
        return b + ".jpg" if is_num(b) else fn

    if not json_dir.exists():
        print(f"  No json dir. Images: {len(on_disk)}")
        return
    # 2) Update JSON file_name
    for jp in json_dir.glob("*.json"):
        try:
            with open(jp) as f:
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
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    # 3) Remove invalid, one JSON per image
    for jp in list(json_dir.glob("*.json")):
        try:
            d = json.load(open(jp))
            stem = Path(d.get("images", [{}])[0].get("file_name", "")).stem
            if not stem or stem not in on_disk:
                jp.unlink()
        except Exception:
            jp.unlink()
    stem_to_j = defaultdict(list)
    for jp in json_dir.glob("*.json"):
        try:
            d = json.load(open(jp))
            stem = Path(d["images"][0]["file_name"]).stem
            if stem in on_disk:
                stem_to_j[stem].append(jp)
        except Exception:
            pass
    for stem, jlist in stem_to_j.items():
        if len(jlist) <= 1:
            continue
        tgt = json_dir / (stem + ".json")
        same = [j for j in jlist if j.stem == stem]
        keep = same[0] if same else jlist[0]
        with open(keep) as f:
            data = json.load(f)
        data["images"][0]["file_name"] = on_disk[stem]
        if "original_filename" in data["images"][0]:
            data["images"][0]["original_filename"] = on_disk[stem]
        with open(tgt, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        for jp in jlist:
            if jp != tgt:
                jp.unlink()
    for jp in list(json_dir.glob("*.json")):
        if jp.stem not in on_disk:
            jp.unlink()
    # 4) Minimal JSON for images with no JSON
    has_j = set(p.stem for p in json_dir.glob("*.json"))
    no_j = [s for s in on_disk if s not in has_j]
    if no_j:
        tpl = next(json_dir.glob("*.json"), None)
        if tpl:
            with open(tpl) as f:
                template = json.load(f)
            cid = template["categories"][0]["id"]
            for stem in no_j:
                fname = on_disk[stem]
                path = fig_dir / fname
                try:
                    im = cv2.imread(str(path))
                    h, w = (im.shape[0], im.shape[1]) if im is not None else (256, 256)
                except Exception:
                    h, w = 256, 256
                iid = 2000000000 + hash(stem) % 1000000000
                if iid < 0:
                    iid = -iid
                aid = 3000000000 + hash(stem) % 1000000000
                if aid < 0:
                    aid = -aid
                new_j = {
                    "info": template["info"],
                    "images": [{"id": iid, "width": w, "height": h, "file_name": fname, "size": w * h * 3, "format": "JPEG" if fname.lower().endswith((".jpg", ".jpeg")) else "PNG", "url": "", "hash": "", "status": "success", "original_filename": fname, "is_multi_fruit": False, "image_type": "single_fruit", "source_dataset": "unknown"}],
                    "annotations": [{"id": aid, "image_id": iid, "category_id": cid, "segmentation": [], "area": w * h, "bbox": [0, 0, w, h], "is_multi_fruit": False}],
                    "categories": template["categories"].copy(),
                }
                with open(json_dir / (stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(new_j, f, indent=2, ensure_ascii=False)
    # 5) Verify: width/height/bbox match image
    fixed_size = 0
    for jp in json_dir.glob("*.json"):
        try:
            with open(jp) as f:
                data = json.load(f)
        except Exception:
            continue
        fn = data.get("images", [{}])[0].get("file_name")
        path = fig_dir / fn
        if not path.exists():
            continue
        im = cv2.imread(str(path))
        if im is None:
            continue
        h, w = im.shape[0], im.shape[1]
        wj, hj = data["images"][0].get("width"), data["images"][0].get("height")
        if wj != w or hj != h:
            data["images"][0]["width"], data["images"][0]["height"] = w, h
            if data.get("annotations"):
                data["annotations"][0]["bbox"] = [0, 0, w, h]
                data["annotations"][0]["area"] = w * h
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            fixed_size += 1
    print(f"  Verified JSON vs image: fixed {fixed_size} size/bbox.")
    # 6) CSV
    if csv_dir.exists():
        for jp in json_dir.glob("*.json"):
            try:
                d = json.load(open(jp))
            except Exception:
                continue
            img, ann = d["images"][0], d.get("annotations", [{}])[0]
            cat = d.get("categories", [{}])[0]
            row = {"image_id": img["id"], "file_name": img["file_name"], "width": img["width"], "height": img["height"], "category_id": ann.get("category_id", cat.get("id")), "category_name": cat.get("name", name), "bbox_x": ann.get("bbox", [0, 0, 0, 0])[0], "bbox_y": ann.get("bbox", [0, 0, 0, 0])[1], "bbox_width": ann.get("bbox", [0, 0, 0, 0])[2], "bbox_height": ann.get("bbox", [0, 0, 0, 0])[3], "area": ann.get("area", 0), "is_multi_fruit": ann.get("is_multi_fruit", False)}
            with open(csv_dir / (jp.stem + ".csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
        has_j = set(p.stem for p in json_dir.glob("*.json"))
        for cp in csv_dir.glob("*.csv"):
            if cp.stem not in has_j:
                cp.unlink()
    # 7) sets
    stems = sorted(set(f.stem for f in fig_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")))
    sets_dir.mkdir(parents=True, exist_ok=True)
    for n in ["train.txt", "all.txt"]:
        with open(sets_dir / n, "w", encoding="utf-8") as f:
            f.write("\n".join(stems) + "\n")
    set_prefix = name + "/"
    for fn in ["train.txt", "all.txt"]:
        p = dataset_sets / fn
        if not p.exists():
            continue
        with open(p) as f:
            lines = [l.strip() for l in f if l.strip()]
        other = [l for l in lines if not l.startswith(set_prefix)]
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(other) + "\n")
            f.write("\n".join(set_prefix + s for s in stems) + "\n")
    print(f"  {name}: {len(on_disk)} images, {len(list(json_dir.glob('*.json')))} JSON, CSV/sets updated.")

if __name__ == "__main__":
    main()
