#!/usr/bin/env python3
"""
将 /home/yuhanlin/APP/data_/apple 的图片和 json 合并到 dataset_detection 的 apple，
按 70% train / 15% val / 15% test 比例分配，图片与 json 在各 split 内按顺序后移命名。
"""
import json
import shutil
import re
from pathlib import Path

DATA_APPLE = Path("/home/yuhanlin/APP/data_/apple")
DATASET_ROOT = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15


def get_max_apple_index(split: str) -> int:
    img_dir = DATASET_ROOT / split / "apple" / "images"
    if not img_dir.is_dir():
        return -1
    max_idx = -1
    for f in img_dir.iterdir():
        if not f.is_file():
            continue
        m = re.match(r"apple_(\d+)", f.stem, re.I)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def main():
    # 收集 data_/apple 中所有图片（与 json 成对）
    pairs = []
    for path in sorted(DATA_APPLE.iterdir()):
        if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
            continue
        jpath = DATA_APPLE / "json" / f"{path.stem}.json"
        if not jpath.is_file():
            continue
        pairs.append((path, jpath))

    if not pairs:
        print("No image+json pairs in data_/apple")
        return

    n = len(pairs)
    n_train = max(0, int(round(n * TRAIN_RATIO)))
    n_val = max(0, int(round(n * VAL_RATIO)))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    print(f"Split: train={n_train}, val={n_val}, test={n_test} (total {n})")

    train_next = get_max_apple_index("train") + 1
    val_next = get_max_apple_index("val") + 1
    test_next = get_max_apple_index("test") + 1

    idx = 0
    for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        img_dir = DATASET_ROOT / split / "apple" / "images"
        json_dir = DATASET_ROOT / split / "apple" / "json"
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        if split == "train":
            next_idx = train_next
        elif split == "val":
            next_idx = val_next
        else:
            next_idx = test_next

        for _ in range(count):
            if idx >= len(pairs):
                break
            img_path, jpath = pairs[idx]
            idx += 1
            ext = img_path.suffix
            new_stem = f"apple_{next_idx:06d}"
            new_img_name = new_stem + ext
            new_json_name = new_stem + ".json"
            dst_img = img_dir / new_img_name
            dst_json = json_dir / new_json_name

            shutil.copy2(img_path, dst_img)
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
            data["images"][0]["file_name"] = new_img_name
            if "source_dataset" in data["images"][0]:
                data["images"][0]["source_dataset"] = "data_apple"
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            next_idx += 1

        print(f"  {split}: added {count} images (indices ending at {next_idx - 1})")

    print("Done.")


if __name__ == "__main__":
    main()
