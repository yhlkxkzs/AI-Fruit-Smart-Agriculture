#!/usr/bin/env python3
"""
一次性：将已合并到 train 的 data_apple 那 92 张（apple_004368..apple_004459）
按 70/15/15 重新分配到 train/val/test。
"""
import json
import shutil
from pathlib import Path

DATASET_ROOT = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection")
# 当前 92 张在 train 的编号范围
START, END = 4368, 4459
N = END - START + 1  # 92
N_TRAIN = int(round(N * 0.70))   # 64
N_VAL = int(round(N * 0.15))     # 14
N_TEST = N - N_TRAIN - N_VAL     # 14

def main():
    train_img = DATASET_ROOT / "train" / "apple" / "images"
    train_json = DATASET_ROOT / "train" / "apple" / "json"
    val_img = DATASET_ROOT / "val" / "apple" / "images"
    val_json = DATASET_ROOT / "val" / "apple" / "json"
    test_img = DATASET_ROOT / "test" / "apple" / "images"
    test_json = DATASET_ROOT / "test" / "apple" / "json"

    val_next = 936   # val 当前最大 935
    test_next = 936  # test 当前最大 935

    # 要移到 val 的编号：4432..4445 (14 张)
    val_src_indices = list(range(START + N_TRAIN, START + N_TRAIN + N_VAL))
    # 要移到 test 的编号：4446..4459 (14 张)
    test_src_indices = list(range(START + N_TRAIN + N_VAL, END + 1))

    def move_to_split(src_indices, dst_img_dir, dst_json_dir, dst_start_idx):
        cur = dst_start_idx
        for i in src_indices:
            # 在 train 里找 apple_XXXXXX
            candidates = list(train_img.glob(f"apple_{i:06d}.*"))
            if not candidates:
                continue
            src_img = candidates[0]
            ext = src_img.suffix
            src_js = train_json / f"apple_{i:06d}.json"
            if not src_js.is_file():
                continue
            new_stem = f"apple_{cur:06d}"
            new_img_name = new_stem + ext
            dst_img = dst_img_dir / new_img_name
            dst_js = dst_json_dir / f"{new_stem}.json"
            shutil.copy2(src_img, dst_img)
            with open(src_js, encoding="utf-8") as f:
                data = json.load(f)
            data["images"][0]["file_name"] = new_img_name
            with open(dst_js, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            src_img.unlink()
            src_js.unlink()
            cur += 1
        return cur - dst_start_idx

    moved_val = move_to_split(val_src_indices, val_img, val_json, val_next)
    moved_test = move_to_split(test_src_indices, test_img, test_json, test_next)
    print(f"Moved {moved_val} to val (apple_{val_next:06d}..), {moved_test} to test (apple_{test_next:06d}..)")
    print("Removed those from train. Train keeps apple_004368..apple_004431 (64).")


if __name__ == "__main__":
    main()
