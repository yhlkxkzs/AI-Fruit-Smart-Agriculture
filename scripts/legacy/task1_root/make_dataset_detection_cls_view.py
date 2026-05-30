#!/usr/bin/env python3
"""创建 dataset_detection_cls：train/<class> -> dataset_detection/train/<class>/images，供 ImageFolder 使用。"""
from pathlib import Path

DATASET = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection")
OUT = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_detection_cls")

def main():
    for split in ("train", "val", "test"):
        src_split = DATASET / split
        dst_split = OUT / split
        if not src_split.is_dir():
            continue
        dst_split.mkdir(parents=True, exist_ok=True)
        for cls_dir in sorted(src_split.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
                continue
            images_dir = cls_dir / "images"
            if not images_dir.is_dir():
                continue
            dst_cls = dst_split / cls_dir.name
            if dst_cls.exists():
                dst_cls.unlink()  # 可能是上次的 symlink
            dst_cls.symlink_to(images_dir.resolve())
            print(dst_cls, "->", images_dir.resolve())
    print("Done. Use --data", OUT)

if __name__ == "__main__":
    main()
