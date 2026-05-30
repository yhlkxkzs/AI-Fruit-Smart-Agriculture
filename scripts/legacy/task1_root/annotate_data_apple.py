#!/usr/bin/env python3
"""
为 /home/yuhanlin/APP/data_/apple 下的图片生成与 dataset_detection/test/apple/json 同格式的 JSON 标注。
无真实框时使用整图 bbox [0,0,width,height]，便于后续替换为真实标注。
"""
import json
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
DATA_APPLE = Path("/home/yuhanlin/APP/data_/apple")
JSON_DIR = DATA_APPLE / "json"
CATEGORY_ID = 1
CATEGORY_NAME = "APPLE"
SUPERCATEGORY = "apple"


def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(DATA_APPLE.iterdir()):
        if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue
        size_bytes = path.stat().st_size
        stem = path.stem
        fname = path.name
        image_id = abs(hash(stem)) % (10**9) + 10**9
        ann_id = image_id + 10**9
        area = w * h
        payload = {
            "info": {
                "description": "data",
                "version": "1.0",
                "year": 2025,
                "contributor": "search engine",
                "source": "web",
                "license": {
                    "name": "Creative Commons Attribution 4.0 International",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                },
            },
            "images": [
                {
                    "id": image_id,
                    "width": w,
                    "height": h,
                    "file_name": fname,
                    "size": size_bytes,
                    "format": "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG",
                    "url": "",
                    "hash": "",
                    "status": "success",
                    "source_dataset": "data_apple",
                }
            ],
            "annotations": [
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "segmentation": [],
                    "area": area,
                    "bbox": [0, 0, w, h],
                }
            ],
            "categories": [
                {"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": SUPERCATEGORY}
            ],
        }
        out_path = JSON_DIR / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        count += 1
    print(f"Wrote {count} JSONs to {JSON_DIR}")


if __name__ == "__main__":
    main()
