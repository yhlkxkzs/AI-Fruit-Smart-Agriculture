#!/bin/bash
# 从 datasetnew（YOLO 检测集）生成 datasetnew_cls，按水果种类子文件夹归纳（train/val/test 下各有 apple/、grape/ 等）。
# 若你用的 datasetnew 类别顺序不同，请改 NAMES 或用 --yaml 指向 data.yaml。

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

# datasetnew 的 6 类（class_id 0~5 对应），与当时 YOLO 训练用的 data.yaml 中 names 顺序一致
# 若不一致请改为你的顺序，或使用 --yaml train_detection/configs/fruit_detection.yaml
NAMES="${NAMES:-almond,apple,grape,lemon,mango,tomato}"

echo "从 datasetnew 生成 datasetnew_cls（按水果种类子文件夹）"
echo "类别顺序: $NAMES"
python train_detection/scripts/build_cls_from_yolo.py \
  --yolo-dir datasetnew \
  --output datasetnew_cls \
  --names "$NAMES" \
  --symlink

echo "完成。datasetnew_cls 已包含 train/val/test 下按类别子文件夹。"
