#!/bin/bash
# 使用 dataset_detection（通过 dataset_detection_cls 视图）重新训练 mobilenet_v3、shufflenet_v2、efficientnet_lite0
# 先创建分类视图（若尚未创建）：python ../../scripts/make_dataset_detection_cls_view.py
set -e
cd "$(dirname "$0")/../.."
DATA="${1:-dataset_detection_cls}"
if [[ ! -d "$DATA" ]]; then
  echo "Creating $DATA symlink view..."
  python scripts/make_dataset_detection_cls_view.py
fi
DATA_ABS="$(cd "$DATA" && pwd)"
echo "Training with data: $DATA_ABS"
echo "=== 1/3 MobileNet V3 ==="
python train_detection/scripts/train_mobilenet_v3.py --data "$DATA_ABS" --epochs 50 --batch-size 32
echo "=== 2/3 ShuffleNet V2 ==="
python train_detection/scripts/train_shufflenet_v2.py --data "$DATA_ABS" --epochs 50 --batch-size 32
echo "=== 3/3 EfficientNet-Lite0 ==="
python train_detection/scripts/train_efficientnet_lite0.py --data "$DATA_ABS" --epochs 50 --batch-size 32
echo "All three trainings finished."
