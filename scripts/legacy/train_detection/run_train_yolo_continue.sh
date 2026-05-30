#!/bin/bash
# YOLO 检测：单一数据集，首次训练或「数据集更新后从原权重继续训练」。
# 用法：
#   首次：RESUME=  DATA_YAML=configs/fruit_detection.yaml  MODEL=yolov8  ./run_train_yolo_continue.sh
#   继续：RESUME=runs/yolov8/weights/best.pt  DATA_YAML=...  MODEL=yolov8  ./run_train_yolo_continue.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/train_detection/runs"
CONFIGS_DIR="$PROJECT_ROOT/train_detection/configs"
cd "$PROJECT_ROOT"

# 模型：yolov8 或 yolo11
MODEL="${MODEL:-yolov8}"
# 唯一数据集 yaml（更新数据后重新生成该 yaml 指向的 dataset）
DATA_YAML="${DATA_YAML:-$CONFIGS_DIR/fruit_detection.yaml}"
# 已有权重（继续训练时填，如 runs/yolov8/weights/best.pt）
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"

if [[ ! -f "$DATA_YAML" ]]; then
  echo "错误: 未找到数据配置 $DATA_YAML"
  exit 1
fi

if [[ "$MODEL" == yolo11 ]]; then
  BASE_MODEL="yolo11n.pt"
  NAME="yolo11"
else
  BASE_MODEL="yolov8n.pt"
  NAME="yolov8"
fi

# 首次训练用预训练权重，继续训练用已有 best.pt
if [[ -n "$RESUME" && -f "$PROJECT_ROOT/train_detection/$RESUME" ]]; then
  WEIGHTS="$PROJECT_ROOT/train_detection/$RESUME"
  echo "继续训练: WEIGHTS=$WEIGHTS"
else
  WEIGHTS="$BASE_MODEL"
  echo "首次训练: model=$WEIGHTS"
fi

echo "DATA_YAML=$DATA_YAML  EPOCHS=$EPOCHS  name=$NAME"
yolo detect train \
  model="$WEIGHTS" \
  data="$DATA_YAML" \
  project="$RUNS_DIR" \
  name="$NAME" \
  epochs=$EPOCHS \
  batch=$BATCH \
  exist_ok=True \
  device=0
