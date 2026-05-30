#!/bin/bash
# EfficientNet-Lite0：水果分类训练，首次或继续训练。
# 依赖：pip install timm
# 使用方式：同 run_train_mobilenet_continue.sh，DATA / RESUME / EPOCHS 等。

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

DATA="${DATA:-$PROJECT_ROOT/dataset_cls}"
RESUME="${RESUME:-}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/train_detection/runs/efficientnet_lite0}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-32}"

if [[ ! -d "$DATA/train" ]]; then
  echo "错误: 未找到 $DATA/train，请先准备分类数据集（含 train/ 与 val/）"
  exit 1
fi

CMD=(python "$PROJECT_ROOT/train_detection/scripts/train_efficientnet_lite0.py"
  --data "$DATA"
  --save-dir "$SAVE_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --batch-size "$BATCH"
)

if [[ -n "$RESUME" && -f "$RESUME" ]]; then
  echo "继续训练: RESUME=$RESUME"
  CMD+=(--resume "$RESUME")
else
  echo "首次训练（无 --resume）"
fi

echo "EfficientNet-Lite0  DATA=$DATA SAVE_DIR=$SAVE_DIR EPOCHS=$EPOCHS"
"${CMD[@]}"
