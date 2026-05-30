#!/bin/bash
# MobileNetV3：单一数据集，首次训练或「数据集更新后继续训练」。
# 使用方式：
#   1) 首次训练：只设 DATA，不设 RESUME
#   2) 数据集更新后：设 DATA 为更新后的分类集，RESUME 为上次的 best.pt

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

# 唯一数据集路径（每次数据更新后重新生成该目录，再跑本脚本）
DATA="${DATA:-$PROJECT_ROOT/dataset_cls}"
# 已有权重（继续训练时必填；首次训练留空）
RESUME="${RESUME:-}"

SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/train_detection/runs/mobilenet_v3}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-32}"

if [[ ! -d "$DATA/train" ]]; then
  echo "错误: 未找到 $DATA/train，请先准备分类数据集（含 train/ 与 val/）"
  exit 1
fi

CMD=(python "$PROJECT_ROOT/train_detection/scripts/train_mobilenet_v3.py"
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

echo "DATA=$DATA SAVE_DIR=$SAVE_DIR EPOCHS=$EPOCHS"
"${CMD[@]}"
