#!/bin/bash
# ShuffleNet V2：水果分类训练，覆盖低端手机用户。首次或继续训练。
# 使用方式：同 run_train_mobilenet_continue.sh。可选 VARIANT=x0_5（更轻）或 x1_0（默认）。

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

DATA="${DATA:-$PROJECT_ROOT/dataset_cls}"
RESUME="${RESUME:-}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/train_detection/runs/shufflenet_v2}"
VARIANT="${VARIANT:-x1_0}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-32}"

if [[ ! -d "$DATA/train" ]]; then
  echo "错误: 未找到 $DATA/train，请先准备分类数据集（含 train/ 与 val/）"
  exit 1
fi

CMD=(python "$PROJECT_ROOT/train_detection/scripts/train_shufflenet_v2.py"
  --data "$DATA"
  --save-dir "$SAVE_DIR"
  --variant "$VARIANT"
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

echo "ShuffleNet V2 ($VARIANT)  DATA=$DATA SAVE_DIR=$SAVE_DIR EPOCHS=$EPOCHS"
"${CMD[@]}"
