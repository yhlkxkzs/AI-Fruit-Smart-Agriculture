#!/bin/bash
# 使用 ShuffleNet V2 权重启动水果分类 API（覆盖低端手机用户）。
# 依赖：pip install flask pillow torch torchvision
# 用法：在项目根目录执行  ./train_detection/scripts/run_api_shufflenet_v2.sh
# 可选：WEIGHTS=path/to/best.pt PORT=32224 bash run_api_shufflenet_v2.sh

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
WEIGHTS="${WEIGHTS:-$PROJECT_ROOT/train_detection/runs/shufflenet_v2/weights/best.pt}"
PORT="${PORT:-32224}"
HOST="${HOST:-0.0.0.0}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "错误: 未找到权重 $WEIGHTS"
  echo "请先训练: python train_detection/scripts/train_shufflenet_v2.py --data <dataset_cls> --epochs 50"
  exit 1
fi

echo "ShuffleNet V2 权重: $WEIGHTS  监听: $HOST:$PORT"
python "$PROJECT_ROOT/train_detection/scripts/api_mobilenet.py" \
  --weights "$WEIGHTS" \
  --port "$PORT" \
  --host "$HOST"
