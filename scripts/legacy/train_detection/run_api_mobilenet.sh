#!/bin/bash
# 在服务器上启动 MobileNet 分类 API，供手机 App 调用。
# 依赖：pip install flask pillow torch torchvision
# 用法：在项目根目录执行  ./train_detection/scripts/run_api_mobilenet.sh
#
# 监听地址：
#   - 默认 HOST=127.0.0.1 仅本机可访问，配合本机 WSL 的 SSH 隧道使用（无需开放防火墙）。
#   - 若已放行防火墙，可改为外网访问：HOST=0.0.0.0 bash run_api_mobilenet.sh

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
WEIGHTS="${WEIGHTS:-$PROJECT_ROOT/train_detection/runs/mobilenet_v3/weights/best.pt}"
PORT="${PORT:-32222}"
HOST="${HOST:-0.0.0.0}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "错误: 未找到权重 $WEIGHTS"
  exit 1
fi

echo "权重: $WEIGHTS  监听: $HOST:$PORT"
if [[ "$HOST" == "127.0.0.1" ]]; then
  echo "仅本机可访问，请在本机 WSL 用 SSH 隧道: ssh -N -L 32222:127.0.0.1:32222 -p <SSH端口> 用户@服务器"
fi
if [[ "$HOST" == "0.0.0.0" ]]; then
  echo "外网可访问；若已做端口转发 32222->本机:32222，手机 baseUrl 填: http://公网IP:32222"
fi
python "$PROJECT_ROOT/train_detection/scripts/api_mobilenet.py" \
  --weights "$WEIGHTS" \
  --port "$PORT" \
  --host "$HOST"
