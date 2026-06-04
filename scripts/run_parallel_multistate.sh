#!/usr/bin/env bash
# 10 个骨干 × multistate 双头，每张 GPU 跑一个（需 10 卡）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
SCRIPT="tasks/fruit_classification/scripts/train_multistate.py"
CONFIG="tasks/fruit_classification/configs/multistate.yaml"
LOG_DIR="tasks/fruit_classification/runs/multistate/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-50}"
SKIP_DONE="${SKIP_DONE:-1}"

# model_id:timm_name:gpu:batch_size
JOBS=(
  "efficientnet_lite0:tf_efficientnet_lite0:0:64"
  "mobilenet_v3:mobilenetv3_large_100:1:64"
  "shufflenet_v2:torchvision_shufflenet_v2_x1_0:2:64"
  "resnet18:resnet18:3:64"
  "efficientnet_lite4:tf_efficientnet_lite4:4:48"
  "mobilevit_s:mobilevit_s:5:48"
  "efficientnetv2_s:tf_efficientnetv2_s:6:32"
  "convnext_tiny:convnext_tiny:7:32"
  "regnety_400mf:regnety_004:8:48"
  "vit_tiny:deit_tiny_patch16_224:9:48"
)

launch() {
  local model_id="$1" timm_name="$2" gpu="$3" bs="$4"
  local export_pt="tasks/fruit_classification/exports/${model_id}/best.pt"
  local export_json="tasks/fruit_classification/exports/${model_id}/classes.json"
  local log="$LOG_DIR/${model_id}.log"

  if [[ "$SKIP_DONE" == "1" ]]; then
    if [[ -f "$export_json" ]] && grep -q classification_multistate "$export_json" 2>/dev/null; then
      echo "[skip] $model_id (已有 multistate export)"
      return 0
    fi
  fi

  echo "[launch] $model_id on cuda:$gpu batch=$bs -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 "$SCRIPT" \
    --config "$CONFIG" \
    --model-id "$model_id" \
    --timm-name "$timm_name" \
    --epochs "$EPOCHS" \
    --batch-size "$bs" \
    --device cuda \
    > "$log" 2>&1 &
  echo $! >> "$LOG_DIR/pids.txt"
}

: > "$LOG_DIR/pids.txt"
for job in "${JOBS[@]}"; do
  IFS=: read -r mid timm gpu bs <<< "$job"
  launch "$mid" "$timm" "$gpu" "$bs"
done

echo ""
echo "已启动并行训练，日志目录: $LOG_DIR"
echo "查看: tail -f $LOG_DIR/<model_id>.log"
echo "PIDs: $(cat "$LOG_DIR/pids.txt" | tr '\n' ' ')"
