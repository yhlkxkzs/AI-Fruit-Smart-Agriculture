#!/bin/bash

LOG_FILE="/tmp/verify_apple_model_full.log"
PID=$(ps aux | grep "[v]erify_apple_with_model.py" | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "脚本未运行"
    exit 1
fi

echo "=========================================="
echo "CLIP模型验证脚本 - 进度查看"
echo "=========================================="
echo "进程ID: $PID"
echo ""

# 显示进程信息
ps aux | grep "[v]erify_apple_with_model.py" | head -1

echo ""
echo "=== 最新进度 ==="
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "日志文件不存在"
fi

echo ""
echo "=== 修正统计 ==="
if [ -f "$LOG_FILE" ]; then
    grep "修正:" "$LOG_FILE" | wc -l | xargs -I {} echo "已修正文件数: {}"
    echo ""
    echo "最新修正："
    grep "修正:" "$LOG_FILE" | tail -10
fi

echo ""
echo "=========================================="
echo "实时查看: tail -f $LOG_FILE"
echo "=========================================="
