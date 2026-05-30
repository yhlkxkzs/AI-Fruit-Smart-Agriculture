#!/bin/bash
# 检查全面标注脚本的进度

LOG_FILE="/tmp/comprehensive_annotation.log"
PID_FILE="/tmp/comprehensive_annotation_pid.txt"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "脚本正在运行 (PID: $PID)"
    else
        echo "脚本已完成"
    fi
else
    PID=$(ps aux | grep "comprehensive_annotation.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$PID" ]; then
        echo "脚本未运行"
        exit 1
    fi
fi

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "=== 最新进度 (最后40行) ==="
    tail -40 "$LOG_FILE"
    echo ""
    echo "=== 统计信息 ==="
    echo "总行数: $(wc -l < "$LOG_FILE")"
    if grep -q "标注统计" "$LOG_FILE"; then
        echo "状态: ✅ 标注完成"
        echo ""
        grep -A 50 "标注统计" "$LOG_FILE"
    else
        echo "状态: 🔄 正在处理中..."
        LAST_CATEGORY=$(grep "处理.*split\|  [a-z]" "$LOG_FILE" | tail -1)
        echo "当前: $LAST_CATEGORY"
    fi
else
    echo "日志文件不存在，脚本可能还未开始"
fi
