#!/bin/bash
# 实时监控全面标注脚本的进度

LOG_FILE="/tmp/comprehensive_annotation.log"
PID_FILE="/tmp/comprehensive_annotation_pid.txt"

# 获取PID
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
else
    PID=$(ps aux | grep "comprehensive_annotation.py" | grep -v grep | awk '{print $2}' | head -1)
fi

if [ -z "$PID" ]; then
    echo "❌ 脚本未运行"
    exit 1
fi

# 检查进程是否还在运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ 脚本已完成"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== 最终结果 ==="
        tail -100 "$LOG_FILE"
    fi
    exit 0
fi

echo "=========================================="
echo "实时监控全面标注脚本进度"
echo "=========================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "按 Ctrl+C 退出监控（不会停止脚本）"
echo "=========================================="
echo ""

# 实时跟踪日志
if [ -f "$LOG_FILE" ]; then
    tail -f "$LOG_FILE"
else
    echo "等待日志文件创建..."
    while [ ! -f "$LOG_FILE" ] && ps -p $PID > /dev/null 2>&1; do
        sleep 1
    done
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "无法找到日志文件"
    fi
fi
