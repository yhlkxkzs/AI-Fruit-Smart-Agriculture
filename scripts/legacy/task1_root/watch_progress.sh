#!/bin/bash
# 实时监控脚本进度

LOG_FILE="/tmp/fix_all_images_progress.log"
PID_FILE="/tmp/fix_all_images_pid.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "未找到PID文件，尝试从进程列表查找..."
    PID=$(ps aux | grep "fix_all_images_and_regenerate.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$PID" ]; then
        echo "脚本未运行"
        exit 1
    fi
else
    PID=$(cat "$PID_FILE")
fi

echo "=========================================="
echo "修复脚本进度监控"
echo "=========================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""

# 检查进程是否还在运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "脚本已完成"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== 最终结果 ==="
        tail -50 "$LOG_FILE"
    fi
    exit 0
fi

echo "脚本正在运行中..."
echo ""
echo "=== 实时输出 (按 Ctrl+C 退出，不会停止脚本) ==="
echo ""

# 使用tail -f实时显示日志
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
