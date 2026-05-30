#!/bin/bash
# 检查修复脚本的进度

LOG_FILE="/tmp/fix_all_images_progress.log"
PID_FILE="/tmp/fix_all_images_pid.txt"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "脚本正在运行 (PID: $PID)"
    else
        echo "脚本已完成"
    fi
else
    echo "未找到PID文件"
fi

if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "=== 最新进度 (最后30行) ==="
    tail -30 "$LOG_FILE"
    echo ""
    echo "=== 统计信息 ==="
    echo "总行数: $(wc -l < "$LOG_FILE")"
    if grep -q "总计重命名" "$LOG_FILE"; then
        echo "状态: 修复完成，正在重新生成标注文件"
        grep "总计重命名\|总计删除" "$LOG_FILE" | tail -2
    elif grep -q "重新生成所有标注文件" "$LOG_FILE"; then
        echo "状态: 正在重新生成标注文件"
    else
        echo "状态: 正在修复图片命名"
    fi
else
    echo "日志文件不存在，脚本可能还未开始"
fi
