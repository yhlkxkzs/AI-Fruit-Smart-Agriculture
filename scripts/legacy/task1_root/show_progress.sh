#!/bin/bash
# 查看综合修正脚本的进度

LOG_FILE="/tmp/comprehensive_fix_annotations.log"

echo "=========================================="
echo "综合修正标注脚本 - 进度查看"
echo "=========================================="

# 检查进程是否运行
if ps aux | grep -q "[c]omprehensive_fix_annotations.py"; then
    PID=$(ps aux | grep "[c]omprehensive_fix_annotations.py" | awk '{print $2}')
    echo "✓ 进程运行中 (PID: $PID)"
    ps aux | grep "[c]omprehensive_fix_annotations.py" | grep -v grep | awk '{printf "  CPU: %s%% | 内存: %s%%\n", $3, $4}'
else
    echo "✗ 进程已结束"
fi

echo ""
echo "=== 最新进度 ==="
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "日志文件不存在"
fi

echo ""
echo "=========================================="
echo "实时查看: tail -f $LOG_FILE"
echo "=========================================="
