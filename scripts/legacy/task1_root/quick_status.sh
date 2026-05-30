#!/bin/bash
# 快速查看脚本状态和进度

LOG_FILE="/tmp/comprehensive_annotation.log"
PID_FILE="/tmp/comprehensive_annotation_pid.txt"

echo "=========================================="
echo "全面标注脚本 - 快速状态"
echo "=========================================="

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

# 检查进程状态
if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ 脚本已完成"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== 最终统计 ==="
        grep -A 50 "标注统计" "$LOG_FILE" | head -60
    fi
    exit 0
fi

# 显示进程信息
echo "进程ID: $PID"
ps -p $PID -o etime,pcpu,pmem --no-headers 2>/dev/null | awk '{printf "运行时间: %s | CPU: %s%% | 内存: %s%%\n", $1, $2, $3}'

echo ""
echo "=== 当前进度 ==="

if [ -f "$LOG_FILE" ]; then
    # 提取最新的进度条
    LATEST_PROGRESS=$(grep -E "(train|val|test):.*%\|" "$LOG_FILE" | tail -1)
    if [ -n "$LATEST_PROGRESS" ]; then
        echo "$LATEST_PROGRESS"
    fi
    
    # 提取当前处理的类别
    CURRENT_CATEGORY=$(grep "  [a-z]" "$LOG_FILE" | tail -1 | sed 's/^[[:space:]]*//')
    if [ -n "$CURRENT_CATEGORY" ]; then
        echo "当前类别: $CURRENT_CATEGORY"
    fi
    
    echo ""
    echo "=== 最新日志 (最后5行) ==="
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    
    # 检查是否完成
    if grep -q "标注统计" "$LOG_FILE"; then
        echo ""
        echo "✅ 标注已完成！"
    fi
else
    echo "等待日志文件..."
fi

echo ""
echo "=========================================="
echo "实时查看: tail -f $LOG_FILE"
echo "或运行: bash scripts/realtime_monitor.sh"
echo "=========================================="
