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
    echo "脚本未运行"
    exit 1
fi

# 检查进程是否还在运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "脚本已完成"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== 最终结果 ==="
        tail -100 "$LOG_FILE"
    fi
    exit 0
fi

# 清屏并显示进度
clear
echo "=========================================="
echo "全面标注脚本进度监控"
echo "=========================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo ""

# 显示进程信息
echo "=== 进程状态 ==="
ps -p $PID -o pid,pcpu,pmem,etime,state,cmd --no-headers 2>/dev/null | awk '{printf "  PID: %s | CPU: %s%% | 内存: %s%% | 运行时间: %s | 状态: %s\n", $1, $2, $3, $4, $5}'

echo ""
echo "=== 当前进度 ==="

if [ -f "$LOG_FILE" ]; then
    # 提取当前处理的split和类别
    CURRENT_SPLIT=$(grep "处理.*split" "$LOG_FILE" | tail -1 | sed 's/处理 //' | sed 's/ split://')
    CURRENT_CATEGORY=$(grep "  [a-z]" "$LOG_FILE" | tail -1 | sed 's/^[[:space:]]*//')
    
    if [ -n "$CURRENT_SPLIT" ]; then
        echo "  当前Split: $CURRENT_SPLIT"
    fi
    
    if [ -n "$CURRENT_CATEGORY" ]; then
        echo "  当前类别: $CURRENT_CATEGORY"
    fi
    
    # 统计已处理的类别
    TRAIN_COUNT=$(grep -c "train:" "$LOG_FILE" 2>/dev/null || echo "0")
    VAL_COUNT=$(grep -c "val:" "$LOG_FILE" 2>/dev/null || echo "0")
    TEST_COUNT=$(grep -c "test:" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo ""
    echo "=== 处理统计 ==="
    echo "  Train: $TRAIN_COUNT 个类别"
    echo "  Val: $VAL_COUNT 个类别"
    echo "  Test: $TEST_COUNT 个类别"
    
    # 显示最新日志
    echo ""
    echo "=== 最新日志 (最后20行) ==="
    tail -20 "$LOG_FILE" | sed 's/^/  /'
    
    # 检查是否完成
    if grep -q "标注统计" "$LOG_FILE"; then
        echo ""
        echo "=== ✅ 标注完成 ==="
        grep -A 100 "标注统计" "$LOG_FILE" | head -50
    fi
else
    echo "  等待日志文件创建..."
fi

echo ""
echo "=========================================="
echo "按 Ctrl+C 退出监控（不会停止脚本）"
echo "=========================================="
