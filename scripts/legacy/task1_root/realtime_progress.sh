#!/bin/bash
# 实时显示综合修正脚本的进度

LOG_FILE="/tmp/comprehensive_fix_annotations.log"

while true; do
    clear
    echo "=========================================="
    echo "综合修正标注脚本 - 实时进度"
    echo "=========================================="
    date
    echo ""
    
    # 检查进程
    if ps aux | grep -q "[c]omprehensive_fix_annotations.py"; then
        PID=$(ps aux | grep "[c]omprehensive_fix_annotations.py" | awk '{print $2}')
        echo "✓ 进程运行中 (PID: $PID)"
        ps aux | grep "[c]omprehensive_fix_annotations.py" | grep -v grep | awk '{printf "  CPU: %s%% | 内存: %s%%\n", $3, $4}'
    else
        echo "✗ 进程已结束"
        echo ""
        echo "=== 最终结果 ==="
        tail -30 "$LOG_FILE" 2>/dev/null | grep -E "(处理完成|检查文件|修正文件|修正统计)" || tail -30 "$LOG_FILE"
        break
    fi
    
    echo ""
    echo "=== 当前进度 ==="
    tail -15 "$LOG_FILE" 2>/dev/null | tail -10
    
    echo ""
    echo "=== 修正统计 ==="
    if [ -f "$LOG_FILE" ]; then
        IS_MULTI=$(grep "is_multi_fruit:" "$LOG_FILE" 2>/dev/null | wc -l)
        APPLE_CONTENT=$(grep "apple.*content_type:" "$LOG_FILE" 2>/dev/null | wc -l)
        BANANA_STYLE=$(grep "banana.*image_style:" "$LOG_FILE" 2>/dev/null | wc -l)
        BLUEBERRY_CONTENT=$(grep "blueberry.*content_type:" "$LOG_FILE" 2>/dev/null | wc -l)
        
        echo "is_multi_fruit 修正: $IS_MULTI"
        echo "苹果内容类型修正: $APPLE_CONTENT"
        echo "香蕉风格修正: $BANANA_STYLE"
        echo "蓝莓内容类型修正: $BLUEBERRY_CONTENT"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 5
done
