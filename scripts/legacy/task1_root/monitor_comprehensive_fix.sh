#!/bin/bash
# 监控综合修正脚本的实时进度

LOG_FILE="/tmp/comprehensive_fix_annotations.log"

echo "=========================================="
echo "综合修正标注脚本 - 实时进度监控"
echo "=========================================="

# 检查进程是否运行
if ps aux | grep -q "[c]omprehensive_fix_annotations.py"; then
    PID=$(ps aux | grep "[c]omprehensive_fix_annotations.py" | awk '{print $2}')
    echo "✓ 进程正在运行 (PID: $PID)"
    
    # 显示进程信息
    ps aux | grep "[c]omprehensive_fix_annotations.py" | grep -v grep | awk '{printf "  CPU: %s%% | 内存: %s%%\n", $3, $4}'
    
    # 计算运行时间
    START_TIME=$(ps -o lstart= -p $PID 2>/dev/null)
    if [ -n "$START_TIME" ]; then
        echo "  启动时间: $START_TIME"
    fi
else
    echo "✗ 进程未运行"
fi

echo ""
echo "=== 最新进度 ==="
if [ -f "$LOG_FILE" ]; then
    # 显示最后几行进度
    tail -10 "$LOG_FILE" | grep -E "(处理|train|val|test|修正|完成|检查)" || tail -10 "$LOG_FILE"
    
    echo ""
    echo "=== 修正统计 ==="
    IS_MULTI=$(grep -c "is_multi_fruit:" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    APPLE_CONTENT=$(grep -c "apple.*content_type:" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    BANANA_STYLE=$(grep -c "banana.*image_style:" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    BLUEBERRY_CONTENT=$(grep -c "blueberry.*content_type:" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    
    echo "is_multi_fruit 修正: $IS_MULTI"
    echo "苹果内容类型修正: $APPLE_CONTENT"
    echo "香蕉风格修正: $BANANA_STYLE"
    echo "蓝莓内容类型修正: $BLUEBERRY_CONTENT"
    
    # 显示总修正数
    TOTAL_FIXED=$(grep -c "修正了" "$LOG_FILE" 2>/dev/null | head -1 || echo "0")
    if [ -n "$TOTAL_FIXED" ] && [ "$TOTAL_FIXED" -gt 0 ] 2>/dev/null; then
        echo ""
        echo "总修正文件数: $TOTAL_FIXED"
    fi
else
    echo "日志文件不存在"
fi

echo ""
echo "=========================================="
echo "实时查看: tail -f $LOG_FILE"
echo "快速查看: bash scripts/monitor_comprehensive_fix.sh"
echo "=========================================="
