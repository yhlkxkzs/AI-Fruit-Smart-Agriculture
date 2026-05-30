#!/bin/bash
# 监控苹果标注验证和修正脚本的进度

LOG_FILE="/tmp/verify_apple_annotations.log"

echo "=========================================="
echo "苹果标注验证和修正 - 进度监控"
echo "=========================================="

# 检查进程是否运行
if ps aux | grep -q "[v]erify_and_fix_apple_annotations.py"; then
    PID=$(ps aux | grep "[v]erify_and_fix_apple_annotations.py" | awk '{print $2}')
    echo "✓ 进程正在运行 (PID: $PID)"
    
    # 显示进程信息
    ps aux | grep "[v]erify_and_fix_apple_annotations.py" | grep -v grep | awk '{printf "  CPU: %s%% | 内存: %s%%\n", $3, $4}'
else
    echo "✗ 进程未运行"
fi

echo ""
echo "=== 最新进度 ==="
if [ -f "$LOG_FILE" ]; then
    # 显示最后几行进度
    tail -5 "$LOG_FILE" | grep -E "(处理|train|val|test|修正|完成)" || tail -5 "$LOG_FILE"
    
    echo ""
    echo "=== 修正统计 ==="
    CORRECTED=$(grep -c "修正:" "$LOG_FILE" 2>/dev/null || echo "0")
    DELETED=$(grep -c "删除" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "已修正标注: $CORRECTED"
    echo "已删除黑白照片: $DELETED"
else
    echo "日志文件不存在"
fi

echo ""
echo "=========================================="
echo "实时查看: tail -f $LOG_FILE"
echo "=========================================="
