#!/bin/bash
# 自动监控全面标注脚本的进度，每10秒更新一次

LOG_FILE="/tmp/comprehensive_annotation.log"
PID_FILE="/tmp/comprehensive_annotation_pid.txt"
INTERVAL=10  # 更新间隔（秒）

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

echo "开始监控全面标注脚本 (PID: $PID)"
echo "每 $INTERVAL 秒更新一次，按 Ctrl+C 退出"
echo ""

# 循环监控
while ps -p $PID > /dev/null 2>&1; do
    clear
    echo "=========================================="
    echo "全面标注脚本进度监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 显示进程信息
    echo "=== 进程状态 ==="
    ps -p $PID -o pid,pcpu,pmem,etime,state --no-headers 2>/dev/null | awk '{printf "  PID: %s | CPU: %s%% | 内存: %s%% | 运行时间: %s | 状态: %s\n", $1, $2, $3, $4, $5}'
    
    if [ -f "$LOG_FILE" ]; then
        # 提取进度信息
        echo ""
        echo "=== 当前进度 ==="
        
        # 查找当前处理的split
        CURRENT_SPLIT=$(grep "处理.*split" "$LOG_FILE" | tail -1)
        if [ -n "$CURRENT_SPLIT" ]; then
            echo "  $CURRENT_SPLIT"
        fi
        
        # 查找当前处理的类别（从tqdm输出中提取）
        CURRENT_PROGRESS=$(grep -E "(train|val|test):.*%\|" "$LOG_FILE" | tail -1)
        if [ -n "$CURRENT_PROGRESS" ]; then
            echo "  $CURRENT_PROGRESS"
        fi
        
        # 统计信息
        echo ""
        echo "=== 处理统计 ==="
        
        # 统计每个split的进度条出现次数（近似类别数）
        TRAIN_PROGRESS=$(grep -c "train:" "$LOG_FILE" 2>/dev/null || echo "0")
        VAL_PROGRESS=$(grep -c "val:" "$LOG_FILE" 2>/dev/null || echo "0")
        TEST_PROGRESS=$(grep -c "test:" "$LOG_FILE" 2>/dev/null || echo "0")
        
        echo "  Train split: 约 $TRAIN_PROGRESS 个类别已处理"
        echo "  Val split: 约 $VAL_PROGRESS 个类别已处理"
        echo "  Test split: 约 $TEST_PROGRESS 个类别已处理"
        
        # 显示最新日志
        echo ""
        echo "=== 最新日志 (最后15行) ==="
        tail -15 "$LOG_FILE" | sed 's/^/  /'
        
        # 检查是否完成
        if grep -q "标注统计" "$LOG_FILE"; then
            echo ""
            echo "=== ✅ 标注完成！==="
            echo ""
            grep -A 50 "标注统计" "$LOG_FILE" | head -60
            echo ""
            echo "脚本已完成，退出监控..."
            break
        fi
    else
        echo ""
        echo "等待日志文件创建..."
    fi
    
    echo ""
    echo "=========================================="
    echo "下次更新: $INTERVAL 秒后 (按 Ctrl+C 退出)"
    echo "=========================================="
    
    sleep $INTERVAL
done

# 如果进程已结束
if ! ps -p $PID > /dev/null 2>&1; then
    clear
    echo "=========================================="
    echo "脚本已完成"
    echo "=========================================="
    echo ""
    if [ -f "$LOG_FILE" ]; then
        echo "=== 最终结果 ==="
        tail -100 "$LOG_FILE"
    fi
fi
