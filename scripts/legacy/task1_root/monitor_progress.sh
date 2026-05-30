#!/bin/bash
# 监控修复脚本的进度

PID_FILE="/tmp/fix_all_images_pid.txt"
PID=$(cat "$PID_FILE" 2>/dev/null)

if [ -z "$PID" ]; then
    # 尝试从进程列表中找到PID
    PID=$(ps aux | grep "fix_all_images_and_regenerate.py" | grep -v grep | awk '{print $2}' | head -1)
fi

if [ -z "$PID" ]; then
    echo "未找到运行中的脚本"
    exit 1
fi

echo "脚本正在运行 (PID: $PID)"
echo ""

# 检查进程是否还在运行
if ! ps -p $PID > /dev/null 2>&1; then
    echo "脚本已完成"
    exit 0
fi

# 查找日志文件
LOG_FILES=(
    "/tmp/fix_all_images_progress.log"
    "/tmp/fix_all_images_full.log"
    "/tmp/fix_progress.log"
    "/proc/$PID/fd/1"
)

LOG_FILE=""
for f in "${LOG_FILES[@]}"; do
    if [ -f "$f" ] || [ -r "$f" ]; then
        LOG_FILE="$f"
        break
    fi
done

if [ -z "$LOG_FILE" ]; then
    echo "未找到日志文件"
    echo "进程信息:"
    ps aux | grep $PID | grep -v grep
    exit 1
fi

echo "=== 最新进度 (最后40行) ==="
tail -40 "$LOG_FILE" 2>/dev/null || echo "无法读取日志文件"

echo ""
echo "=== 统计信息 ==="
if [ -f "$LOG_FILE" ]; then
    TOTAL_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    echo "日志总行数: $TOTAL_LINES"
    
    if grep -q "总计重命名\|总计删除" "$LOG_FILE" 2>/dev/null; then
        echo "状态: ✅ 修复完成，正在重新生成标注文件"
        echo ""
        grep "总计重命名\|总计删除" "$LOG_FILE" | tail -2
    elif grep -q "重新生成所有标注文件" "$LOG_FILE" 2>/dev/null; then
        echo "状态: 🔄 正在重新生成标注文件"
    elif grep -q "处理.*split" "$LOG_FILE" 2>/dev/null; then
        LAST_CATEGORY=$(grep "处理.*/" "$LOG_FILE" | tail -1)
        echo "状态: 🔄 正在修复图片命名"
        echo "当前处理: $LAST_CATEGORY"
    else
        echo "状态: 🔄 脚本正在运行"
    fi
fi

echo ""
echo "=== 进程资源使用 ==="
ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null || echo "无法获取进程信息"
