#!/bin/bash
# 等待数据集准备脚本完成并检查结果

echo "等待数据集准备脚本完成..."
echo "脚本PID: $(ps aux | grep 'prepare_dataset.py' | grep -v grep | awk '{print $2}')"

# 等待脚本完成
while ps aux | grep -q '[p]repare_dataset.py'; do
    echo -n "."
    sleep 10
done

echo ""
echo "脚本已完成！"

# 检查结果文件
if [ -f "/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_sources.json" ]; then
    echo "✅ unknown_sources.json 已生成"
    echo ""
    echo "文件大小: $(ls -lh /home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_sources.json | awk '{print $5}')"
    echo ""
    echo "数据集来源摘要（Top 10）:"
    python3 -c "
import json
with open('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_sources.json', 'r') as f:
    data = json.load(f)
    summary = data.get('summary_by_dataset', {})
    sorted_datasets = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    for i, (dataset, info) in enumerate(sorted_datasets, 1):
        print(f'  {i:2d}. {dataset:40s}: {info[\"total\"]:5d} 张')
"
else
    echo "❌ unknown_sources.json 未生成"
fi
