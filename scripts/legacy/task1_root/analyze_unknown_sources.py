#!/usr/bin/env python3
"""
分析unknown图像的原始来源，尝试确定类别
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# 读取数据集统计信息，查看unknown的来源
def analyze_unknown_sources():
    """分析unknown图像的来源"""
    
    base_dir = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset")
    
    # 检查是否有日志或统计文件记录unknown的来源
    stats_file = base_dir / "dataset_stats.json"
    
    unknown_info = {
        'train': {'count': 0, 'sources': []},
        'val': {'count': 0, 'sources': []},
        'test': {'count': 0, 'sources': []}
    }
    
    # 统计unknown图像数量
    for split in ['train', 'val', 'test']:
        unknown_dir = base_dir / split / "unknown"
        if unknown_dir.exists():
            images = list(unknown_dir.glob("*.jpg")) + list(unknown_dir.glob("*.png")) + list(unknown_dir.glob("*.jpeg"))
            unknown_info[split]['count'] = len(images)
    
    # 检查prepare_dataset.py的日志或运行历史
    # 由于unknown图像已经被重命名，我们需要查看原始数据集来确定来源
    
    print("="*80)
    print("Unknown图像分析报告")
    print("="*80)
    
    total_unknown = sum(info['count'] for info in unknown_info.values())
    print(f"\n📊 Unknown图像统计:")
    print(f"  训练集: {unknown_info['train']['count']} 张")
    print(f"  验证集: {unknown_info['val']['count']} 张")
    print(f"  测试集: {unknown_info['test']['count']} 张")
    print(f"  总计: {total_unknown} 张")
    
    print(f"\n⚠️  问题分析:")
    print(f"  - Unknown图像的文件名已被重命名为 unknown_xxxxx 格式")
    print(f"  - 无法从文件名直接推断原始类别")
    print(f"  - 需要查看原始数据集或运行日志来确定来源")
    
    print(f"\n💡 建议:")
    print(f"  1. 查看数据集准备脚本的运行日志，了解unknown图像的原始来源")
    print(f"  2. 重新运行数据集准备脚本，添加更详细的日志记录")
    print(f"  3. 对于无法自动分类的图像，可以考虑:")
    print(f"     - 使用预训练模型进行初步分类")
    print(f"     - 人工抽样检查并分类")
    print(f"     - 暂时保留为unknown类别，在训练时作为负样本")
    
    # 检查可能包含unknown图像的数据集
    print(f"\n🔍 可能包含unknown图像的数据集:")
    print(f"  根据之前的分析，以下数据集可能产生了unknown图像:")
    print(f"  - 目录结构复杂的数据集（如deepfruits, fruit-salad）")
    print(f"  - 包含非标准命名的数据集")
    print(f"  - 无法匹配到已知水果类别的数据集")
    
    return unknown_info

if __name__ == '__main__':
    info = analyze_unknown_sources()
    
    # 保存结果
    output_file = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/unknown_sources_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存到: {output_file}")
