#!/usr/bin/env python3
"""
数据集分析脚本
分析github和local数据库中的所有数据集，汇总水果类型和内容类型（完整水果/叶子）
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import re
import csv

# 数据集路径
GITHUB_DB = Path("/home/yuhanlin/Database/github/database")
LOCAL_DB = Path("/home/yuhanlin/Database/local/database")

# 水果关键词（用于识别）
FRUIT_KEYWORDS = {
    'apple', 'pear', 'peach', 'grape', 'strawberry', 'orange', 'banana',
    'watermelon', 'cantaloupe', 'cherry', 'plum', 'apricot', 'kiwi',
    'mango', 'lychee', 'longan', 'pineapple', 'pomelo', 'lemon', 'lime',
    'blueberry', 'raspberry', 'tomato', 'pomegranate', 'avocado', 'fig',
    'guava', 'broccoli', 'pistachio', 'papaya', 'coconut', 'almond',
    'corn', 'pepper', 'potato', 'soybean', 'squash'
}

# 叶子相关关键词
LEAF_KEYWORDS = {'leaf', 'leaves', 'disease', 'scab', 'blight', 'spot', 'rust'}

# 完整水果相关关键词
FRUIT_KEYWORDS_FULL = {'fruit', 'fruits', 'ripe', 'ripeness', 'detection', 'segmentation'}


def extract_fruit_from_name(dataset_name):
    """从数据集名称中提取水果类型"""
    dataset_lower = dataset_name.lower()
    found_fruits = []
    
    for fruit in FRUIT_KEYWORDS:
        if fruit in dataset_lower:
            found_fruits.append(fruit)
    
    return found_fruits if found_fruits else ['unknown']


def check_content_type(dataset_path, dataset_name):
    """
    检查数据集内容类型：
    - '完整水果': 包含完整水果图像
    - '叶子': 只包含叶子图像
    - '混合': 同时包含水果和叶子
    - '未知': 无法确定
    """
    dataset_lower = dataset_name.lower()
    
    # 从名称判断
    has_leaf_keyword = any(kw in dataset_lower for kw in LEAF_KEYWORDS)
    has_fruit_keyword = any(kw in dataset_lower for kw in FRUIT_KEYWORDS_FULL)
    
    # 检查目录结构
    fruit_count = 0
    leaf_count = 0
    sample_images = []
    
    # 尝试找到图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 检查前几层目录
    max_depth = 3
    for root, dirs, files in os.walk(dataset_path):
        depth = root.replace(str(dataset_path), '').count(os.sep)
        if depth > max_depth:
            continue
            
        # 检查文件名和目录名
        dir_name = os.path.basename(root).lower()
        if any(kw in dir_name for kw in LEAF_KEYWORDS):
            leaf_count += 1
        if any(kw in dir_name for kw in FRUIT_KEYWORDS_FULL):
            fruit_count += 1
        
        # 收集样本图像
        for file in files[:5]:  # 只检查前5个文件
            if Path(file).suffix.lower() in image_extensions:
                sample_images.append(os.path.join(root, file))
                if len(sample_images) >= 10:
                    break
        
        if len(sample_images) >= 10:
            break
    
    # 分析样本图像文件名
    for img_path in sample_images[:5]:
        img_name = os.path.basename(img_path).lower()
        if any(kw in img_name for kw in LEAF_KEYWORDS):
            leaf_count += 1
        if any(kw in img_name for kw in FRUIT_KEYWORDS_FULL):
            fruit_count += 1
    
    # 判断类型
    if has_leaf_keyword and not has_fruit_keyword and leaf_count > fruit_count:
        return '叶子'
    elif has_fruit_keyword and not has_leaf_keyword and fruit_count > leaf_count:
        return '完整水果'
    elif has_leaf_keyword and has_fruit_keyword:
        return '混合'
    elif leaf_count > 0 and fruit_count > 0:
        return '混合'
    elif leaf_count > 0:
        return '叶子'
    elif fruit_count > 0:
        return '完整水果'
    else:
        return '未知'


def count_images(dataset_path):
    """统计数据集中的图像数量"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    count = 0
    
    try:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    count += 1
    except Exception as e:
        print(f"  警告: 无法统计 {dataset_path}: {e}")
    
    return count


def analyze_dataset(dataset_path, source):
    """分析单个数据集"""
    dataset_name = dataset_path.name
    
    # 提取水果类型
    fruits = extract_fruit_from_name(dataset_name)
    
    # 检查内容类型
    content_type = check_content_type(dataset_path, dataset_name)
    
    # 统计图像数量
    image_count = count_images(dataset_path)
    
    return {
        '数据集名称': dataset_name,
        '来源': source,
        '水果类型': ', '.join(fruits) if fruits else '未知',
        '内容类型': content_type,
        '图像数量': image_count,
        '路径': str(dataset_path)
    }


def main():
    """主函数"""
    print("🔍 开始分析数据集...")
    print(f"GitHub数据库: {GITHUB_DB}")
    print(f"Local数据库: {LOCAL_DB}")
    print()
    
    results = []
    
    # 分析GitHub数据库
    if GITHUB_DB.exists():
        print(f"📁 分析GitHub数据库...")
        for item in sorted(GITHUB_DB.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                print(f"  分析: {item.name}")
                try:
                    result = analyze_dataset(item, 'GitHub')
                    results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    results.append({
                        '数据集名称': item.name,
                        '来源': 'GitHub',
                        '水果类型': '错误',
                        '内容类型': '错误',
                        '图像数量': 0,
                        '路径': str(item)
                    })
    
    print()
    
    # 分析Local数据库
    if LOCAL_DB.exists():
        print(f"📁 分析Local数据库...")
        for item in sorted(LOCAL_DB.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                print(f"  分析: {item.name}")
                try:
                    result = analyze_dataset(item, 'Local')
                    results.append(result)
                except Exception as e:
                    print(f"    错误: {e}")
                    results.append({
                        '数据集名称': item.name,
                        '来源': 'Local',
                        '水果类型': '错误',
                        '内容类型': '错误',
                        '图像数量': 0,
                        '路径': str(item)
                    })
    
    # 保存为CSV
    output_csv = Path(__file__).parent.parent / 'dataset_analysis.csv'
    if results:
        fieldnames = ['数据集名称', '来源', '水果类型', '内容类型', '图像数量', '路径']
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ 分析结果已保存到: {output_csv}")
    
    # 统计信息
    total_count = len(results)
    github_count = sum(1 for r in results if r['来源'] == 'GitHub')
    local_count = sum(1 for r in results if r['来源'] == 'Local')
    total_images = sum(r['图像数量'] for r in results)
    
    # 按内容类型统计
    content_stats = defaultdict(int)
    for r in results:
        content_stats[r['内容类型']] += 1
    
    # 按水果类型统计
    fruit_counts = defaultdict(int)
    for r in results:
        for fruit in r['水果类型'].split(', '):
            fruit_counts[fruit] += 1
    
    # 保存为Markdown表格
    output_md = Path(__file__).parent.parent / 'dataset_analysis.md'
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# 数据集分析汇总表\n\n")
        f.write("## 统计信息\n\n")
        f.write(f"- **总数据集数量**: {total_count}\n")
        f.write(f"- **GitHub数据集**: {github_count}\n")
        f.write(f"- **Local数据集**: {local_count}\n")
        f.write(f"- **总图像数量**: {total_images:,}\n\n")
        
        f.write("## 按内容类型统计\n\n")
        f.write("| 内容类型 | 数据集数量 |\n")
        f.write("|---------|-----------|\n")
        for content_type, count in sorted(content_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {content_type} | {count} |\n")
        f.write("\n")
        
        f.write("## 按水果类型统计（前15）\n\n")
        f.write("| 水果类型 | 数据集数量 |\n")
        f.write("|---------|-----------|\n")
        for fruit, count in sorted(fruit_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            f.write(f"| {fruit} | {count} |\n")
        f.write("\n")
        
        f.write("## 数据集详情\n\n")
        f.write("| 数据集名称 | 来源 | 水果类型 | 内容类型 | 图像数量 |\n")
        f.write("|-----------|------|---------|---------|---------|\n")
        for r in results:
            name = r['数据集名称']
            source = r['来源']
            fruits = r['水果类型']
            content = r['内容类型']
            count = r['图像数量']
            f.write(f"| {name} | {source} | {fruits} | {content} | {count:,} |\n")
    
    print(f"✅ Markdown表格已保存到: {output_md}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 分析摘要")
    print("="*80)
    print(f"\n总数据集数量: {total_count}")
    print(f"GitHub数据集: {github_count}")
    print(f"Local数据集: {local_count}")
    print(f"总图像数量: {total_images:,}")
    
    print("\n按内容类型统计:")
    for content_type, count in sorted(content_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {content_type}: {count}个数据集")
    
    print("\n按水果类型统计（前10）:")
    for fruit, count in sorted(fruit_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fruit}: {count}个数据集")
    
    print("\n" + "="*80)
    print(f"\n详细结果请查看: {output_csv} 和 {output_md}")


if __name__ == '__main__':
    main()
