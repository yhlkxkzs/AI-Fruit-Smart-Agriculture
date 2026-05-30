#!/usr/bin/env python3
"""
基于prepare_dataset.py的逻辑推断unknown图像的来源
分析哪些数据集会产生unknown图像
"""
import os
import json
from pathlib import Path
from collections import defaultdict

# 路径配置
UNKNOWN_DIRS = [
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
]

GITHUB_DB = Path('/home/yuhanlin/Database/github/database')
LOCAL_DB = Path('/home/yuhanlin/Database/local/database')

# 从prepare_dataset.py复制的逻辑
MULTIFRUIT_DATASETS = ['deepfruits', 'fruit-salad', 'acfr-multifruit-2016', 'Multi-species-fruit-flower', 'RawRipe', 'plant_village']

def count_unknown_images():
    """统计unknown图像数量"""
    total = 0
    for unknown_dir in UNKNOWN_DIRS:
        if os.path.exists(unknown_dir):
            count = len([f for f in os.listdir(unknown_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            total += count
            print(f"  {Path(unknown_dir).parent.name}/unknown: {count} 张")
    return total

def analyze_multifruit_datasets():
    """分析多水果数据集，找出可能产生unknown图像的目录"""
    sources_summary = defaultdict(lambda: {'count': 0, 'subdirectories': defaultdict(int)})
    
    # 处理GitHub数据库
    if GITHUB_DB.exists():
        print("\n分析GitHub数据库中的多水果数据集...")
        for dataset_name in MULTIFRUIT_DATASETS:
            dataset_path = GITHUB_DB / dataset_name
            if not dataset_path.exists() or not dataset_path.is_dir():
                continue
            
            print(f"  检查数据集: {dataset_name}...")
            unknown_dirs = find_unknown_directories(dataset_path)
            if unknown_dirs:
                for subdir, count in unknown_dirs.items():
                    sources_summary[dataset_name]['count'] += count
                    sources_summary[dataset_name]['subdirectories'][subdir] += count
    
    # 处理本地数据库
    if LOCAL_DB.exists():
        print("\n分析本地数据库中的多水果数据集...")
        for dataset_name in MULTIFRUIT_DATASETS:
            dataset_path = LOCAL_DB / dataset_name
            if not dataset_path.exists() or not dataset_path.is_dir():
                continue
            
            print(f"  检查数据集: {dataset_name}...")
            unknown_dirs = find_unknown_directories(dataset_path)
            if unknown_dirs:
                for subdir, count in unknown_dirs.items():
                    sources_summary[dataset_name]['count'] += count
                    sources_summary[dataset_name]['subdirectories'][subdir] += count
    
    return sources_summary

def find_unknown_directories(dataset_path):
    """找出可能产生unknown图像的目录"""
    unknown_dirs = {}
    
    # 检查子目录
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d)) 
               and d not in ['__pycache__', '.git', 'scripts', 'labels', 'annotations', 'images', 'data', 'docs']]
    
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            # 检查目录名是否包含已知水果名称
            # 这里使用简化的逻辑：如果目录名不包含常见水果关键词，可能产生unknown
            subdir_lower = subdir.lower()
            common_fruit_keywords = ['apple', 'banana', 'orange', 'grape', 'strawberry', 'peach', 
                                    'pear', 'cherry', 'mango', 'kiwi', 'plum', 'apricot', 'pineapple',
                                    'pomegranate', 'avocado', 'fig', 'guava', 'papaya', 'coconut',
                                    'lychee', 'lemon', 'lime', 'blueberry', 'raspberry', 'tomato']
            
            # 如果目录名不包含任何水果关键词，可能产生unknown
            if not any(keyword in subdir_lower for keyword in common_fruit_keywords):
                # 统计图像数量
                image_count = count_images_in_dir(subdir_path)
                if image_count > 0:
                    unknown_dirs[subdir] = image_count
    else:
        # 如果没有子目录，根目录可能产生unknown
        image_count = count_images_in_dir(dataset_path)
        if image_count > 0:
            unknown_dirs['根目录'] = image_count
    
    return unknown_dirs

def count_images_in_dir(directory):
    """统计目录中的图像数量（快速估算）"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
    count = 0
    try:
        for root, dirs, files in os.walk(directory):
            # 限制深度，只检查前两层
            depth = root[len(directory):].count(os.sep)
            if depth > 2:
                dirs[:] = []
                continue
            
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                    count += 1
                    if count >= 1000:  # 限制最大计数
                        return count
    except:
        pass
    return count

def main():
    """主函数"""
    print("=" * 70)
    print("Unknown图像来源推断分析")
    print("=" * 70)
    
    # 1. 统计unknown图像数量
    print("\n1. 统计unknown图像数量:")
    total_unknown = count_unknown_images()
    print(f"   总计: {total_unknown} 张")
    
    # 2. 分析多水果数据集
    print("\n2. 分析可能产生unknown图像的数据集...")
    sources_summary = analyze_multifruit_datasets()
    
    # 3. 保存结果
    output_file = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_sources.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_unknown_images': total_unknown,
            'note': '此分析基于prepare_dataset.py的逻辑推断，可能产生unknown图像的数据集和目录',
            'summary_by_dataset': {k: {
                'total': v['count'],
                'subdirectories': dict(v['subdirectories'])
            } for k, v in sources_summary.items()}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析完成！")
    print(f"   结果已保存到: {output_file}")
    
    # 4. 打印摘要
    if sources_summary:
        print("\n可能产生unknown图像的数据集（Top 20）:")
        print("-" * 70)
        sorted_datasets = sorted(sources_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
        for i, (dataset, info) in enumerate(sorted_datasets, 1):
            subdirs_count = len(info['subdirectories'])
            print(f"{i:2d}. {dataset:45s}: 约 {info['count']:6d} 张 (来自 {subdirs_count} 个子目录)")
            if info['subdirectories']:
                top_subdirs = sorted(info['subdirectories'].items(), key=lambda x: x[1], reverse=True)[:3]
                for subdir, count in top_subdirs:
                    print(f"    └─ {subdir}: 约 {count} 张")
    else:
        print("\n⚠️  未找到可能产生unknown图像的数据集")

if __name__ == '__main__':
    main()
