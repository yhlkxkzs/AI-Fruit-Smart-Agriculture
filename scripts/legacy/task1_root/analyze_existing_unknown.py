#!/usr/bin/env python3
"""
分析现有unknown图像的来源
通过比较文件哈希值来匹配原始数据集
"""
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

# 路径配置
UNKNOWN_DIRS = [
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
]

GITHUB_DB = Path('/home/yuhanlin/Database/github/database')
LOCAL_DB = Path('/home/yuhanlin/Database/local/database')

def get_file_hash(filepath):
    """计算文件的MD5哈希值"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        return None

def get_file_size(filepath):
    """获取文件大小"""
    try:
        return os.path.getsize(filepath)
    except:
        return None

def find_images_in_dataset(dataset_path, max_files=None):
    """在数据集中查找所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for root, dirs, files in os.walk(dataset_path):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'scripts', 'labels', 'annotations', 'data', 'docs']]
        
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                filepath = os.path.join(root, file)
                try:
                    # 验证图像
                    with Image.open(filepath) as img:
                        img.verify()
                    images.append(filepath)
                    if max_files and len(images) >= max_files:
                        return images
                except:
                    continue
    
    return images

def analyze_unknown_sources():
    """分析unknown图像的来源"""
    print("=" * 70)
    print("分析Unknown图像来源")
    print("=" * 70)
    
    # 1. 收集所有unknown图像
    unknown_images = []
    for unknown_dir in UNKNOWN_DIRS:
        if os.path.exists(unknown_dir):
            for filename in os.listdir(unknown_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    filepath = os.path.join(unknown_dir, filename)
                    unknown_images.append({
                        'path': filepath,
                        'filename': filename,
                        'split': Path(unknown_dir).parent.name
                    })
    
    print(f"找到 {len(unknown_images)} 张unknown图像")
    
    # 2. 为unknown图像计算哈希值（使用文件大小作为快速匹配）
    print("\n计算unknown图像的文件大小...")
    unknown_sizes = {}
    for img_info in unknown_images:
        size = get_file_size(img_info['path'])
        if size:
            if size not in unknown_sizes:
                unknown_sizes[size] = []
            unknown_sizes[size].append(img_info)
    
    print(f"  不同大小的文件数: {len(unknown_sizes)}")
    
    # 3. 在原始数据集中查找匹配
    print("\n在原始数据集中查找匹配...")
    sources_summary = defaultdict(lambda: {'count': 0, 'subdirectories': defaultdict(int)})
    matched_count = 0
    
    # 处理GitHub数据库
    if GITHUB_DB.exists():
        print("\n处理GitHub数据库...")
        for dataset_name in sorted(os.listdir(GITHUB_DB)):
            dataset_path = GITHUB_DB / dataset_name
            if not dataset_path.is_dir() or dataset_name == 'scripts':
                continue
            
            print(f"  检查数据集: {dataset_name}...")
            # 只检查文件大小匹配的图像
            dataset_images = find_images_in_dataset(dataset_path, max_files=10000)  # 限制数量以提高速度
            
            for dataset_img in dataset_images:
                size = get_file_size(dataset_img)
                if size and size in unknown_sizes:
                    # 计算哈希值进行精确匹配
                    dataset_hash = get_file_hash(dataset_img)
                    if dataset_hash:
                        for unknown_info in unknown_sizes[size]:
                            unknown_hash = get_file_hash(unknown_info['path'])
                            if unknown_hash == dataset_hash:
                                # 找到匹配！
                                subdir = Path(dataset_img).parent.name
                                sources_summary[dataset_name]['count'] += 1
                                sources_summary[dataset_name]['subdirectories'][subdir] += 1
                                matched_count += 1
                                if matched_count % 10 == 0:
                                    print(f"    已匹配 {matched_count} 张图像...")
    
    # 处理本地数据库
    if LOCAL_DB.exists():
        print("\n处理本地数据库...")
        for dataset_name in sorted(os.listdir(LOCAL_DB)):
            dataset_path = LOCAL_DB / dataset_name
            if not dataset_path.is_dir():
                continue
            
            print(f"  检查数据集: {dataset_name}...")
            dataset_images = find_images_in_dataset(dataset_path, max_files=10000)
            
            for dataset_img in dataset_images:
                size = get_file_size(dataset_img)
                if size and size in unknown_sizes:
                    dataset_hash = get_file_hash(dataset_img)
                    if dataset_hash:
                        for unknown_info in unknown_sizes[size]:
                            unknown_hash = get_file_hash(unknown_info['path'])
                            if unknown_hash == dataset_hash:
                                subdir = Path(dataset_img).parent.name
                                sources_summary[dataset_name]['count'] += 1
                                sources_summary[dataset_name]['subdirectories'][subdir] += 1
                                matched_count += 1
                                if matched_count % 10 == 0:
                                    print(f"    已匹配 {matched_count} 张图像...")
    
    # 4. 保存结果
    output_file = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_sources.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_unknown_images': len(unknown_images),
            'matched_images': matched_count,
            'unmatched_images': len(unknown_images) - matched_count,
            'summary_by_dataset': {k: {
                'total': v['count'],
                'subdirectories': dict(v['subdirectories'])
            } for k, v in sources_summary.items()}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析完成！")
    print(f"   总unknown图像数: {len(unknown_images)}")
    print(f"   已匹配图像数: {matched_count}")
    print(f"   未匹配图像数: {len(unknown_images) - matched_count}")
    print(f"   涉及数据集数: {len(sources_summary)}")
    print(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    if sources_summary:
        print("\n数据集来源分布（Top 20）:")
        print("-" * 70)
        sorted_datasets = sorted(sources_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
        for i, (dataset, info) in enumerate(sorted_datasets, 1):
            subdirs_count = len(info['subdirectories'])
            print(f"{i:2d}. {dataset:45s}: {info['count']:6d} 张 (来自 {subdirs_count} 个子目录)")
            if info['subdirectories']:
                top_subdirs = sorted(info['subdirectories'].items(), key=lambda x: x[1], reverse=True)[:3]
                for subdir, count in top_subdirs:
                    print(f"    └─ {subdir}: {count} 张")

if __name__ == '__main__':
    analyze_unknown_sources()
