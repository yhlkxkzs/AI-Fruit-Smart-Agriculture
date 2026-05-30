#!/usr/bin/env python3
"""
数据集加载和处理脚本
用于加载和处理未知类型和混合类型的数据集
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

# 数据集路径
GITHUB_DB = Path("/home/yuhanlin/Database/github/database")
LOCAL_DB = Path("/home/yuhanlin/Database/local/database")
OUTPUT_DIR = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset_processed")

# 未知数据集列表
UNKNOWN_DATASETS = [
    {
        'name': 'Pistachio',
        'path': GITHUB_DB / 'Pistachio',
        'source': 'GitHub',
        'fruit_type': 'pistachio',
        'image_count': 4296
    },
    {
        'name': 'cassava',
        'path': GITHUB_DB / 'cassava',
        'source': 'GitHub',
        'fruit_type': 'unknown',
        'image_count': 28121
    },
    {
        'name': 'lemon-datase',
        'path': GITHUB_DB / 'lemon-datase',
        'source': 'GitHub',
        'fruit_type': 'lemon',
        'image_count': 5400
    },
    {
        'name': 'carrot_weeds_germany',
        'path': LOCAL_DB / 'carrot_weeds_germany',
        'source': 'Local',
        'fruit_type': 'unknown',
        'image_count': 300
    },
    {
        'name': 'deep_weeds',
        'path': LOCAL_DB / 'deep_weeds',
        'source': 'Local',
        'fruit_type': 'unknown',
        'image_count': 8160
    },
    {
        'name': 'rangeland_weeds_australia',
        'path': LOCAL_DB / 'rangeland_weeds_australia',
        'source': 'Local',
        'fruit_type': 'unknown',
        'image_count': 1765
    },
    {
        'name': 'soybean_weed_uav_brazil',
        'path': LOCAL_DB / 'soybean_weed_uav_brazil',
        'source': 'Local',
        'fruit_type': 'soybean',
        'image_count': 15336
    },
    {
        'name': 'wheat_head_counting',
        'path': LOCAL_DB / 'wheat_head_counting',
        'source': 'Local',
        'fruit_type': 'unknown',
        'image_count': 8117
    }
]

# 混合数据集列表
MIXED_DATASETS = [
    {
        'name': 'AppleScabFDs',
        'path': GITHUB_DB / 'AppleScabFDs',
        'source': 'GitHub',
        'fruit_type': 'apple',
        'image_count': 297,
        'fruit_dirs': ['apples/healthy'],  # 可能包含完整水果的目录
        'leaf_dirs': ['apples/scab']  # 可能包含叶子的目录
    },
    {
        'name': 'cucumber_disease_classification',
        'path': LOCAL_DB / 'cucumber_disease_classification',
        'source': 'Local',
        'fruit_type': 'cucumber',
        'image_count': 14258,
        'fruit_dirs': ['cucumbers/fresh_cucumber'],  # 完整黄瓜
        'leaf_dirs': ['cucumbers/fresh_leaf']  # 黄瓜叶子
    },
    {
        'name': 'plant_doc_detection',
        'path': LOCAL_DB / 'plant_doc_detection',
        'source': 'Local',
        'fruit_type': 'multiple',
        'image_count': 5156,
        'fruit_keywords': ['fruit', 'cucumber', 'tomato', 'apple', 'strawberry', 'pepper', 'corn', 'grape', 'cherry', 'peach'],
        'leaf_keywords': ['leaf', 'leaves', 'disease']
    }
]

# 图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}


def validate_image(image_path):
    """验证图像文件是否有效"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def find_images(directory, max_depth=5):
    """在目录中查找所有图像文件"""
    images = []
    for root, dirs, files in os.walk(directory):
        depth = root.replace(str(directory), '').count(os.sep)
        if depth > max_depth:
            continue
        
        for file in files:
            if Path(file).suffix in IMAGE_EXTENSIONS:
                images.append(Path(root) / file)
    return images


def analyze_unknown_dataset(dataset_info):
    """分析未知数据集，检查是否包含完整水果"""
    print(f"\n{'='*80}")
    print(f"🔍 分析未知数据集: {dataset_info['name']}")
    print(f"{'='*80}")
    
    dataset_path = dataset_info['path']
    
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return None
    
    # 查找所有图像
    print(f"📁 扫描目录: {dataset_path}")
    images = find_images(dataset_path)
    print(f"   找到 {len(images)} 张图像")
    
    if len(images) == 0:
        print("⚠️  未找到图像文件")
        return None
    
    # 分析目录结构
    print(f"\n📂 目录结构分析:")
    dirs = []
    for root, dirnames, filenames in os.walk(dataset_path):
        depth = root.replace(str(dataset_path), '').count(os.sep)
        if depth <= 2:  # 只显示前两层
            rel_path = os.path.relpath(root, dataset_path)
            if rel_path == '.':
                rel_path = '根目录'
            dirs.append((depth, rel_path, len([f for f in filenames if Path(f).suffix in IMAGE_EXTENSIONS])))
    
    for depth, dir_name, img_count in sorted(dirs):
        indent = "  " * depth
        print(f"{indent}{dir_name}: {img_count} 张图像")
    
    # 检查样本图像
    print(f"\n🖼️  样本图像分析 (前10张):")
    sample_images = images[:10]
    for i, img_path in enumerate(sample_images, 1):
        print(f"   {i}. {img_path.name}")
        # 检查文件名中是否包含水果关键词
        img_name_lower = img_path.name.lower()
        fruit_keywords = ['fruit', 'apple', 'orange', 'banana', 'strawberry', 'grape', 'cherry', 'peach', 'pear', 'tomato', 'cucumber', 'pistachio', 'lemon']
        found_keywords = [kw for kw in fruit_keywords if kw in img_name_lower]
        if found_keywords:
            print(f"      → 可能包含: {', '.join(found_keywords)}")
    
    # 统计信息
    result = {
        'name': dataset_info['name'],
        'path': str(dataset_path),
        'total_images': len(images),
        'directories': len(set(img.parent for img in images)),
        'sample_images': [str(img) for img in sample_images[:5]]
    }
    
    return result


def analyze_mixed_dataset(dataset_info):
    """分析混合数据集，分离完整水果和叶子图像"""
    print(f"\n{'='*80}")
    print(f"🔍 分析混合数据集: {dataset_info['name']}")
    print(f"{'='*80}")
    
    dataset_path = dataset_info['path']
    
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return None
    
    # 查找所有图像
    print(f"📁 扫描目录: {dataset_path}")
    all_images = find_images(dataset_path)
    print(f"   找到 {len(all_images)} 张图像")
    
    if len(all_images) == 0:
        print("⚠️  未找到图像文件")
        return None
    
    # 根据数据集类型分类
    fruit_images = []
    leaf_images = []
    unknown_images = []
    
    if dataset_info['name'] == 'AppleScabFDs':
        # AppleScabFDs: apples/healthy 可能是完整水果，apples/scab 可能是叶子
        for img in all_images:
            img_str = str(img)
            if 'healthy' in img_str.lower():
                fruit_images.append(img)
            elif 'scab' in img_str.lower():
                leaf_images.append(img)
            else:
                unknown_images.append(img)
    
    elif dataset_info['name'] == 'cucumber_disease_classification':
        # cucumber: fresh_cucumber 是完整水果，fresh_leaf 是叶子
        for img in all_images:
            img_str = str(img)
            if 'fresh_cucumber' in img_str or 'cucumber' in img_str.lower() and 'leaf' not in img_str.lower():
                fruit_images.append(img)
            elif 'fresh_leaf' in img_str or 'leaf' in img_str.lower():
                leaf_images.append(img)
            else:
                unknown_images.append(img)
    
    elif dataset_info['name'] == 'plant_doc_detection':
        # plant_doc: 根据文件名关键词判断
        for img in all_images:
            img_name_lower = img.name.lower()
            if any(kw in img_name_lower for kw in dataset_info.get('fruit_keywords', [])):
                if not any(kw in img_name_lower for kw in dataset_info.get('leaf_keywords', [])):
                    fruit_images.append(img)
                else:
                    leaf_images.append(img)
            elif any(kw in img_name_lower for kw in dataset_info.get('leaf_keywords', [])):
                leaf_images.append(img)
            else:
                unknown_images.append(img)
    
    # 统计信息
    print(f"\n📊 分类统计:")
    print(f"   完整水果图像: {len(fruit_images)}")
    print(f"   叶子图像: {len(leaf_images)}")
    print(f"   未知类型: {len(unknown_images)}")
    
    # 显示样本
    if fruit_images:
        print(f"\n🍎 完整水果样本 (前5张):")
        for i, img in enumerate(fruit_images[:5], 1):
            print(f"   {i}. {img.name}")
    
    if leaf_images:
        print(f"\n🍃 叶子样本 (前5张):")
        for i, img in enumerate(leaf_images[:5], 1):
            print(f"   {i}. {img.name}")
    
    result = {
        'name': dataset_info['name'],
        'path': str(dataset_path),
        'total_images': len(all_images),
        'fruit_images': len(fruit_images),
        'leaf_images': len(leaf_images),
        'unknown_images': len(unknown_images),
        'fruit_image_paths': [str(img) for img in fruit_images[:10]],
        'leaf_image_paths': [str(img) for img in leaf_images[:10]]
    }
    
    return result


def main():
    """主函数"""
    print("="*80)
    print("📦 数据集加载和处理工具")
    print("="*80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'unknown_datasets': [],
        'mixed_datasets': []
    }
    
    # 分析未知数据集
    print("\n" + "="*80)
    print("🔴 分析未知数据集")
    print("="*80)
    
    for dataset_info in UNKNOWN_DATASETS:
        result = analyze_unknown_dataset(dataset_info)
        if result:
            results['unknown_datasets'].append(result)
    
    # 分析混合数据集
    print("\n" + "="*80)
    print("🟡 分析混合数据集")
    print("="*80)
    
    for dataset_info in MIXED_DATASETS:
        result = analyze_mixed_dataset(dataset_info)
        if result:
            results['mixed_datasets'].append(result)
    
    # 保存结果
    output_file = OUTPUT_DIR / 'dataset_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存到: {output_file}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 分析摘要")
    print("="*80)
    print(f"\n未知数据集: {len(results['unknown_datasets'])} 个")
    for result in results['unknown_datasets']:
        print(f"  - {result['name']}: {result['total_images']} 张图像")
    
    print(f"\n混合数据集: {len(results['mixed_datasets'])} 个")
    for result in results['mixed_datasets']:
        print(f"  - {result['name']}: {result['total_images']} 张图像")
        print(f"    完整水果: {result['fruit_images']} 张")
        print(f"    叶子: {result['leaf_images']} 张")
        print(f"    未知: {result['unknown_images']} 张")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
