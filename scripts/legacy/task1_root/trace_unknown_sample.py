#!/usr/bin/env python3
"""
追踪unknown图像的原始数据集来源（样本版本）
只处理前100张图像作为快速测试
"""

import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

# 数据集路径
GITHUB_DB = Path("/home/yuhanlin/Database/github/database")
LOCAL_DB = Path("/home/yuhanlin/Database/local/database")
UNKNOWN_DIRS = [
    Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown"),
    Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown"),
    Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown")
]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}

def get_image_signature(filepath):
    """获取图像的签名（大小 + 尺寸）"""
    try:
        file_size = filepath.stat().st_size
        with Image.open(filepath) as img:
            width, height = img.size
            return (file_size, width, height)
    except Exception:
        return None

def find_match_in_dataset(unknown_sig, dataset_path):
    """在数据集中查找匹配的文件"""
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix in IMAGE_EXTENSIONS:
                filepath = Path(root) / file
                try:
                    sig = get_image_signature(filepath)
                    if sig == unknown_sig:
                        return {
                            'path': str(filepath),
                            'relative_path': str(filepath.relative_to(dataset_path)),
                            'dataset': dataset_path.name
                        }
                except Exception:
                    continue
    return None

def trace_sample(sample_size=100):
    """追踪样本图像"""
    print("="*80)
    print(f"🔍 追踪Unknown图像来源（样本测试：前{sample_size}张）")
    print("="*80)
    
    # 收集unknown图像
    unknown_images = []
    for unknown_dir in UNKNOWN_DIRS:
        if unknown_dir.exists():
            split_name = unknown_dir.parent.name
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(unknown_dir.glob(f"*{ext}"))
            for img in images[:sample_size//3]:  # 每个split取一部分
                unknown_images.append((img, split_name))
    
    unknown_images = unknown_images[:sample_size]
    print(f"\n📦 处理 {len(unknown_images)} 张样本图像...\n")
    
    results = {
        'matched': [],
        'unmatched': [],
        'dataset_distribution': defaultdict(int)
    }
    
    # 处理每张图像
    for i, (img_path, split_name) in enumerate(unknown_images, 1):
        print(f"  [{i}/{len(unknown_images)}] {img_path.name}...", end=' ', flush=True)
        
        signature = get_image_signature(img_path)
        if not signature:
            print("❌ 无法读取")
            continue
        
        # 在GitHub数据库中查找
        match = None
        if GITHUB_DB.exists():
            for dataset_name in sorted(os.listdir(GITHUB_DB)):
                dataset_path = GITHUB_DB / dataset_name
                if dataset_path.is_dir() and dataset_name != 'scripts':
                    match = find_match_in_dataset(signature, dataset_path)
                    if match:
                        break
        
        # 在本地数据库中查找
        if not match and LOCAL_DB.exists():
            for dataset_name in sorted(os.listdir(LOCAL_DB)):
                dataset_path = LOCAL_DB / dataset_name
                if dataset_path.is_dir():
                    match = find_match_in_dataset(signature, dataset_path)
                    if match:
                        break
        
        if match:
            dataset_name = match['dataset']
            results['dataset_distribution'][dataset_name] += 1
            results['matched'].append({
                'unknown_path': str(img_path),
                'split': split_name,
                'source': match
            })
            print(f"✅ 来自: {dataset_name}")
        else:
            results['unmatched'].append({
                'unknown_path': str(img_path),
                'split': split_name
            })
            print("❌ 未找到匹配")
    
    # 打印结果
    print("\n" + "="*80)
    print("📊 样本分析结果")
    print("="*80)
    print(f"\n总样本数: {len(unknown_images)}")
    print(f"成功匹配: {len(results['matched'])} ({len(results['matched'])/len(unknown_images)*100:.1f}%)")
    print(f"未匹配: {len(results['unmatched'])} ({len(results['unmatched'])/len(unknown_images)*100:.1f}%)")
    
    if results['dataset_distribution']:
        print(f"\n📈 数据集来源分布（样本）:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)
        for dataset, count in sorted_datasets:
            percentage = (count / len(results['matched']) * 100) if results['matched'] else 0
            print(f"  {dataset}: {count} 张 ({percentage:.1f}%)")
    
    # 保存结果
    output_file = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/unknown_sources_sample.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 样本结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    trace_sample(sample_size=50)
