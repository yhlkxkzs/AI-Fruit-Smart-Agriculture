#!/usr/bin/env python3
"""
快速追踪unknown图像的原始数据集来源
使用文件大小和图像尺寸进行匹配（比MD5哈希快得多）
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
    """获取图像的签名（大小 + 尺寸）用于快速匹配"""
    try:
        file_size = filepath.stat().st_size
        with Image.open(filepath) as img:
            width, height = img.size
            return (file_size, width, height)
    except Exception:
        return None

def scan_dataset_fast(dataset_path, max_depth=5, sample_limit=None):
    """快速扫描数据集，建立基于文件大小和图像尺寸的索引"""
    signature_index = defaultdict(list)
    count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        depth = root.replace(str(dataset_path), '').count(os.sep)
        if depth > max_depth:
            continue
        
        for file in files:
            if Path(file).suffix in IMAGE_EXTENSIONS:
                filepath = Path(root) / file
                try:
                    signature = get_image_signature(filepath)
                    if signature:
                        signature_index[signature].append({
                            'path': str(filepath),
                            'relative_path': str(filepath.relative_to(dataset_path)),
                            'dataset': dataset_path.name
                        })
                        count += 1
                        if sample_limit and count >= sample_limit:
                            return signature_index, count
                except Exception:
                    continue
    
    return signature_index, count

def find_unknown_sources_fast():
    """快速查找unknown图像的原始来源"""
    print("="*80)
    print("🔍 快速追踪Unknown图像的原始数据集来源")
    print("="*80)
    
    # 1. 扫描所有原始数据集
    print("\n📦 步骤1: 快速扫描原始数据集（建立索引）...")
    all_datasets_index = defaultdict(list)
    total_scanned = 0
    
    # 扫描GitHub数据库
    if GITHUB_DB.exists():
        print(f"\n扫描GitHub数据库: {GITHUB_DB}")
        for dataset_name in sorted(os.listdir(GITHUB_DB)):
            dataset_path = GITHUB_DB / dataset_name
            if dataset_path.is_dir() and dataset_name != 'scripts':
                print(f"  处理 {dataset_name}...", end=' ', flush=True)
                index, count = scan_dataset_fast(dataset_path)
                for sig, files in index.items():
                    all_datasets_index[sig].extend(files)
                total_scanned += count
                print(f"✅ {count} 张图像")
    
    # 扫描本地数据库
    if LOCAL_DB.exists():
        print(f"\n扫描本地数据库: {LOCAL_DB}")
        for dataset_name in sorted(os.listdir(LOCAL_DB)):
            dataset_path = LOCAL_DB / dataset_name
            if dataset_path.is_dir():
                print(f"  处理 {dataset_name}...", end=' ', flush=True)
                index, count = scan_dataset_fast(dataset_path)
                for sig, files in index.items():
                    all_datasets_index[sig].extend(files)
                total_scanned += count
                print(f"✅ {count} 张图像")
    
    print(f"\n✅ 总共扫描了 {total_scanned} 张图像，建立了 {len(all_datasets_index)} 个唯一签名")
    
    # 2. 扫描unknown图像并匹配
    print("\n📦 步骤2: 扫描并匹配unknown图像...")
    
    results = {
        'matched': [],
        'unmatched': [],
        'dataset_distribution': defaultdict(int)
    }
    
    total_unknown = 0
    matched_count = 0
    
    for unknown_dir in UNKNOWN_DIRS:
        if not unknown_dir.exists():
            continue
        
        split_name = unknown_dir.parent.name
        print(f"\n处理 {split_name}/unknown...")
        
        # 获取所有图像文件
        unknown_images = []
        for ext in IMAGE_EXTENSIONS:
            unknown_images.extend(unknown_dir.glob(f"*{ext}"))
            unknown_images.extend(unknown_dir.glob(f"*{ext.upper()}"))
        
        print(f"  找到 {len(unknown_images)} 张图像")
        total_unknown += len(unknown_images)
        
        # 匹配每张图像
        for i, img_path in enumerate(unknown_images):
            if (i + 1) % 100 == 0:
                print(f"    处理进度: {i+1}/{len(unknown_images)}", end='\r', flush=True)
            
            signature = get_image_signature(img_path)
            
            if signature and signature in all_datasets_index:
                # 找到匹配
                sources = all_datasets_index[signature]
                matched_count += 1
                
                # 统计数据集分布（取第一个匹配的数据集）
                if sources:
                    dataset_name = sources[0]['dataset']
                    results['dataset_distribution'][dataset_name] += 1
                
                results['matched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name,
                    'sources': sources[:3]  # 只保存前3个匹配
                })
            else:
                # 未找到匹配
                results['unmatched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name
                })
        
        print(f"    处理完成: {len(unknown_images)} 张")
    
    # 3. 打印结果
    print("\n" + "="*80)
    print("📊 匹配结果统计")
    print("="*80)
    print(f"\n总unknown图像数: {total_unknown}")
    print(f"成功匹配: {matched_count} ({matched_count/total_unknown*100:.1f}%)")
    print(f"未匹配: {len(results['unmatched'])} ({len(results['unmatched'])/total_unknown*100:.1f}%)")
    
    if results['dataset_distribution']:
        print(f"\n📈 数据集来源分布:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)
        for dataset, count in sorted_datasets:
            percentage = (count / matched_count * 100) if matched_count > 0 else 0
            print(f"  {dataset}: {count} 张 ({percentage:.1f}%)")
    
    # 4. 保存详细结果
    output_file = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/unknown_sources_trace.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_unknown': total_unknown,
                'matched': matched_count,
                'unmatched': len(results['unmatched']),
                'match_rate': matched_count/total_unknown*100 if total_unknown > 0 else 0,
                'total_scanned': total_scanned
            },
            'dataset_distribution': dict(results['dataset_distribution']),
            'matched_samples': results['matched'][:100],  # 保存前100个匹配样本
            'unmatched_count': len(results['unmatched'])
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    results = find_unknown_sources_fast()
    
    # 生成简要报告
    print("\n" + "="*80)
    print("📝 简要报告")
    print("="*80)
    
    if results['dataset_distribution']:
        print("\n🎯 Unknown图像主要来源数据集（Top 15）:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            percentage = (count / results['matched'].__len__() * 100) if results['matched'] else 0
            print(f"  {i:2d}. {dataset:40s}: {count:5d} 张 ({percentage:5.1f}%)")
