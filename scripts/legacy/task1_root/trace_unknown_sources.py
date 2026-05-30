#!/usr/bin/env python3
"""
追踪unknown图像的原始数据集来源
通过文件特征（大小、时间戳、图像特征）匹配原始数据集
"""

import os
import hashlib
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

def get_file_hash(filepath, chunk_size=8192):
    """计算文件的MD5哈希值"""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None

def get_image_info(filepath):
    """获取图像的基本信息"""
    try:
        with Image.open(filepath) as img:
            return {
                'width': img.size[0],
                'height': img.size[1],
                'mode': img.mode,
                'format': img.format
            }
    except Exception:
        return None

def scan_dataset_for_images(dataset_path, max_depth=5):
    """扫描数据集中的所有图像文件，建立索引"""
    image_index = {}
    
    print(f"  扫描数据集: {dataset_path.name}")
    
    for root, dirs, files in os.walk(dataset_path):
        depth = root.replace(str(dataset_path), '').count(os.sep)
        if depth > max_depth:
            continue
        
        for file in files:
            if Path(file).suffix in IMAGE_EXTENSIONS:
                filepath = Path(root) / file
                try:
                    # 计算文件大小和哈希
                    file_size = filepath.stat().st_size
                    file_hash = get_file_hash(filepath)
                    
                    if file_hash:
                        # 获取图像信息
                        img_info = get_image_info(filepath)
                        
                        # 使用哈希作为主键，存储文件信息
                        if file_hash not in image_index:
                            image_index[file_hash] = []
                        
                        image_index[file_hash].append({
                            'path': str(filepath),
                            'relative_path': str(filepath.relative_to(dataset_path)),
                            'size': file_size,
                            'dataset': dataset_path.name,
                            'image_info': img_info
                        })
                except Exception as e:
                    continue
    
    return image_index

def find_unknown_sources():
    """查找unknown图像的原始来源"""
    print("="*80)
    print("🔍 追踪Unknown图像的原始数据集来源")
    print("="*80)
    
    # 1. 扫描所有原始数据集，建立图像索引
    print("\n📦 步骤1: 扫描原始数据集...")
    all_datasets_index = {}
    
    # 扫描GitHub数据库
    if GITHUB_DB.exists():
        print(f"\n扫描GitHub数据库: {GITHUB_DB}")
        for dataset_name in sorted(os.listdir(GITHUB_DB)):
            dataset_path = GITHUB_DB / dataset_name
            if dataset_path.is_dir() and dataset_name != 'scripts':
                index = scan_dataset_for_images(dataset_path)
                all_datasets_index.update(index)
                print(f"  ✅ {dataset_name}: 索引了 {len(index)} 个唯一图像")
    
    # 扫描本地数据库
    if LOCAL_DB.exists():
        print(f"\n扫描本地数据库: {LOCAL_DB}")
        for dataset_name in sorted(os.listdir(LOCAL_DB)):
            dataset_path = LOCAL_DB / dataset_name
            if dataset_path.is_dir():
                index = scan_dataset_for_images(dataset_path)
                all_datasets_index.update(index)
                print(f"  ✅ {dataset_name}: 索引了 {len(index)} 个唯一图像")
    
    print(f"\n✅ 总共索引了 {len(all_datasets_index)} 个唯一图像")
    
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
        for img_path in unknown_images:
            file_hash = get_file_hash(img_path)
            
            if file_hash and file_hash in all_datasets_index:
                # 找到匹配
                sources = all_datasets_index[file_hash]
                matched_count += 1
                
                for source in sources:
                    results['dataset_distribution'][source['dataset']] += 1
                
                results['matched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name,
                    'sources': sources
                })
            else:
                # 未找到匹配
                results['unmatched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name
                })
    
    # 3. 打印结果
    print("\n" + "="*80)
    print("📊 匹配结果统计")
    print("="*80)
    print(f"\n总unknown图像数: {total_unknown}")
    print(f"成功匹配: {matched_count} ({matched_count/total_unknown*100:.1f}%)")
    print(f"未匹配: {len(results['unmatched'])} ({len(results['unmatched'])/total_unknown*100:.1f}%)")
    
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
                'match_rate': matched_count/total_unknown*100 if total_unknown > 0 else 0
            },
            'dataset_distribution': dict(results['dataset_distribution']),
            'matched_samples': results['matched'][:50],  # 保存前50个匹配样本
            'unmatched_samples': results['unmatched'][:50]  # 保存前50个未匹配样本
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    results = find_unknown_sources()
    
    # 生成简要报告
    print("\n" + "="*80)
    print("📝 简要报告")
    print("="*80)
    
    if results['dataset_distribution']:
        print("\n🎯 Unknown图像主要来源数据集（Top 10）:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            print(f"  {i}. {dataset}: {count} 张")
