#!/usr/bin/env python3
"""
智能追踪unknown图像的原始数据集来源
只扫描unknown图像，然后按需在原始数据集中查找匹配
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

def find_matching_files_in_dataset(unknown_sig, dataset_path, max_matches=5):
    """在指定数据集中查找匹配的文件"""
    matches = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix in IMAGE_EXTENSIONS:
                filepath = Path(root) / file
                try:
                    sig = get_image_signature(filepath)
                    if sig == unknown_sig:
                        matches.append({
                            'path': str(filepath),
                            'relative_path': str(filepath.relative_to(dataset_path)),
                            'dataset': dataset_path.name
                        })
                        if len(matches) >= max_matches:
                            return matches
                except Exception:
                    continue
    
    return matches

def trace_unknown_sources_smart():
    """智能追踪unknown图像来源"""
    print("="*80)
    print("🔍 智能追踪Unknown图像的原始数据集来源")
    print("="*80)
    
    # 1. 收集所有unknown图像
    print("\n📦 步骤1: 收集unknown图像...")
    unknown_images = []
    
    for unknown_dir in UNKNOWN_DIRS:
        if not unknown_dir.exists():
            continue
        
        split_name = unknown_dir.parent.name
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(unknown_dir.glob(f"*{ext}"))
            images.extend(unknown_dir.glob(f"*{ext.upper()}"))
        
        print(f"  {split_name}/unknown: {len(images)} 张")
        for img in images:
            unknown_images.append((img, split_name))
    
    total_unknown = len(unknown_images)
    print(f"\n✅ 总共找到 {total_unknown} 张unknown图像")
    
    # 2. 对每张unknown图像，在原始数据集中查找匹配
    print("\n📦 步骤2: 在原始数据集中查找匹配（这可能需要一些时间）...")
    print("  建议：如果数据量大，可以按Ctrl+C中断，脚本会保存已找到的结果")
    
    results = {
        'matched': [],
        'unmatched': [],
        'dataset_distribution': defaultdict(int),
        'processed': 0
    }
    
    try:
        for i, (img_path, split_name) in enumerate(unknown_images):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{total_unknown} ({i+1/total_unknown*100:.1f}%)", end='\r', flush=True)
            
            signature = get_image_signature(img_path)
            if not signature:
                results['unmatched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name,
                    'reason': '无法读取图像'
                })
                continue
            
            # 在GitHub数据库中查找
            matches = []
            if GITHUB_DB.exists():
                for dataset_name in os.listdir(GITHUB_DB):
                    dataset_path = GITHUB_DB / dataset_name
                    if dataset_path.is_dir() and dataset_name != 'scripts':
                        found = find_matching_files_in_dataset(signature, dataset_path, max_matches=1)
                        if found:
                            matches.extend(found)
                            break  # 找到就停止
            
            # 在本地数据库中查找
            if not matches and LOCAL_DB.exists():
                for dataset_name in os.listdir(LOCAL_DB):
                    dataset_path = LOCAL_DB / dataset_name
                    if dataset_path.is_dir():
                        found = find_matching_files_in_dataset(signature, dataset_path, max_matches=1)
                        if found:
                            matches.extend(found)
                            break  # 找到就停止
            
            if matches:
                dataset_name = matches[0]['dataset']
                results['dataset_distribution'][dataset_name] += 1
                results['matched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name,
                    'sources': matches
                })
            else:
                results['unmatched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name
                })
            
            results['processed'] = i + 1
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断，已处理 {results['processed']}/{total_unknown} 张图像")
    
    # 3. 打印结果
    print("\n" + "="*80)
    print("📊 匹配结果统计")
    print("="*80)
    print(f"\n已处理图像数: {results['processed']}")
    print(f"成功匹配: {len(results['matched'])} ({len(results['matched'])/results['processed']*100:.1f}%)" if results['processed'] > 0 else "0")
    print(f"未匹配: {len(results['unmatched'])} ({len(results['unmatched'])/results['processed']*100:.1f}%)" if results['processed'] > 0 else "0")
    
    if results['dataset_distribution']:
        print(f"\n📈 数据集来源分布:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)
        for dataset, count in sorted_datasets:
            percentage = (count / len(results['matched']) * 100) if results['matched'] else 0
            print(f"  {dataset}: {count} 张 ({percentage:.1f}%)")
    
    # 4. 保存结果
    output_file = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/unknown_sources_trace.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_unknown': total_unknown,
                'processed': results['processed'],
                'matched': len(results['matched']),
                'unmatched': len(results['unmatched']),
                'match_rate': len(results['matched'])/results['processed']*100 if results['processed'] > 0 else 0
            },
            'dataset_distribution': dict(results['dataset_distribution']),
            'matched_samples': results['matched'][:50],
            'unmatched_samples': results['unmatched'][:50]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    results = trace_unknown_sources_smart()
    
    if results['dataset_distribution']:
        print("\n" + "="*80)
        print("📝 Top 10 数据集来源")
        print("="*80)
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            print(f"  {i:2d}. {dataset:40s}: {count:5d} 张")
