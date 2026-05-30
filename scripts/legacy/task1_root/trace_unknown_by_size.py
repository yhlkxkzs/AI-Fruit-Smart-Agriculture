#!/usr/bin/env python3
"""
通过文件大小快速追踪unknown图像来源
只使用文件大小进行匹配（快速但可能有误匹配）
"""

import os
from pathlib import Path
from collections import defaultdict
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

def build_size_index(dataset_paths, max_files_per_dataset=10000):
    """建立基于文件大小的索引（只索引每个数据集的前N个文件以加快速度）"""
    size_index = defaultdict(list)
    total_files = 0
    
    for base_path in dataset_paths:
        if not base_path.exists():
            continue
        
        print(f"  索引 {base_path.name}...", end=' ', flush=True)
        count = 0
        
        for dataset_name in sorted(os.listdir(base_path)):
            dataset_path = base_path / dataset_name
            if not dataset_path.is_dir() or dataset_name == 'scripts':
                continue
            
            dataset_files = 0
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if Path(file).suffix in IMAGE_EXTENSIONS:
                        filepath = Path(root) / file
                        try:
                            file_size = filepath.stat().st_size
                            size_index[file_size].append({
                                'path': str(filepath),
                                'relative_path': str(filepath.relative_to(dataset_path)),
                                'dataset': dataset_name,
                                'source_db': base_path.name
                            })
                            dataset_files += 1
                            total_files += 1
                            
                            # 限制每个数据集的文件数
                            if dataset_files >= max_files_per_dataset:
                                break
                        except Exception:
                            continue
                
                if dataset_files >= max_files_per_dataset:
                    break
        
        print(f"✅")
    
    print(f"\n✅ 总共索引了 {total_files} 个文件，{len(size_index)} 个唯一大小")
    return size_index

def trace_unknown_by_size():
    """通过文件大小追踪unknown图像"""
    print("="*80)
    print("🔍 通过文件大小快速追踪Unknown图像来源")
    print("="*80)
    
    # 1. 收集unknown图像
    print("\n📦 步骤1: 收集unknown图像...")
    unknown_images = []
    for unknown_dir in UNKNOWN_DIRS:
        if unknown_dir.exists():
            split_name = unknown_dir.parent.name
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(unknown_dir.glob(f"*{ext}"))
            for img in images:
                unknown_images.append((img, split_name))
    
    print(f"✅ 找到 {len(unknown_images)} 张unknown图像")
    
    # 2. 建立原始数据集的大小索引（限制每个数据集的文件数以加快速度）
    print("\n📦 步骤2: 建立原始数据集索引（这可能需要几分钟）...")
    dataset_paths = []
    if GITHUB_DB.exists():
        dataset_paths.append(GITHUB_DB)
    if LOCAL_DB.exists():
        dataset_paths.append(LOCAL_DB)
    
    size_index = build_size_index(dataset_paths, max_files_per_dataset=5000)
    
    # 3. 匹配unknown图像
    print("\n📦 步骤3: 匹配unknown图像...")
    results = {
        'matched': [],
        'unmatched': [],
        'dataset_distribution': defaultdict(int)
    }
    
    for i, (img_path, split_name) in enumerate(unknown_images, 1):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(unknown_images)} ({i/len(unknown_images)*100:.1f}%)", end='\r', flush=True)
        
        try:
            file_size = img_path.stat().st_size
            if file_size in size_index:
                # 找到匹配（可能有多个，取第一个）
                matches = size_index[file_size]
                match = matches[0]
                dataset_name = match['dataset']
                results['dataset_distribution'][dataset_name] += 1
                results['matched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name,
                    'source': match
                })
            else:
                results['unmatched'].append({
                    'unknown_path': str(img_path),
                    'split': split_name
                })
        except Exception:
            results['unmatched'].append({
                'unknown_path': str(img_path),
                'split': split_name,
                'reason': '无法读取文件'
            })
    
    print(f"\n  完成: {len(unknown_images)} 张")
    
    # 4. 打印结果
    print("\n" + "="*80)
    print("📊 匹配结果统计")
    print("="*80)
    print(f"\n总unknown图像数: {len(unknown_images)}")
    print(f"成功匹配: {len(results['matched'])} ({len(results['matched'])/len(unknown_images)*100:.1f}%)")
    print(f"未匹配: {len(results['unmatched'])} ({len(results['unmatched'])/len(unknown_images)*100:.1f}%)")
    
    if results['dataset_distribution']:
        print(f"\n📈 数据集来源分布:")
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)
        for dataset, count in sorted_datasets:
            percentage = (count / len(results['matched']) * 100) if results['matched'] else 0
            print(f"  {dataset}: {count} 张 ({percentage:.1f}%)")
    
    # 5. 保存结果
    output_file = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/unknown_sources_by_size.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_unknown': len(unknown_images),
                'matched': len(results['matched']),
                'unmatched': len(results['unmatched']),
                'match_rate': len(results['matched'])/len(unknown_images)*100 if unknown_images else 0,
                'note': '基于文件大小匹配，可能有误匹配'
            },
            'dataset_distribution': dict(results['dataset_distribution']),
            'matched_samples': results['matched'][:100]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return results

if __name__ == '__main__':
    results = trace_unknown_by_size()
    
    if results['dataset_distribution']:
        print("\n" + "="*80)
        print("📝 Top 15 数据集来源")
        print("="*80)
        sorted_datasets = sorted(results['dataset_distribution'].items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            print(f"  {i:2d}. {dataset:40s}: {count:5d} 张")
