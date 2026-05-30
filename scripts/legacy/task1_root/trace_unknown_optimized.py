#!/usr/bin/env python3
"""
优化版：详细追踪每张unknown图片的原始来源
使用文件大小预筛选，然后计算哈希值匹配
"""
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

# 路径配置
UNKNOWN_DIRS = {
    'train': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    'val': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    'test': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
}

GITHUB_DB = Path('/home/yuhanlin/Database/github/database')
LOCAL_DB = Path('/home/yuhanlin/Database/local/database')

# 重点检查的数据集
PRIORITY_DATASETS = ['plant_village', 'deepfruits', 'acfr-multifruit-2016', 'RawRipe', 'fruit-salad', 
                     'Multi-species-fruit-flower', 'acfr-multifruit-2016']

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

def collect_unknown_images():
    """收集所有unknown图片的信息（按文件大小分组）"""
    unknown_by_size = defaultdict(list)  # {size: [{'filename': ..., 'path': ..., 'hash': ...}, ...]}
    
    print("=" * 70)
    print("收集Unknown图片信息")
    print("=" * 70)
    
    for split, unknown_dir in UNKNOWN_DIRS.items():
        if not os.path.exists(unknown_dir):
            continue
        
        print(f"\n处理 {split} 集...")
        count = 0
        for filename in os.listdir(unknown_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                filepath = os.path.join(unknown_dir, filename)
                file_size = get_file_size(filepath)
                if file_size:
                    unknown_by_size[file_size].append({
                        'filename': filename,
                        'path': filepath,
                        'split': split,
                        'size': file_size
                    })
                    count += 1
        print(f"  找到 {count} 张图片")
    
    total = sum(len(imgs) for imgs in unknown_by_size.values())
    print(f"\n总计: {total} 张unknown图片，{len(unknown_by_size)} 种不同大小")
    
    return unknown_by_size

def find_images_in_dataset_by_size(dataset_path, size_filter=None):
    """在数据集中查找图像文件，按大小分组"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
    images_by_size = defaultdict(list)  # {size: [{'path': ..., 'subdir': ...}, ...]}
    
    for root, dirs, files in os.walk(dataset_path):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'scripts', 'labels', 'annotations', 'data', 'docs', 'sets', 'origin']]
        
        # 获取相对路径中的子目录名
        rel_path = os.path.relpath(root, dataset_path)
        if rel_path == '.':
            subdir = '根目录'
        else:
            subdir = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
        
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                filepath = os.path.join(root, file)
                try:
                    # 快速检查：只验证文件大小
                    file_size = get_file_size(filepath)
                    if file_size and (size_filter is None or file_size in size_filter):
                        # 验证图像
                        try:
                            with Image.open(filepath) as img:
                                img.verify()
                            images_by_size[file_size].append({
                                    'path': filepath,
                                    'subdirectory': subdir,
                                    'relative_path': os.path.relpath(filepath, dataset_path)
                                })
                        except:
                            continue
                except:
                    continue
    
    return images_by_size

def trace_unknown_sources():
    """追踪unknown图片的原始来源"""
    # 1. 收集unknown图片（按大小分组）
    unknown_by_size = collect_unknown_images()
    
    # 2. 创建结果字典
    results = {}
    matched_count = 0
    total_unknown = sum(len(imgs) for imgs in unknown_by_size.values())
    
    # 3. 检查GitHub数据库（优先检查重点数据集）
    print("\n" + "=" * 70)
    print("在GitHub数据库中查找匹配")
    print("=" * 70)
    
    if GITHUB_DB.exists():
        # 先检查优先级数据集
        for dataset_name in PRIORITY_DATASETS:
            dataset_path = GITHUB_DB / dataset_name
            if not dataset_path.exists() or not dataset_path.is_dir():
                continue
            
            print(f"\n检查数据集: {dataset_name}")
            # 只查找与unknown图片大小匹配的图像
            size_filter = set(unknown_by_size.keys())
            candidate_images = find_images_in_dataset_by_size(dataset_path, size_filter)
            
            print(f"  找到 {sum(len(imgs) for imgs in candidate_images.values())} 张候选图片")
            
            # 对相同大小的图片计算哈希值并匹配
            for size, candidate_list in candidate_images.items():
                if size not in unknown_by_size:
                    continue
                
                print(f"  处理大小 {size} 字节的图片 ({len(candidate_list)} 张候选, {len(unknown_by_size[size])} 张unknown)...")
                
                # 计算候选图片的哈希值
                candidate_hashes = {}
                for candidate in candidate_list:
                    img_hash = get_file_hash(candidate['path'])
                    if img_hash:
                        candidate_hashes[img_hash] = candidate
                
                # 计算unknown图片的哈希值并匹配
                for unknown_info in unknown_by_size[size]:
                    if unknown_info['filename'] in results:
                        continue  # 已匹配
                    
                    unknown_hash = get_file_hash(unknown_info['path'])
                    if unknown_hash and unknown_hash in candidate_hashes:
                        candidate = candidate_hashes[unknown_hash]
                        results[unknown_info['filename']] = {
                            'original_path': candidate['path'],
                            'original_relative_path': candidate['relative_path'],
                            'dataset': dataset_name,
                            'dataset_source': 'github',
                            'subdirectory': candidate['subdirectory'],
                            'split': unknown_info['split'],
                            'matched': True
                        }
                        matched_count += 1
                        if matched_count % 10 == 0:
                            print(f"    已匹配 {matched_count}/{total_unknown} 张图片...")
        
        # 检查其他数据集
        print(f"\n检查其他数据集...")
        remaining_sizes = set()
        for size, imgs in unknown_by_size.items():
            for img in imgs:
                if img['filename'] not in results:
                    remaining_sizes.add(size)
        
        if remaining_sizes:
            for dataset_name in sorted(os.listdir(GITHUB_DB)):
                if dataset_name in PRIORITY_DATASETS or dataset_name == 'scripts':
                    continue
                
                dataset_path = GITHUB_DB / dataset_name
                if not dataset_path.is_dir():
                    continue
                
                print(f"  检查数据集: {dataset_name}")
                candidate_images = find_images_in_dataset_by_size(dataset_path, remaining_sizes)
                
                for size, candidate_list in candidate_images.items():
                    if size not in unknown_by_size:
                        continue
                    
                    candidate_hashes = {}
                    for candidate in candidate_list:
                        img_hash = get_file_hash(candidate['path'])
                        if img_hash:
                            candidate_hashes[img_hash] = candidate
                    
                    for unknown_info in unknown_by_size[size]:
                        if unknown_info['filename'] in results:
                            continue
                        
                        unknown_hash = get_file_hash(unknown_info['path'])
                        if unknown_hash and unknown_hash in candidate_hashes:
                            candidate = candidate_hashes[unknown_hash]
                            results[unknown_info['filename']] = {
                                'original_path': candidate['path'],
                                'original_relative_path': candidate['relative_path'],
                                'dataset': dataset_name,
                                'dataset_source': 'github',
                                'subdirectory': candidate['subdirectory'],
                                'split': unknown_info['split'],
                                'matched': True
                            }
                            matched_count += 1
                            if matched_count % 10 == 0:
                                print(f"    已匹配 {matched_count}/{total_unknown} 张图片...")
    
    # 4. 检查本地数据库
    print("\n" + "=" * 70)
    print("在本地数据库中查找匹配")
    print("=" * 70)
    
    if LOCAL_DB.exists():
        remaining_sizes = set()
        for size, imgs in unknown_by_size.items():
            for img in imgs:
                if img['filename'] not in results:
                    remaining_sizes.add(size)
        
        if remaining_sizes:
            for dataset_name in sorted(os.listdir(LOCAL_DB)):
                dataset_path = LOCAL_DB / dataset_name
                if not dataset_path.is_dir():
                    continue
                
                print(f"检查数据集: {dataset_name}")
                candidate_images = find_images_in_dataset_by_size(dataset_path, remaining_sizes)
                
                for size, candidate_list in candidate_images.items():
                    if size not in unknown_by_size:
                        continue
                    
                    candidate_hashes = {}
                    for candidate in candidate_list:
                        img_hash = get_file_hash(candidate['path'])
                        if img_hash:
                            candidate_hashes[img_hash] = candidate
                    
                    for unknown_info in unknown_by_size[size]:
                        if unknown_info['filename'] in results:
                            continue
                        
                        unknown_hash = get_file_hash(unknown_info['path'])
                        if unknown_hash and unknown_hash in candidate_hashes:
                            candidate = candidate_hashes[unknown_hash]
                            results[unknown_info['filename']] = {
                                'original_path': candidate['path'],
                                'original_relative_path': candidate['relative_path'],
                                'dataset': dataset_name,
                                'dataset_source': 'local',
                                'subdirectory': candidate['subdirectory'],
                                'split': unknown_info['split'],
                                'matched': True
                            }
                            matched_count += 1
                            if matched_count % 10 == 0:
                                print(f"  已匹配 {matched_count}/{total_unknown} 张图片...")
    
    # 5. 标记未匹配的图片
    for size, imgs in unknown_by_size.items():
        for unknown_info in imgs:
            if unknown_info['filename'] not in results:
                results[unknown_info['filename']] = {
                    'original_path': None,
                    'original_relative_path': None,
                    'dataset': None,
                    'dataset_source': None,
                    'subdirectory': None,
                    'split': unknown_info['split'],
                    'matched': False
                }
    
    # 6. 保存结果
    output_file = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/unknown_images_trace.json')
    
    output_data = {
        'summary': {
            'total_unknown_images': total_unknown,
            'matched_images': matched_count,
            'unmatched_images': total_unknown - matched_count,
            'match_rate': f"{matched_count / total_unknown * 100:.2f}%" if total_unknown > 0 else "0%"
        },
        'images': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("追踪完成！")
    print("=" * 70)
    print(f"总unknown图片数: {total_unknown}")
    print(f"已匹配图片数: {matched_count}")
    print(f"未匹配图片数: {total_unknown - matched_count}")
    print(f"匹配率: {matched_count / total_unknown * 100:.2f}%" if total_unknown > 0 else "0%")
    print(f"\n结果已保存到: {output_file}")
    
    # 打印数据集统计
    if matched_count > 0:
        print("\n数据集来源统计:")
        dataset_stats = defaultdict(int)
        for img_info in results.values():
            if img_info.get('matched') and img_info.get('dataset'):
                dataset_stats[img_info['dataset']] += 1
        
        sorted_datasets = sorted(dataset_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            print(f"  {i:2d}. {dataset:40s}: {count:5d} 张")

if __name__ == '__main__':
    trace_unknown_sources()
