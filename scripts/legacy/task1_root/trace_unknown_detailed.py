#!/usr/bin/env python3
"""
详细追踪每张unknown图片的原始来源
通过文件哈希值匹配，记录每张图片的原始数据集路径
"""
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# 路径配置
UNKNOWN_DIRS = {
    'train': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/train/unknown',
    'val': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/val/unknown',
    'test': '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/unknown'
}

GITHUB_DB = Path('/home/yuhanlin/Database/github/database')
LOCAL_DB = Path('/home/yuhanlin/Database/local/database')

# 重点检查的数据集（根据之前的分析）
PRIORITY_DATASETS = ['plant_village', 'deepfruits', 'acfr-multifruit-2016', 'RawRipe', 'fruit-salad', 
                     'Multi-species-fruit-flower']

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

def find_all_images_in_dataset(dataset_path):
    """在数据集中查找所有图像文件，返回 {hash: path} 字典"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}
    image_map = {}  # {hash: {'path': ..., 'subdir': ...}}
    
    print(f"    扫描数据集: {dataset_path.name}...")
    
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
                    # 验证图像
                    with Image.open(filepath) as img:
                        img.verify()
                    
                    # 计算哈希值
                    file_hash = get_file_hash(filepath)
                    if file_hash:
                        image_map[file_hash] = {
                            'path': filepath,
                            'subdirectory': subdir,
                            'relative_path': os.path.relpath(filepath, dataset_path)
                        }
                except:
                    continue
    
    return image_map

def collect_unknown_images():
    """收集所有unknown图片的信息"""
    unknown_images = {}
    
    for split, unknown_dir in UNKNOWN_DIRS.items():
        if not os.path.exists(unknown_dir):
            continue
        
        print(f"收集 {split} 集的unknown图片...")
        for filename in os.listdir(unknown_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                filepath = os.path.join(unknown_dir, filename)
                file_hash = get_file_hash(filepath)
                if file_hash:
                    unknown_images[file_hash] = {
                        'filename': filename,
                        'path': filepath,
                        'split': split
                    }
    
    return unknown_images

def trace_unknown_sources():
    """追踪unknown图片的原始来源"""
    print("=" * 70)
    print("详细追踪Unknown图片的原始来源")
    print("=" * 70)
    
    # 1. 收集所有unknown图片
    print("\n1. 收集unknown图片...")
    unknown_images = collect_unknown_images()
    print(f"   找到 {len(unknown_images)} 张unknown图片")
    
    # 2. 创建结果字典
    results = {}
    matched_count = 0
    
    # 3. 检查GitHub数据库
    print("\n2. 在GitHub数据库中查找匹配...")
    if GITHUB_DB.exists():
        # 先检查优先级数据集
        for dataset_name in PRIORITY_DATASETS:
            dataset_path = GITHUB_DB / dataset_name
            if not dataset_path.exists() or not dataset_path.is_dir():
                continue
            
            print(f"\n  检查数据集: {dataset_name}")
            image_map = find_all_images_in_dataset(dataset_path)
            print(f"    找到 {len(image_map)} 张图片")
            
            # 匹配unknown图片
            for img_hash, img_info in image_map.items():
                if img_hash in unknown_images:
                    unknown_info = unknown_images[img_hash]
                    results[unknown_info['filename']] = {
                        'original_path': img_info['path'],
                        'original_relative_path': img_info['relative_path'],
                        'dataset': dataset_name,
                        'dataset_source': 'github',
                        'subdirectory': img_info['subdirectory'],
                        'split': unknown_info['split'],
                        'matched': True
                    }
                    matched_count += 1
                    if matched_count % 50 == 0:
                        print(f"    已匹配 {matched_count} 张图片...")
        
        # 检查其他数据集
        print(f"\n  检查其他数据集...")
        for dataset_name in sorted(os.listdir(GITHUB_DB)):
            if dataset_name in PRIORITY_DATASETS or dataset_name == 'scripts':
                continue
            
            dataset_path = GITHUB_DB / dataset_name
            if not dataset_path.is_dir():
                continue
            
            print(f"  检查数据集: {dataset_name}")
            image_map = find_all_images_in_dataset(dataset_path)
            
            for img_hash, img_info in image_map.items():
                if img_hash in unknown_images and unknown_images[img_hash]['filename'] not in results:
                    unknown_info = unknown_images[img_hash]
                    results[unknown_info['filename']] = {
                        'original_path': img_info['path'],
                        'original_relative_path': img_info['relative_path'],
                        'dataset': dataset_name,
                        'dataset_source': 'github',
                        'subdirectory': img_info['subdirectory'],
                        'split': unknown_info['split'],
                        'matched': True
                    }
                    matched_count += 1
                    if matched_count % 50 == 0:
                        print(f"    已匹配 {matched_count} 张图片...")
    
    # 4. 检查本地数据库
    print("\n3. 在本地数据库中查找匹配...")
    if LOCAL_DB.exists():
        for dataset_name in sorted(os.listdir(LOCAL_DB)):
            dataset_path = LOCAL_DB / dataset_name
            if not dataset_path.is_dir():
                continue
            
            # 跳过已匹配的图片
            remaining_unknown = {h: info for h, info in unknown_images.items() 
                                if info['filename'] not in results}
            if not remaining_unknown:
                break
            
            print(f"  检查数据集: {dataset_name}")
            image_map = find_all_images_in_dataset(dataset_path)
            
            for img_hash, img_info in image_map.items():
                if img_hash in remaining_unknown:
                    unknown_info = remaining_unknown[img_hash]
                    results[unknown_info['filename']] = {
                        'original_path': img_info['path'],
                        'original_relative_path': img_info['relative_path'],
                        'dataset': dataset_name,
                        'dataset_source': 'local',
                        'subdirectory': img_info['subdirectory'],
                        'split': unknown_info['split'],
                        'matched': True
                    }
                    matched_count += 1
                    if matched_count % 50 == 0:
                        print(f"    已匹配 {matched_count} 张图片...")
    
    # 5. 标记未匹配的图片
    for img_hash, unknown_info in unknown_images.items():
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
    
    # 按split分组保存
    output_data = {
        'summary': {
            'total_unknown_images': len(unknown_images),
            'matched_images': matched_count,
            'unmatched_images': len(unknown_images) - matched_count,
            'match_rate': f"{matched_count / len(unknown_images) * 100:.2f}%" if unknown_images else "0%"
        },
        'images': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 追踪完成！")
    print(f"   总unknown图片数: {len(unknown_images)}")
    print(f"   已匹配图片数: {matched_count}")
    print(f"   未匹配图片数: {len(unknown_images) - matched_count}")
    print(f"   匹配率: {matched_count / len(unknown_images) * 100:.2f}%" if unknown_images else "0%")
    print(f"\n结果已保存到: {output_file}")
    
    # 打印统计信息
    if matched_count > 0:
        print("\n数据集来源统计:")
        dataset_stats = defaultdict(int)
        for img_info in results.values():
            if img_info.get('matched') and img_info.get('dataset'):
                dataset_stats[img_info['dataset']] += 1
        
        sorted_datasets = sorted(dataset_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (dataset, count) in enumerate(sorted_datasets, 1):
            print(f"  {i:2d}. {dataset:40s}: {count:5d} 张")

if __name__ == '__main__':
    try:
        from tqdm import tqdm
    except ImportError:
        print("警告: tqdm未安装，将不使用进度条")
        # 创建一个空的tqdm函数
        def tqdm(iterable, **kwargs):
            return iterable
    
    trace_unknown_sources()
