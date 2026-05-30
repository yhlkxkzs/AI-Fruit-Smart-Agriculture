#!/usr/bin/env python3
"""
修复所有图片命名问题并重新生成标注文件
1. 使用图片哈希识别所有不同的图片
2. 确保每个不同的图片都有唯一的文件名
3. 重新生成所有JSON、CSV和split文件
"""
import os
import json
import hashlib
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# 数据集路径
DATASET_DIR = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
IMAGE_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.gif', '.GIF']

def calculate_image_hash(img_path):
    """计算图片的MD5哈希（优化：使用文件大小+部分内容）"""
    try:
        stat = img_path.stat()
        # 使用文件大小和部分内容来计算哈希，加快速度
        with open(img_path, 'rb') as f:
            # 读取前8KB和最后8KB
            head = f.read(8192)
            f.seek(max(0, stat.st_size - 8192))
            tail = f.read(8192)
            content = head + tail + str(stat.st_size).encode()
            return hashlib.md5(content).hexdigest()
    except:
        return None

def get_image_info(img_path):
    """获取图片信息"""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
        size = img_path.stat().st_size
        hash_value = calculate_image_hash(img_path)
        return {
            'width': width,
            'height': height,
            'mode': mode,
            'format': format_name,
            'size': size,
            'hash': hash_value
        }
    except Exception as e:
        return None

def detect_image_type(img_path, width, height):
    """检测图片类型"""
    if width == 64 and height == 64:
        return True, 'combination', 'deepfruits'
    if width == 1024 and height == 1024 and img_path.suffix.lower() == '.png':
        return False, 'artistic', 'fruit-salad'
    if width < 200 or height < 200:
        return True, 'combination', 'deepfruits'
    return False, 'single_fruit', 'unknown'

def find_max_index_in_category(category_dir, category_name):
    """找出类别中最大的索引号"""
    max_index = -1
    pattern = re.compile(rf'^{re.escape(category_name)}_(\d+)(?:_[a-f0-9]+)?\.[^.]+$', re.IGNORECASE)
    
    for img_file in category_dir.iterdir():
        if img_file.is_file():
            match = pattern.match(img_file.name)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)
    
    return max_index

def fix_category_images(category_dir, category_name, split_name):
    """修复单个类别的所有图片命名"""
    if not category_dir.exists():
        return 0, []
    
    # 收集所有图片文件
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(list(category_dir.glob(f'*{ext}')))
    
    if not image_files:
        return 0, []
    
    # 按哈希分组，找出重复内容和需要重命名的
    hash_to_files = defaultdict(list)
    stem_to_files = defaultdict(list)
    
    for img_file in image_files:
        info = get_image_info(img_file)
        if info and info['hash']:
            hash_to_files[info['hash']].append(img_file)
        stem_to_files[img_file.stem].append(img_file)
    
    # 找出需要重命名的情况：
    # 1. 相同stem但不同哈希（相同文件名但内容不同）
    # 2. 不同stem但相同哈希（不同文件名但内容相同，保留一个）
    
    renamed_files = []
    max_index = find_max_index_in_category(category_dir, category_name)
    next_index = max_index + 1
    
    # 处理相同stem但不同内容的文件
    for stem, files in stem_to_files.items():
        if len(files) > 1:
            # 检查这些文件是否真的不同
            hashes = set()
            for f in files:
                info = get_image_info(f)
                if info and info['hash']:
                    hashes.add(info['hash'])
            
            # 如果哈希不同，说明内容不同，需要重命名
            if len(hashes) > 1:
                # 保留第一个，重命名其他的
                for f in files[1:]:
                    info = get_image_info(f)
                    if not info or not info['hash']:
                        continue
                    
                    # 使用哈希的后6位作为后缀
                    hash_suffix = info['hash'][-6:]
                    new_stem = f"{category_name}_{next_index:05d}_{hash_suffix}"
                    new_name = f"{new_stem}{f.suffix}"
                    new_path = category_dir / new_name
                    
                    # 确保新文件名不存在
                    while new_path.exists():
                        next_index += 1
                        new_stem = f"{category_name}_{next_index:05d}_{hash_suffix}"
                        new_name = f"{new_stem}{f.suffix}"
                        new_path = category_dir / new_name
                    
                    try:
                        f.rename(new_path)
                        renamed_files.append({
                            'old': f.name,
                            'new': new_name,
                            'reason': 'same_stem_different_content'
                        })
                        next_index += 1
                    except Exception as e:
                        print(f"  错误: 无法重命名 {f.name}: {e}")
    
    # 处理相同内容但不同文件名的文件（保留第一个，删除或重命名其他的）
    for hash_val, files in hash_to_files.items():
        if len(files) > 1:
            # 保留第一个，其他的标记为重复
            for f in files[1:]:
                # 检查是否已经被重命名过
                if any(r['old'] == f.name for r in renamed_files):
                    continue
                
                # 重命名为带_duplicate后缀
                new_stem = f"{f.stem}_duplicate_{hash_val[:6]}"
                new_name = f"{new_stem}{f.suffix}"
                new_path = category_dir / new_name
                
                try:
                    f.rename(new_path)
                    renamed_files.append({
                        'old': f.name,
                        'new': new_name,
                        'reason': 'duplicate_content'
                    })
                except Exception as e:
                    print(f"  错误: 无法重命名重复文件 {f.name}: {e}")
    
    return len(renamed_files), renamed_files

def regenerate_json_for_image(img_path, category_name, category_id):
    """为单个图片生成JSON文件"""
    try:
        info = get_image_info(img_path)
        if not info:
            return None
        
        file_stem = img_path.stem
        file_name = img_path.name
        
        # 检测图片类型
        is_multi_fruit, image_type, source_dataset = detect_image_type(
            img_path, info['width'], info['height']
        )
        
        # 生成ID（基于文件名哈希）
        id_hash = int(hashlib.md5(file_stem.encode()).hexdigest()[:8], 16)
        
        json_data = {
            "info": {
                "description": "Fruit Classification Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "task1_fruit_classification",
                "source": "multiple_datasets",
                "license": {
                    "name": "Creative Commons Attribution 4.0 International",
                    "url": "https://creativecommons.org/licenses/by/4.0/"
                }
            },
            "images": [{
                "id": id_hash,
                "width": info['width'],
                "height": info['height'],
                "file_name": file_name,
                "size": info['size'],
                "format": info['format'],
                "url": "",
                "hash": info['hash'],
                "status": "success",
                "original_filename": file_name,
                "is_multi_fruit": is_multi_fruit,
                "image_type": image_type,
                "source_dataset": source_dataset
            }],
            "annotations": [{
                "id": id_hash + 1000000000,
                "image_id": id_hash,
                "category_id": category_id,
                "segmentation": [],
                "area": info['width'] * info['height'],
                "bbox": [0, 0, info['width'], info['height']],
                "is_multi_fruit": is_multi_fruit
            }],
            "categories": [{
                "id": category_id,
                "name": category_name,
                "supercategory": "Fruit",
                "is_multi_fruit": is_multi_fruit
            }]
        }
        
        return json_data
    except Exception as e:
        return None

def regenerate_all_annotations():
    """重新生成所有JSON和CSV文件"""
    print("\n重新生成所有标注文件...")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    category_id_map = {}
    category_id_counter = 2000000000
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        
        categories = sorted([d for d in split_dir.iterdir() 
                          if d.is_dir() and d.name not in ['json', 'csv', 'sets']])
        
        for category_dir in tqdm(categories, desc=f"处理 {split}"):
            category_name = category_dir.name
            
            # 获取类别ID
            if category_name not in category_id_map:
                category_id_map[category_name] = category_id_counter
                category_id_counter += 1
            
            category_id = category_id_map[category_name]
            
            # 创建json和csv目录
            json_dir = category_dir / 'json'
            csv_dir = category_dir / 'csv'
            json_dir.mkdir(exist_ok=True)
            csv_dir.mkdir(exist_ok=True)
            
            # 获取所有图片文件（排除json、csv、sets目录和重复标记的文件）
            image_files = []
            for ext in IMAGE_EXTENSIONS:
                for img_file in category_dir.glob(f'*{ext}'):
                    # 排除重复标记的文件
                    if '_duplicate_' not in img_file.stem:
                        image_files.append(img_file)
            
            # 按文件名排序
            image_files = sorted(image_files, key=lambda x: x.name)
            
            # 为每个图片生成JSON和CSV
            for img_file in tqdm(image_files, desc=f"  {category_name}", leave=False):
                if img_file.parent.name in ['json', 'csv', 'sets']:
                    continue
                
                file_stem = img_file.stem
                json_path = json_dir / f"{file_stem}.json"
                csv_path = csv_dir / f"{file_stem}.csv"
                
                # 生成JSON
                json_data = regenerate_json_for_image(img_file, category_name, category_id)
                if json_data:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    # 生成CSV
                    img_info = json_data['images'][0]
                    ann_info = json_data['annotations'][0]
                    csv_content = f"image_id,file_name,width,height,category_id,category_name,bbox_x,bbox_y,bbox_width,bbox_height,area,is_multi_fruit\n"
                    csv_content += f"{img_info['id']},{img_info['file_name']},{img_info['width']},{img_info['height']},"
                    csv_content += f"{ann_info['category_id']},{category_name},"
                    csv_content += f"{ann_info['bbox'][0]},{ann_info['bbox'][1]},{ann_info['bbox'][2]},{ann_info['bbox'][3]},"
                    csv_content += f"{ann_info['area']},{ann_info['is_multi_fruit']}\n"
                    
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write(csv_content)

def regenerate_split_files():
    """重新生成split文件"""
    print("\n重新生成split文件...")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    all_images = []
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        
        categories = sorted([d for d in split_dir.iterdir() 
                          if d.is_dir() and d.name not in ['json', 'csv', 'sets']])
        
        split_images = []
        for category_dir in categories:
            category_name = category_dir.name
            
            # 获取所有图片文件stem（排除重复标记的）
            image_stems = set()
            for ext in IMAGE_EXTENSIONS:
                for img_file in category_dir.glob(f'*{ext}'):
                    if '_duplicate_' not in img_file.stem:
                        image_stems.add(img_file.stem)
            
            for stem in sorted(image_stems):
                split_images.append(f"{category_name}/{stem}")
                all_images.append(f"{split}/{category_name}/{stem}")
        
        # 写入split文件
        sets_dir = DATASET_DIR / split / 'sets'
        sets_dir.mkdir(exist_ok=True)
        
        split_file = sets_dir / f"{split}.txt"
        with open(split_file, 'w', encoding='utf-8') as f:
            for img_path in sorted(split_images):
                f.write(f"{img_path}\n")
        
        print(f"  {split}: {len(split_images)} 个图片")
    
    # 生成all.txt
    all_file = DATASET_DIR / 'all.txt'
    with open(all_file, 'w', encoding='utf-8') as f:
        for img_path in sorted(all_images):
            f.write(f"{img_path}\n")
    
    print(f"  总计: {len(all_images)} 个图片")

def main():
    print("="*60)
    print("修复所有图片命名并重新生成标注文件")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    total_renamed = 0
    
    # 第一步：修复图片命名
    print("\n第一步：检测并修复图片命名问题...")
    print("-"*60)
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        
        print(f"\n处理 {split.upper()} split:")
        categories = sorted([d for d in split_dir.iterdir() 
                          if d.is_dir() and d.name not in ['json', 'csv', 'sets']])
        
        for category_dir in tqdm(categories, desc=f"{split}"):
            category_name = category_dir.name
            renamed_count, renamed_files = fix_category_images(
                category_dir, category_name, split
            )
            total_renamed += renamed_count
            
            if renamed_files:
                print(f"  {category_name}: 重命名了 {renamed_count} 个文件")
    
    print(f"\n总计重命名: {total_renamed} 个文件")
    
    # 第二步：重新生成所有标注文件
    regenerate_all_annotations()
    
    # 第三步：重新生成split文件
    regenerate_split_files()
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)

if __name__ == '__main__':
    main()
