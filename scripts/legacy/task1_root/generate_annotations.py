#!/usr/bin/env python3
"""
按照 Plant_Village_Apple 数据集格式规范，为 task1_fruit_classification 数据集生成标注文件
包括：CSV标注、JSON标注、labelmap.json、数据集划分文件
"""
import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

def generate_random_id():
    """生成10位随机整数ID"""
    return random.randint(1000000000, 9999999999)

def get_image_info(image_path):
    """获取图像信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            file_size = os.path.getsize(image_path)
            return {
                'width': width,
                'height': height,
                'size': file_size,
                'format': img.format or 'JPG'
            }
    except Exception as e:
        print(f"警告: 无法读取图像 {image_path}: {e}")
        return None

def create_csv_annotation(image_path, category_id, image_info):
    """
    创建CSV标注文件（全图标注）
    对于分类任务，使用整个图像作为边界框
    """
    if image_info is None:
        return None
    
    width = image_info['width']
    height = image_info['height']
    
    # 全图边界框：整个图像
    x = 0
    y = 0
    bbox_width = width
    bbox_height = height
    
    # 生成标注项ID
    item_id = generate_random_id()
    
    # CSV格式：表头 + 数据行
    csv_content = f"#item,x,y,width,height,label\n"
    csv_content += f"{item_id},{x},{y},{bbox_width},{bbox_height},{category_id}\n"
    
    return csv_content

def create_json_annotation(image_path, category_id, category_name, image_info, image_id):
    """
    创建JSON标注文件（COCO格式）
    对于分类任务，使用整个图像作为边界框
    """
    if image_info is None:
        return None
    
    width = image_info['width']
    height = image_info['height']
    file_size = image_info['size']
    image_format = image_info['format']
    
    # 获取文件名
    file_name = os.path.basename(image_path)
    file_stem = os.path.splitext(file_name)[0]
    
    # 全图边界框
    bbox = [0, 0, width, height]
    area = width * height
    
    # 生成标注ID
    annotation_id = generate_random_id()
    
    # JSON格式ID映射：标准ID -> JSON大整数格式
    json_category_id = 2000000000 + category_id
    
    # 构建JSON结构
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
        "images": [
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
                "size": file_size,
                "format": image_format,
                "url": "",
                "hash": "",
                "status": "success",
                "original_filename": file_name
            }
        ],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": json_category_id,
                "segmentation": [],
                "area": area,
                "bbox": bbox
            }
        ],
        "categories": [
            {
                "id": json_category_id,
                "name": category_name.lower(),
                "supercategory": "Fruit"
            }
        ]
    }
    
    return json_data

def create_labelmap(categories):
    """
    创建labelmap.json文件
    categories: 类别列表（已排序）
    """
    labelmap = [
        {
            "object_id": 0,
            "label_id": 0,
            "keyboard_shortcut": "0",
            "object_name": "background"
        }
    ]
    
    for idx, category in enumerate(categories, start=1):
        labelmap.append({
            "object_id": idx,
            "label_id": idx,
            "keyboard_shortcut": str(idx),
            "object_name": category.lower()
        })
    
    return labelmap

def process_dataset(base_dir, output_structure=False):
    """
    处理数据集，生成标注文件
    
    Args:
        base_dir: 数据集根目录
        output_structure: 是否创建标准目录结构（csv/, json/, images/, sets/）
    """
    base_dir = Path(base_dir)
    splits = ['train', 'val', 'test']
    
    # 收集所有类别
    all_categories = set()
    for split in splits:
        split_dir = base_dir / split
        if split_dir.exists():
            categories = [d for d in os.listdir(split_dir) 
                         if (split_dir / d).is_dir() and d != 'unknown']
            all_categories.update(categories)
    
    all_categories = sorted(list(all_categories))
    print(f"找到 {len(all_categories)} 个类别")
    
    # 创建类别ID映射
    category_to_id = {cat: idx + 1 for idx, cat in enumerate(all_categories)}
    
    # 创建labelmap.json
    labelmap = create_labelmap(all_categories)
    labelmap_path = base_dir / 'labelmap.json'
    with open(labelmap_path, 'w', encoding='utf-8') as f:
        json.dump(labelmap, f, indent=2, ensure_ascii=False)
    print(f"✓ 创建 labelmap.json: {len(labelmap)} 个类别")
    
    # 统计信息
    stats = {
        'total_images': 0,
        'total_csv': 0,
        'total_json': 0,
        'categories': {}
    }
    
    # 用于收集所有split的图像（用于生成all.txt）
    category_all_images = {cat: [] for cat in all_categories}
    
    # 处理每个split和类别
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\n处理 {split} split...")
        
        for category in all_categories:
            category_dir = split_dir / category
            if not category_dir.exists():
                continue
            
            category_id = category_to_id[category]
            print(f"  处理类别: {category} (ID: {category_id})")
            
            # 获取所有图像文件（按stem去重，每个图像只保留一个文件）
            image_files_dict = {}
            for ext in IMAGE_EXTENSIONS:
                for img_path in category_dir.glob(f'*{ext}'):
                    stem = img_path.stem
                    # 如果还没有这个stem的文件，或者当前文件是首选格式，则添加
                    if stem not in image_files_dict:
                        image_files_dict[stem] = img_path
                    else:
                        # 优先选择JPG/PNG格式
                        current_ext = img_path.suffix.lower()
                        existing_ext = image_files_dict[stem].suffix.lower()
                        if current_ext in ['.jpg', '.png'] and existing_ext not in ['.jpg', '.png']:
                            image_files_dict[stem] = img_path
            
            image_files = sorted(image_files_dict.values())
            
            if len(image_files) == 0:
                continue
            
            # 创建输出目录结构
            if output_structure:
                csv_dir = category_dir / 'csv'
                json_dir = category_dir / 'json'
                images_dir = category_dir / 'images'
                sets_dir = category_dir / 'sets'
                
                csv_dir.mkdir(exist_ok=True)
                json_dir.mkdir(exist_ok=True)
                images_dir.mkdir(exist_ok=True)
                sets_dir.mkdir(exist_ok=True)
            else:
                # 在当前目录创建csv和json子目录
                csv_dir = category_dir / 'csv'
                json_dir = category_dir / 'json'
                sets_dir = category_dir / 'sets'
                
                csv_dir.mkdir(exist_ok=True)
                json_dir.mkdir(exist_ok=True)
                sets_dir.mkdir(exist_ok=True)
            
            # 处理每个图像
            image_list = []
            csv_count = 0
            json_count = 0
            
            # 使用tqdm显示进度
            for image_path in tqdm(image_files, desc=f"    {category} ({split})", leave=False):
                # 获取图像信息
                image_info = get_image_info(image_path)
                if image_info is None:
                    continue
                
                file_stem = image_path.stem
                file_name = image_path.name
                
                # 检查文件是否已存在，如果存在则跳过
                csv_path = csv_dir / f"{file_stem}.csv"
                json_path = json_dir / f"{file_stem}.json"
                
                csv_created = False
                json_created = False
                
                # 创建CSV标注（如果不存在）
                if not csv_path.exists():
                    csv_content = create_csv_annotation(image_path, category_id, image_info)
                    if csv_content:
                        with open(csv_path, 'w', encoding='utf-8') as f:
                            f.write(csv_content)
                        csv_count += 1
                        csv_created = True
                else:
                    csv_count += 1  # 统计已存在的文件
                
                # 创建JSON标注（如果不存在）
                if not json_path.exists():
                    image_id = generate_random_id()
                    json_data = create_json_annotation(
                        str(image_path), category_id, category, image_info, image_id
                    )
                    if json_data:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        json_count += 1
                        json_created = True
                else:
                    json_count += 1  # 统计已存在的文件
                
                # 添加到图像列表（用于划分文件）
                image_list.append(file_stem)
                
                stats['total_images'] += 1
            
            # 创建数据集划分文件
            image_list = sorted(image_list)
            
            # 根据当前split，创建对应的划分文件
            # 每个类别在每个split中都有独立的划分文件
            split_file = sets_dir / f'{split}.txt'
            with open(split_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(image_list) + '\n')
            
            # 收集所有split的图像（用于all.txt）
            category_all_images[category].extend(image_list)
            
            print(f"    ✓ 处理 {len(image_files)} 个图像")
            print(f"    ✓ 创建 {csv_count} 个CSV文件")
            print(f"    ✓ 创建 {json_count} 个JSON文件")
            print(f"    ✓ 创建划分文件")
            
            # 更新统计
            if category not in stats['categories']:
                stats['categories'][category] = {}
            stats['categories'][category][split] = {
                'images': len(image_files),
                'csv': csv_count,
                'json': json_count
            }
            stats['total_csv'] += csv_count
            stats['total_json'] += json_count
    
    # 为每个类别生成all.txt文件（包含所有split的图像）
    print(f"\n生成 all.txt 文件...")
    for category in all_categories:
        all_images = sorted(list(set(category_all_images[category])))
        if len(all_images) > 0:
            # 找到任意一个split的sets目录
            for split in splits:
                sets_dir = base_dir / split / category / 'sets'
                if sets_dir.exists():
                    all_file = sets_dir / 'all.txt'
                    with open(all_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(all_images) + '\n')
                    break
    
    # 保存统计信息
    stats_path = base_dir / 'annotation_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"总图像数: {stats['total_images']}")
    print(f"总CSV文件: {stats['total_csv']}")
    print(f"总JSON文件: {stats['total_json']}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import sys
    
    base_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset'
    
    # 是否创建标准目录结构（将图像移动到images/目录）
    output_structure = '--output-structure' in sys.argv
    
    if output_structure:
        print("注意: --output-structure 选项会创建标准目录结构（csv/, json/, images/, sets/）")
        print("但不会移动现有图像文件，只创建标注文件")
    
    process_dataset(base_dir, output_structure=output_structure)
