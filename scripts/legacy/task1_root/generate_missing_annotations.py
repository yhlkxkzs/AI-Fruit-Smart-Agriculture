#!/usr/bin/env python3
"""
为缺失标注的图像生成标注文件（CSV和JSON）
"""

import json
import random
import os
from pathlib import Path
from PIL import Image

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
    """创建CSV标注文件"""
    if image_info is None:
        return None
    
    width = image_info['width']
    height = image_info['height']
    
    item_id = generate_random_id()
    csv_content = f"#item,x,y,width,height,label\n"
    csv_content += f"{item_id},0,0,{width},{height},{category_id}\n"
    
    return csv_content

def create_json_annotation(image_path, category_id, category_name, image_info, image_id):
    """创建JSON标注文件"""
    if image_info is None:
        return None
    
    width = image_info['width']
    height = image_info['height']
    file_size = image_info['size']
    image_format = image_info['format']
    
    file_name = os.path.basename(image_path)
    bbox = [0, 0, width, height]
    area = width * height
    annotation_id = generate_random_id()
    json_category_id = 2000000000 + category_id
    
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

def classify_apple_image(img_path):
    """分类苹果图像：花或果实"""
    name_lower = img_path.stem.lower()
    if any(keyword in name_lower for keyword in ['flower', 'blossom', 'bloom', '花']):
        return 'flower'
    
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        import numpy as np
        img_array = np.array(img)
        
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        avg_brightness = (avg_r + avg_g + avg_b) / 3
        
        max_channel = max(avg_r, avg_g, avg_b)
        min_channel = min(avg_r, avg_g, avg_b)
        saturation = (max_channel - min_channel) / max_channel if max_channel > 0 else 0
        
        if avg_brightness > 200 and saturation < 0.3:
            return 'flower'
        elif avg_r > avg_g + 30 and avg_r > avg_b + 30:
            return 'fruit'
        elif avg_g > avg_r + 30 and avg_g > avg_b + 30:
            return 'fruit'
        elif avg_brightness > 180:
            return 'flower'
        else:
            return 'fruit'
    except Exception as e:
        print(f"  错误分类 {img_path.name}: {e}")
        return 'fruit'

def process_missing_annotations(apple_dir):
    """为缺失标注的图像生成标注文件"""
    apple_dir = Path(apple_dir)
    if not apple_dir.exists():
        print(f"错误: 目录不存在 {apple_dir}")
        return
    
    print(f"处理目录: {apple_dir}")
    print("="*60)
    
    image_dir = apple_dir
    json_dir = apple_dir / 'json'
    csv_dir = apple_dir / 'csv'
    
    # 确保目录存在
    json_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)
    
    # 找出所有图像stem
    all_stems = set()
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
        for img in image_dir.glob(ext):
            all_stems.add(img.stem)
    
    # 找出已有JSON的stem
    json_stems = set()
    for json_file in json_dir.glob('*.json'):
        json_stems.add(json_file.stem)
    
    # 找出缺失标注的stem
    missing_stems = sorted(list(all_stems - json_stems))
    
    print(f"缺失标注的图像数: {len(missing_stems)}")
    
    if not missing_stems:
        print("所有图像都有标注！")
        return
    
    # apple的category_id是2（根据之前的labelmap.json）
    category_id = 2
    category_name = "apple"
    
    generated_count = 0
    flower_count = 0
    fruit_count = 0
    
    for stem in missing_stems:
        # 找到对应的图像文件（优先选择.jpg或.JPG）
        img_path = None
        for ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP']:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            print(f"  警告: 找不到图像文件 {stem}")
            continue
        
        # 获取图像信息
        image_info = get_image_info(img_path)
        if image_info is None:
            print(f"  警告: 无法读取图像 {img_path.name}")
            continue
        
        # 分类图像（花或果实）
        subcategory = classify_apple_image(img_path)
        
        # 生成ID
        image_id = generate_random_id()
        
        # 创建CSV标注
        csv_content = create_csv_annotation(img_path, category_id, image_info)
        if csv_content:
            csv_path = csv_dir / f"{stem}.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        # 创建JSON标注
        json_data = create_json_annotation(img_path, category_id, category_name, image_info, image_id)
        if json_data:
            # 添加subcategory信息
            json_data['categories'][0]['subcategory'] = subcategory
            json_data['categories'][0]['name'] = f"apple_{subcategory}"
            json_data['annotations'][0]['subcategory'] = subcategory
            
            json_path = json_dir / f"{stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        generated_count += 1
        if subcategory == 'flower':
            flower_count += 1
        else:
            fruit_count += 1
        
        if generated_count % 10 == 0:
            print(f"  已处理 {generated_count}/{len(missing_stems)} 个图像...")
    
    print(f"\n完成！")
    print(f"  生成标注文件: {generated_count}")
    print(f"  苹果花: {flower_count}")
    print(f"  苹果果实: {fruit_count}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        apple_dir = sys.argv[1]
    else:
        apple_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/apple'
    
    process_missing_annotations(apple_dir)
