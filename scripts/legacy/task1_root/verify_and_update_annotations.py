#!/usr/bin/env python3
"""
验证并更新所有JSON文件的标注
通过检查实际图片来判断图片类型，确保标注准确
"""
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

def detect_image_type(img_path, width, height):
    """
    根据图片尺寸和特征判断图片类型
    
    Returns:
        tuple: (is_multi_fruit, image_type, source_dataset, note)
    """
    # 64x64像素：通常是DeepFruits的多水果组合
    if width == 64 and height == 64:
        return True, 'combination', 'deepfruits', '64x64 pixel image from DeepFruits dataset (multi-fruit combination)'
    
    # 1024x1024像素：可能是fruit-SALAD的艺术风格图片
    if width == 1024 and height == 1024:
        # 检查是否是PNG格式（fruit-SALAD通常是PNG）
        if img_path.suffix.lower() == '.png':
            return False, 'artistic', 'fruit-salad', '1024x1024 PNG image from fruit-SALAD dataset (synthetic artistic style)'
        else:
            return False, 'artistic', 'unknown', '1024x1024 image (possibly synthetic/artistic)'
    
    # 非常小的图片（<200像素）：可能是多水果组合的缩略图
    if width < 200 or height < 200:
        return True, 'combination', 'deepfruits', f'Small image ({width}x{height}) likely from multi-fruit dataset'
    
    # 中等尺寸的图片：通常是单一水果的真实照片
    if 200 <= width <= 2000 and 200 <= height <= 2000:
        return False, 'single_fruit', 'unknown', f'Standard size image ({width}x{height}) - likely single fruit photo'
    
    # 超大图片：可能是艺术风格或合成图片
    if width > 2000 or height > 2000:
        return False, 'artistic', 'unknown', f'Large image ({width}x{height}) - possibly artistic or synthetic'
    
    # 默认：单一水果
    return False, 'single_fruit', 'unknown', f'Image ({width}x{height}) - default to single fruit'

def find_image_file(category_dir, file_stem):
    """查找图片文件（支持多种扩展名）"""
    extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']
    for ext in extensions:
        img_path = category_dir / f"{file_stem}{ext}"
        if img_path.exists():
            return img_path
    return None

def update_json_annotation(json_path, category_dir, category_name):
    """更新单个JSON文件的标注"""
    try:
        with open(json_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False, "No images in JSON"
        
        img_info = data['images'][0]
        file_name = img_info.get('file_name', '')
        file_stem = Path(file_name).stem
        
        # 查找实际的图片文件
        img_path = find_image_file(category_dir, file_stem)
        if not img_path:
            return False, f"Image file not found for {file_name}"
        
        # 读取图片信息
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            return False, f"Cannot read image: {e}"
        
        # 检测图片类型
        is_multi_fruit, image_type, source_dataset, note = detect_image_type(img_path, width, height)
        
        # 更新images部分
        data['images'][0]['is_multi_fruit'] = is_multi_fruit
        data['images'][0]['image_type'] = image_type
        if source_dataset != 'unknown':
            data['images'][0]['source_dataset'] = source_dataset
        if 'note' not in data['images'][0]:
            data['images'][0]['note'] = note
        
        # 更新annotations部分
        if 'annotations' in data and len(data['annotations']) > 0:
            data['annotations'][0]['is_multi_fruit'] = is_multi_fruit
        
        # 更新categories部分
        if 'categories' in data and len(data['categories']) > 0:
            data['categories'][0]['is_multi_fruit'] = is_multi_fruit
            if is_multi_fruit:
                data['categories'][0]['note'] = "This image contains multiple fruits or is from a multi-fruit dataset, not suitable for single fruit classification"
            elif image_type == 'artistic':
                data['categories'][0]['note'] = "This is a synthetic/artistic image, may not be suitable for real-world fruit classification"
        
        # 写回文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True, f"Updated: multi={is_multi_fruit}, type={image_type}, source={source_dataset}"
    
    except Exception as e:
        return False, f"Error: {e}"

def process_category(category_dir, category_name, split_name):
    """处理单个类别"""
    json_dir = category_dir / 'json'
    if not json_dir.exists():
        return 0, 0, []
    
    json_files = sorted(list(json_dir.glob('*.json')))
    if not json_files:
        return 0, 0, []
    
    updated_count = 0
    error_count = 0
    errors = []
    
    for json_file in tqdm(json_files, desc=f"  {category_name} ({split_name})", leave=False):
        success, message = update_json_annotation(json_file, category_dir, category_name)
        if success:
            updated_count += 1
        else:
            error_count += 1
            errors.append(f"{json_file.name}: {message}")
    
    return updated_count, error_count, errors

def main():
    dataset_dir = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    splits = ['train', 'val', 'test']
    
    total_updated = 0
    total_errors = 0
    all_errors = []
    
    print("="*60)
    print("验证并更新所有JSON文件的标注")
    print("="*60)
    print()
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\n处理 {split.upper()} split:")
        print("-"*60)
        
        # 获取所有类别目录
        categories = sorted([d for d in split_dir.iterdir() 
                          if d.is_dir() and d.name not in ['json', 'csv', 'sets']])
        
        for category_dir in tqdm(categories, desc=f"{split}"):
            category_name = category_dir.name
            updated, errors, error_list = process_category(category_dir, category_name, split)
            total_updated += updated
            total_errors += errors
            all_errors.extend(error_list)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总计更新: {total_updated} 个JSON文件")
    print(f"错误: {total_errors} 个文件")
    
    if all_errors:
        print(f"\n错误详情（前20个）:")
        for error in all_errors[:20]:
            print(f"  {error}")
        if len(all_errors) > 20:
            print(f"  ... 还有 {len(all_errors) - 20} 个错误")

if __name__ == '__main__':
    main()
