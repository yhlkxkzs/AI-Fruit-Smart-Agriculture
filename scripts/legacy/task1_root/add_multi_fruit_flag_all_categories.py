#!/usr/bin/env python3
"""
为所有水果类别的图片添加is_multi_fruit标记
识别多水果组合图片并添加标记
"""

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def is_multi_fruit_image(img_path):
    """
    判断图片是否为多水果组合图片
    判断标准：
    1. 图片尺寸为64x64（DeepFruits数据集特征）
    2. 或者来自已知的多水果组合数据集
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        # DeepFruits数据集特征：64x64像素
        if width == 64 and height == 64:
            return True, 'deepfruits'
        
        # 可以添加其他判断标准
        # 例如：检查文件名、路径等
        
        return False, None
    except Exception as e:
        return False, None

def update_json_with_multi_fruit_flag(json_path, is_multi_fruit, source_dataset=None):
    """更新JSON文件，添加is_multi_fruit标记"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新images字段
        if 'images' in data and len(data['images']) > 0:
            data['images'][0]['is_multi_fruit'] = is_multi_fruit
            if is_multi_fruit and source_dataset:
                data['images'][0]['source_dataset'] = source_dataset
                data['images'][0]['image_type'] = 'combination'
            elif not is_multi_fruit:
                # 单一水果图片
                data['images'][0]['is_multi_fruit'] = False
                data['images'][0]['image_type'] = 'single_fruit'
        
        # 更新annotations字段
        if 'annotations' in data and len(data['annotations']) > 0:
            data['annotations'][0]['is_multi_fruit'] = is_multi_fruit
        
        # 更新categories字段
        if 'categories' in data and len(data['categories']) > 0:
            data['categories'][0]['is_multi_fruit'] = is_multi_fruit
            if is_multi_fruit:
                data['categories'][0]['note'] = 'This image contains multiple fruits, not suitable for single fruit classification'
            else:
                # 移除note或设置为单一水果
                if 'note' in data['categories'][0] and 'multiple fruits' in data['categories'][0]['note']:
                    del data['categories'][0]['note']
        
        # 保存更新后的JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"  错误更新 {json_path.name}: {e}")
        return False

def process_category(category_dir, split_name):
    """处理单个类别的所有图片"""
    category = category_dir.name
    json_dir = category_dir / 'json'
    
    if not json_dir.exists():
        return 0, 0
    
    json_files = list(json_dir.glob('*.json'))
    if not json_files:
        return 0, 0
    
    multi_fruit_count = 0
    single_fruit_count = 0
    
    for json_file in tqdm(json_files, desc=f'  {category}', leave=False):
        try:
            # 读取JSON获取图片文件名
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'images' not in data or len(data['images']) == 0:
                continue
            
            img_filename = data['images'][0].get('file_name', json_file.stem)
            img_path = category_dir / img_filename
            
            # 如果图片不存在，尝试其他扩展名
            if not img_path.exists():
                for ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP']:
                    alt_path = category_dir / f"{json_file.stem}{ext}"
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            if not img_path.exists():
                continue
            
            # 判断是否为多水果组合图片
            is_multi, source = is_multi_fruit_image(img_path)
            
            # 更新JSON
            if update_json_with_multi_fruit_flag(json_file, is_multi, source):
                if is_multi:
                    multi_fruit_count += 1
                else:
                    single_fruit_count += 1
        
        except Exception as e:
            continue
    
    return multi_fruit_count, single_fruit_count

def process_all_categories(dataset_dir):
    """处理所有类别的图片"""
    dataset_dir = Path(dataset_dir)
    
    print("="*60)
    print("为所有类别添加is_multi_fruit标记")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    total_stats = {
        'multi_fruit': 0,
        'single_fruit': 0,
        'categories_processed': 0
    }
    
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\n处理 {split.upper()} split...")
        
        # 获取所有类别目录
        categories = [d for d in split_dir.iterdir() 
                     if d.is_dir() and d.name not in ['json', 'csv', 'sets']]
        
        split_multi = 0
        split_single = 0
        
        for category_dir in tqdm(sorted(categories), desc=f'{split}'):
            multi_count, single_count = process_category(category_dir, split)
            split_multi += multi_count
            split_single += single_count
            
            if multi_count > 0 or single_count > 0:
                total_stats['categories_processed'] += 1
        
        total_stats['multi_fruit'] += split_multi
        total_stats['single_fruit'] += split_single
        
        print(f"  {split.upper()}:")
        print(f"    多水果组合: {split_multi:6d} 张")
        print(f"    单一水果: {split_single:6d} 张")
        print(f"    总计: {split_multi + split_single:6d} 张")
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总计:")
    print(f"  多水果组合图片: {total_stats['multi_fruit']:,} 张")
    print(f"  单一水果图片: {total_stats['single_fruit']:,} 张")
    print(f"  处理类别数: {total_stats['categories_processed']} 个")
    print("\n✅ 所有图片已添加is_multi_fruit标记！")

def main():
    import sys
    
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset'
    
    process_all_categories(dataset_dir)

if __name__ == '__main__':
    main()
