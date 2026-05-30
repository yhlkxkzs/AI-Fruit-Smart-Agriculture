#!/usr/bin/env python3
"""
修正苹果图片风格标注
将特定图片（如color_peaks数据集的2704x1520图片）的风格从real改为cartoon
"""

import json
from pathlib import Path
from tqdm import tqdm

def fix_apple_style_annotation(json_path):
    """修正单个JSON文件的风格标注"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False
        
        img_info = data['images'][0]
        current_style = img_info.get('image_style', 'real')
        source_dataset = img_info.get('source_dataset', '')
        width = img_info.get('width', 0)
        height = img_info.get('height', 0)
        
        # 判断是否需要修正
        # 用户指定的这些图片（2704x1520的苹果花图片）应该是real，不是cartoon
        should_be_real = False
        
        if width == 2704 and height == 1520:
            should_be_real = True
        
        if should_be_real and current_style != 'real':
            # 更新images字段
            img_info['image_style'] = 'real'
            
            # 更新categories字段
            if 'categories' in data and len(data['categories']) > 0:
                data['categories'][0]['image_style'] = 'real'
            
            # 保存
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"  错误处理 {json_path.name}: {e}")
        return False

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    splits = ['train', 'val', 'test']
    
    total_fixed = 0
    
    # 用户指定的特定文件列表
    specific_files = [
        'apple_000000.bmp',
        'apple_000001.bmp',
        'apple_000002.bmp',
        'apple_000003.bmp',
        'apple_000004.bmp',
        'apple_000005.bmp',
        'apple_000006.bmp',
        'apple_000007.bmp',
        'apple_000008.bmp',
        'apple_000009.bmp',
        'apple_000010.bmp',
        'apple_000011.bmp',
        'apple_000012.bmp',
    ]
    
    print("="*60)
    print("修正苹果图片风格标注（cartoon -> real）")
    print("="*60)
    
    for split in splits:
        apple_dir = dataset_root / split / 'apple'
        json_dir = apple_dir / 'json'
        
        if not json_dir.exists():
            continue
        
        print(f"\n处理 {split}/apple...")
        
        # 处理用户指定的文件
        for img_file in specific_files:
            json_file = json_dir / f"{Path(img_file).stem}.json"
            if json_file.exists():
                if fix_apple_style_annotation(json_file):
                    total_fixed += 1
                    print(f"  修正: {json_file.name}")
        
        # 同时处理所有color_peaks数据集的2704x1520图片
        json_files = list(json_dir.glob('*.json'))
        for json_path in tqdm(json_files, desc=f"  {split}/apple", leave=False):
            if json_path.stem not in [Path(f).stem for f in specific_files]:
                if fix_apple_style_annotation(json_path):
                    total_fixed += 1
    
    print("\n" + "="*60)
    print(f"处理完成！共修正 {total_fixed} 个文件")
    print("="*60)

if __name__ == '__main__':
    main()
