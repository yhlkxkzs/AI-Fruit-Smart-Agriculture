#!/usr/bin/env python3
"""
修正apple文件夹中所有图片的标注：
- 除了 apple_000000.bmp 到 apple_000012.bmp（这13个是苹果花）之外
- 所有其他图片都应该是：
  - content_type: "fruit"
  - image_style: "real"
"""

import json
from pathlib import Path
from tqdm import tqdm

# 需要排除的BMP文件（这些是苹果花）
EXCLUDED_FILES = {f"apple_{i:06d}.bmp" for i in range(13)}

def fix_apple_annotation(json_path):
    """修正单个JSON文件的标注"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False, "no images"
        
        img_info = data['images'][0]
        filename = img_info.get('file_name', '')
        
        # 检查是否是需要排除的文件
        if filename in EXCLUDED_FILES:
            return False, "excluded (flower)"
        
        # 检查是否需要修正
        needs_fix = False
        changes = []
        
        # 修正 images[0]
        if img_info.get('content_type') != 'fruit':
            img_info['content_type'] = 'fruit'
            needs_fix = True
            changes.append(f"content_type: {img_info.get('content_type', 'N/A')} -> fruit")
        
        if img_info.get('subcategory') != 'fruit':
            img_info['subcategory'] = 'fruit'
            needs_fix = True
            changes.append(f"subcategory: {img_info.get('subcategory', 'N/A')} -> fruit")
        
        if img_info.get('image_style') != 'real':
            old_style = img_info.get('image_style', 'N/A')
            img_info['image_style'] = 'real'
            needs_fix = True
            changes.append(f"image_style: {old_style} -> real")
        
        # 修正 annotations[0]
        if 'annotations' in data and len(data['annotations']) > 0:
            ann = data['annotations'][0]
            if ann.get('content_type') != 'fruit':
                ann['content_type'] = 'fruit'
                needs_fix = True
        
        # 修正 categories[0]
        if 'categories' in data and len(data['categories']) > 0:
            cat = data['categories'][0]
            if cat.get('subcategory') != 'fruit':
                cat['subcategory'] = 'fruit'
                needs_fix = True
            
            if cat.get('name') != 'apple_fruit':
                cat['name'] = 'apple_fruit'
                needs_fix = True
            
            if cat.get('image_style') != 'real':
                cat['image_style'] = 'real'
                needs_fix = True
        
        if needs_fix:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True, "; ".join(changes)
        
        return False, "already correct"
        
    except Exception as e:
        return False, f"error: {e}"

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    if not json_dir.exists():
        print(f"错误: JSON目录不存在: {json_dir}")
        return
    
    print("="*70)
    print("修正apple文件夹中所有图片的标注")
    print("="*70)
    print(f"排除文件（苹果花）: {sorted(EXCLUDED_FILES)}")
    print(f"其他所有文件将设置为: content_type='fruit', image_style='real'")
    print()
    
    # 获取所有JSON文件
    json_files = sorted(json_dir.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")
    print()
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    already_correct_count = 0
    
    for json_path in tqdm(json_files, desc="处理中"):
        fixed, reason = fix_apple_annotation(json_path)
        
        if fixed:
            fixed_count += 1
            if fixed_count <= 10:  # 只显示前10个修正的文件
                print(f"  ✓ 修正: {json_path.name} ({reason})")
        elif reason == "excluded (flower)":
            skipped_count += 1
        elif reason == "already correct":
            already_correct_count += 1
        elif reason.startswith("error"):
            error_count += 1
            print(f"  ✗ 错误: {json_path.name} - {reason}")
        elif reason == "no images":
            skipped_count += 1
    
    print()
    print("="*70)
    print("修正完成！")
    print("="*70)
    print(f"修正的文件数: {fixed_count}")
    print(f"已正确的文件数: {already_correct_count}")
    print(f"跳过的文件数（苹果花）: {skipped_count}")
    print(f"错误的文件数: {error_count}")
    print(f"总计: {len(json_files)}")
    print("="*70)

if __name__ == '__main__':
    main()
