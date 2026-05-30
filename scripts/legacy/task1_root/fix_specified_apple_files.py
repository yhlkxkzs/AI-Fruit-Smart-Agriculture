#!/usr/bin/env python3
"""
先修正用户指定的文件（apple_000000 到 apple_000012 都应该是 flower）
然后使用CLIP模型识别其他文件
"""

import json
from pathlib import Path
from tqdm import tqdm

def fix_json_annotation(json_path, detected_type):
    """修正JSON文件的标注"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False
        
        img_info = data['images'][0]
        current_type = img_info.get('content_type', '')
        
        if detected_type and detected_type != current_type:
            # 更新所有相关字段
            img_info['content_type'] = detected_type
            img_info['subcategory'] = detected_type
            
            if 'annotations' in data and len(data['annotations']) > 0:
                data['annotations'][0]['content_type'] = detected_type
            
            if 'categories' in data and len(data['categories']) > 0:
                category = data['categories'][0]
                category['subcategory'] = detected_type
                if detected_type == 'flower':
                    category['name'] = 'apple_flower'
                elif detected_type == 'fruit':
                    category['name'] = 'apple_fruit'
                elif detected_type == 'leaf':
                    category['name'] = 'apple_leaf'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"  错误修正 {json_path.name}: {e}")
        return False

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    print("="*70)
    print("修正用户指定的苹果图片标注")
    print("="*70)
    print()
    
    # 用户指定的文件（apple_000000 到 apple_000012 都应该是 flower）
    specified_files = []
    for i in range(13):
        json_name = f"apple_{i:06d}.json"
        json_path = json_dir / json_name
        if json_path.exists():
            specified_files.append((json_path, 'flower'))
    
    print(f"找到 {len(specified_files)} 个需要修正的文件")
    print()
    
    fixed_count = 0
    for json_path, correct_type in specified_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            current_type = data['images'][0].get('content_type', '')
            
            if current_type != correct_type:
                if fix_json_annotation(json_path, correct_type):
                    fixed_count += 1
                    print(f"  修正: {json_path.name} ({current_type} -> {correct_type})")
        except Exception as e:
            print(f"  错误处理 {json_path.name}: {e}")
    
    print()
    print("="*70)
    print(f"修正完成！共修正 {fixed_count} 个文件")
    print("="*70)

if __name__ == '__main__':
    main()
