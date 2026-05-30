#!/usr/bin/env python3
"""
列出所有JPG格式中标注为油画的图片
"""

import json
from pathlib import Path

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    # 获取所有JPG文件
    jpg_files = list(apple_dir.glob('*.jpg')) + list(apple_dir.glob('*.jpeg'))
    
    painting_jpgs = []
    
    print("="*80)
    print("查找JPG格式中标注为油画的图片")
    print("="*80)
    print()
    
    for img_path in jpg_files:
        json_path = json_dir / f'{img_path.stem}.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'images' in data and len(data['images']) > 0:
                    img_info = data['images'][0]
                    if img_info.get('image_style') == 'painting':
                        # 获取更多信息
                        content_type = img_info.get('content_type', 'unknown')
                        source = img_info.get('source_dataset', 'unknown')
                        painting_jpgs.append({
                            'path': str(img_path),
                            'name': img_path.name,
                            'content_type': content_type,
                            'source': source
                        })
            except Exception as e:
                pass
    
    print(f"找到 {len(painting_jpgs)} 个JPG格式的油画图片")
    print()
    
    # 按内容类型统计
    content_stats = {}
    for item in painting_jpgs:
        ct = item['content_type']
        content_stats[ct] = content_stats.get(ct, 0) + 1
    
    print("按内容类型统计：")
    for ct, count in sorted(content_stats.items()):
        print(f"  {ct}: {count}")
    print()
    
    # 输出文件列表
    print("="*80)
    print("文件列表：")
    print("="*80)
    for item in painting_jpgs:
        print(f"{item['path']}")
    
    # 保存到文件
    output_file = Path('/tmp/painting_jpgs_list.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in painting_jpgs:
            f.write(f"{item['path']}\n")
    
    print()
    print("="*80)
    print(f"文件列表已保存到: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
