#!/usr/bin/env python3
"""
训练时过滤多水果组合图片，只保留单一水果图片
用于数据加载器，确保训练数据质量
"""

import json
from pathlib import Path

def filter_single_fruit_images(dataset_dir, split='train', category='apricot'):
    """
    过滤出单一水果图片（排除多水果组合图片）
    
    Args:
        dataset_dir: 数据集根目录
        split: 数据集划分 (train/val/test)
        category: 水果类别
    
    Returns:
        list: 单一水果图片路径列表
    """
    category_dir = Path(dataset_dir) / split / category
    json_dir = category_dir / 'json'
    
    if not json_dir.exists():
        return []
    
    single_fruit_images = []
    
    for json_file in json_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否为多水果组合图片
            is_multi_fruit = False
            
            if 'images' in data and len(data['images']) > 0:
                img_info = data['images'][0]
                is_multi_fruit = img_info.get('is_multi_fruit', False)
            
            # 如果不是多水果组合，添加到列表
            if not is_multi_fruit:
                # 获取对应的图片路径
                img_name = img_info.get('file_name', json_file.stem)
                img_path = category_dir / img_name
                
                if img_path.exists():
                    single_fruit_images.append(str(img_path))
        
        except Exception as e:
            print(f"警告: 无法读取 {json_file.name}: {e}")
    
    return single_fruit_images

def get_statistics(dataset_dir):
    """统计单一水果和多水果组合图片的数量"""
    stats = {}
    
    for split in ['train', 'val', 'test']:
        for category_dir in (Path(dataset_dir) / split).iterdir():
            if not category_dir.is_dir() or category_dir.name in ['json', 'csv', 'sets']:
                continue
            
            category = category_dir.name
            json_dir = category_dir / 'json'
            
            if not json_dir.exists():
                continue
            
            single_count = 0
            multi_count = 0
            
            for json_file in json_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'images' in data and len(data['images']) > 0:
                        is_multi = data['images'][0].get('is_multi_fruit', False)
                        if is_multi:
                            multi_count += 1
                        else:
                            single_count += 1
                except:
                    pass
            
            if single_count > 0 or multi_count > 0:
                key = f"{split}/{category}"
                stats[key] = {
                    'single_fruit': single_count,
                    'multi_fruit': multi_count,
                    'total': single_count + multi_count
                }
    
    return stats

def main():
    """主函数"""
    import sys
    
    dataset_dir = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        # 显示统计信息
        print("="*60)
        print("数据集统计（单一水果 vs 多水果组合）")
        print("="*60)
        
        stats = get_statistics(dataset_dir)
        
        for key, values in sorted(stats.items()):
            single = values['single_fruit']
            multi = values['multi_fruit']
            total = values['total']
            multi_pct = (multi / total * 100) if total > 0 else 0
            
            print(f"\n{key}:")
            print(f"  单一水果: {single:5d} 张")
            print(f"  多水果组合: {multi:5d} 张 ({multi_pct:.1f}%)")
            print(f"  总计: {total:5d} 张")
    else:
        # 示例：过滤apricot的单一水果图片
        print("="*60)
        print("过滤单一水果图片示例")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            single_images = filter_single_fruit_images(dataset_dir, split, 'apricot')
            print(f"\n{split.upper()}/apricot:")
            print(f"  单一水果图片: {len(single_images)} 张")
            
            if len(single_images) > 0:
                print(f"  示例: {single_images[0]}")

if __name__ == '__main__':
    main()
