#!/usr/bin/env python3
"""
分析unknown目录中的图像，尝试识别类别
"""

import os
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict

# 图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG'}

# 关键词映射（用于从文件名或路径推断类别）
KEYWORD_MAPPING = {
    'apple': ['apple', 'apples'],
    'tomato': ['tomato', 'tomatoes'],
    'strawberry': ['strawberry', 'strawberries'],
    'cherry': ['cherry', 'cherries'],
    'grape': ['grape', 'grapes'],
    'orange': ['orange', 'oranges'],
    'banana': ['banana', 'bananas'],
    'pear': ['pear', 'pears'],
    'peach': ['peach', 'peaches'],
    'corn': ['corn', 'maize'],
    'cucumber': ['cucumber', 'cucumbers'],
    'pepper': ['pepper', 'peppers', 'capsicum', 'capsicums'],
    'lemon': ['lemon', 'lemons'],
    'pistachio': ['pistachio', 'pistachios'],
    'soybean': ['soybean', 'soybeans', 'soy'],
    'wheat': ['wheat'],
    'carrot': ['carrot', 'carrots'],
    'cassava': ['cassava', 'manioc'],
    'mango': ['mango', 'mangoes'],
    'avocado': ['avocado', 'avocados'],
    'rockmelon': ['rockmelon', 'rockmelons', 'cantaloupe'],
}

def analyze_image_path(image_path):
    """从图像路径和文件名推断可能的类别"""
    path_str = str(image_path).lower()
    name = image_path.name.lower()
    
    found_categories = []
    for category, keywords in KEYWORD_MAPPING.items():
        for keyword in keywords:
            if keyword in path_str or keyword in name:
                found_categories.append(category)
                break
    
    return found_categories

def get_image_info(image_path):
    """获取图像基本信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            return {
                'valid': True,
                'width': width,
                'height': height,
                'mode': mode,
                'size_kb': os.path.getsize(image_path) / 1024
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def analyze_unknown_directory(unknown_dir):
    """分析unknown目录"""
    print(f"\n{'='*80}")
    print(f"分析目录: {unknown_dir}")
    print(f"{'='*80}")
    
    if not os.path.exists(unknown_dir):
        print(f"❌ 目录不存在: {unknown_dir}")
        return None
    
    # 查找所有图像
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(unknown_dir).glob(f"*{ext}"))
        images.extend(Path(unknown_dir).glob(f"*{ext.upper()}"))
    
    print(f"找到 {len(images)} 张图像")
    
    if len(images) == 0:
        return None
    
    # 分析每张图像
    results = {
        'total': len(images),
        'valid_images': 0,
        'invalid_images': 0,
        'category_suggestions': defaultdict(int),
        'sample_images': []
    }
    
    # 分析前100张图像作为样本
    sample_size = min(100, len(images))
    print(f"\n分析前 {sample_size} 张图像作为样本...")
    
    for i, img_path in enumerate(images[:sample_size]):
        # 获取图像信息
        img_info = get_image_info(img_path)
        if img_info['valid']:
            results['valid_images'] += 1
        else:
            results['invalid_images'] += 1
            continue
        
        # 尝试推断类别
        categories = analyze_image_path(img_path)
        if categories:
            for cat in categories:
                results['category_suggestions'][cat] += 1
        
        # 保存样本信息
        if i < 20:  # 保存前20张的详细信息
            results['sample_images'].append({
                'path': str(img_path),
                'name': img_path.name,
                'suggested_categories': categories,
                'info': img_info
            })
    
    # 打印结果
    print(f"\n📊 分析结果:")
    print(f"  有效图像: {results['valid_images']}")
    print(f"  无效图像: {results['invalid_images']}")
    
    if results['category_suggestions']:
        print(f"\n🎯 可能的类别建议:")
        sorted_cats = sorted(results['category_suggestions'].items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats:
            percentage = (count / sample_size) * 100
            print(f"  {cat}: {count} 张 ({percentage:.1f}%)")
    
    print(f"\n📸 样本图像 (前10张):")
    for i, sample in enumerate(results['sample_images'][:10], 1):
        print(f"  {i}. {sample['name']}")
        if sample['suggested_categories']:
            print(f"     → 可能类别: {', '.join(sample['suggested_categories'])}")
        else:
            print(f"     → 无法从文件名推断类别")
    
    return results

def main():
    """主函数"""
    print("="*80)
    print("🔍 Unknown图像类别分析工具")
    print("="*80)
    
    base_dir = Path("/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset")
    
    unknown_dirs = [
        base_dir / "train" / "unknown",
        base_dir / "val" / "unknown",
        base_dir / "test" / "unknown"
    ]
    
    all_results = {}
    
    for unknown_dir in unknown_dirs:
        results = analyze_unknown_directory(unknown_dir)
        if results:
            all_results[str(unknown_dir)] = results
    
    # 保存结果
    output_file = base_dir.parent / "unknown_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析结果已保存到: {output_file}")
    
    # 汇总统计
    print("\n" + "="*80)
    print("📊 汇总统计")
    print("="*80)
    
    total_valid = sum(r['valid_images'] for r in all_results.values())
    total_invalid = sum(r['invalid_images'] for r in all_results.values())
    
    print(f"总有效图像: {total_valid}")
    print(f"总无效图像: {total_invalid}")
    
    # 合并所有类别建议
    all_categories = defaultdict(int)
    for results in all_results.values():
        for cat, count in results['category_suggestions'].items():
            all_categories[cat] += count
    
    if all_categories:
        print(f"\n🎯 总体类别建议:")
        sorted_cats = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats:
            print(f"  {cat}: {count} 张")

if __name__ == '__main__':
    main()
