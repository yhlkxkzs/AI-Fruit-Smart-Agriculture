#!/usr/bin/env python3
"""
分析train/apple文件夹中PNG和JPG图片的区别
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import statistics

def analyze_image(img_path):
    """分析图片特征"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        file_size = img_path.stat().st_size
        
        info = {
            'size': img.size,
            'mode': img.mode,
            'file_size': file_size,
            'width': img.size[0],
            'height': img.size[1],
            'aspect_ratio': img.size[0] / img.size[1] if img.size[1] > 0 else 0,
        }
        
        if img.mode == 'RGB':
            info['avg_r'] = np.mean(img_array[:, :, 0])
            info['avg_g'] = np.mean(img_array[:, :, 1])
            info['avg_b'] = np.mean(img_array[:, :, 2])
            info['unique_colors'] = len(np.unique(img_array.reshape(-1, 3), axis=0))
        elif img.mode == 'L':
            info['avg_brightness'] = np.mean(img_array)
            info['unique_gray'] = len(np.unique(img_array))
        
        return info
    except Exception as e:
        return None

def get_json_info(json_path):
    """获取JSON文件中的信息"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return None
        
        img_info = data['images'][0]
        return {
            'content_type': img_info.get('content_type', 'unknown'),
            'image_style': img_info.get('image_style', 'unknown'),
            'source_dataset': img_info.get('source_dataset', 'unknown'),
            'is_multi_fruit': img_info.get('is_multi_fruit', False),
        }
    except:
        return None

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    print("="*80)
    print("分析 train/apple 文件夹中 PNG 和 JPG 图片的区别")
    print("="*80)
    print()
    
    # 收集PNG和JPG图片
    png_files = list(apple_dir.glob("*.png"))
    jpg_files = list(apple_dir.glob("*.jpg")) + list(apple_dir.glob("*.jpeg"))
    
    print(f"PNG图片数量: {len(png_files)}")
    print(f"JPG图片数量: {len(jpg_files)}")
    print()
    
    # 分析PNG图片
    print("="*80)
    print("PNG图片分析")
    print("="*80)
    
    png_stats = {
        'sizes': [],
        'file_sizes': [],
        'content_types': defaultdict(int),
        'image_styles': defaultdict(int),
        'source_datasets': defaultdict(int),
        'aspect_ratios': [],
        'unique_colors': [],
    }
    
    for img_path in png_files[:500]:  # 采样分析前500个
        img_info = analyze_image(img_path)
        if img_info:
            png_stats['sizes'].append(img_info['size'])
            png_stats['file_sizes'].append(img_info['file_size'])
            png_stats['aspect_ratios'].append(img_info['aspect_ratio'])
            if 'unique_colors' in img_info:
                png_stats['unique_colors'].append(img_info['unique_colors'])
        
        json_path = json_dir / f"{img_path.stem}.json"
        if json_path.exists():
            json_info = get_json_info(json_path)
            if json_info:
                png_stats['content_types'][json_info['content_type']] += 1
                png_stats['image_styles'][json_info['image_style']] += 1
                png_stats['source_datasets'][json_info['source_dataset']] += 1
    
    # 分析JPG图片
    print("="*80)
    print("JPG图片分析")
    print("="*80)
    
    jpg_stats = {
        'sizes': [],
        'file_sizes': [],
        'content_types': defaultdict(int),
        'image_styles': defaultdict(int),
        'source_datasets': defaultdict(int),
        'aspect_ratios': [],
        'unique_colors': [],
    }
    
    for img_path in jpg_files[:500]:  # 采样分析前500个
        img_info = analyze_image(img_path)
        if img_info:
            jpg_stats['sizes'].append(img_info['size'])
            jpg_stats['file_sizes'].append(img_info['file_size'])
            jpg_stats['aspect_ratios'].append(img_info['aspect_ratio'])
            if 'unique_colors' in img_info:
                jpg_stats['unique_colors'].append(img_info['unique_colors'])
        
        json_path = json_dir / f"{img_path.stem}.json"
        if json_path.exists():
            json_info = get_json_info(json_path)
            if json_info:
                jpg_stats['content_types'][json_info['content_type']] += 1
                jpg_stats['image_styles'][json_info['image_style']] += 1
                jpg_stats['source_datasets'][json_info['source_dataset']] += 1
    
    # 输出对比结果
    print()
    print("="*80)
    print("对比分析结果")
    print("="*80)
    print()
    
    # 文件大小对比
    if png_stats['file_sizes'] and jpg_stats['file_sizes']:
        print("文件大小对比：")
        print(f"  PNG平均: {statistics.mean(png_stats['file_sizes']) / 1024:.1f} KB")
        print(f"  JPG平均: {statistics.mean(jpg_stats['file_sizes']) / 1024:.1f} KB")
        print(f"  PNG中位数: {statistics.median(png_stats['file_sizes']) / 1024:.1f} KB")
        print(f"  JPG中位数: {statistics.median(jpg_stats['file_sizes']) / 1024:.1f} KB")
        print()
    
    # 尺寸对比
    if png_stats['sizes'] and jpg_stats['sizes']:
        print("图片尺寸对比：")
        png_widths = [s[0] for s in png_stats['sizes']]
        png_heights = [s[1] for s in png_stats['sizes']]
        jpg_widths = [s[0] for s in jpg_stats['sizes']]
        jpg_heights = [s[1] for s in jpg_stats['sizes']]
        
        print(f"  PNG平均尺寸: {statistics.mean(png_widths):.0f} x {statistics.mean(png_heights):.0f}")
        print(f"  JPG平均尺寸: {statistics.mean(jpg_widths):.0f} x {statistics.mean(jpg_heights):.0f}")
        print()
    
    # 内容类型分布
    print("内容类型分布：")
    print("  PNG:")
    for content_type, count in sorted(png_stats['content_types'].items()):
        print(f"    {content_type}: {count}")
    print("  JPG:")
    for content_type, count in sorted(jpg_stats['content_types'].items()):
        print(f"    {content_type}: {count}")
    print()
    
    # 图片风格分布
    print("图片风格分布：")
    print("  PNG:")
    for style, count in sorted(png_stats['image_styles'].items()):
        print(f"    {style}: {count}")
    print("  JPG:")
    for style, count in sorted(jpg_stats['image_styles'].items()):
        print(f"    {style}: {count}")
    print()
    
    # 来源数据集分布
    print("来源数据集分布：")
    print("  PNG:")
    for source, count in sorted(png_stats['source_datasets'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {source}: {count}")
    print("  JPG:")
    for source, count in sorted(jpg_stats['source_datasets'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {source}: {count}")
    print()
    
    # 颜色数量对比
    if png_stats['unique_colors'] and jpg_stats['unique_colors']:
        print("唯一颜色数量对比：")
        print(f"  PNG平均: {statistics.mean(png_stats['unique_colors']):.0f}")
        print(f"  JPG平均: {statistics.mean(jpg_stats['unique_colors']):.0f}")
        print()

if __name__ == '__main__':
    main()
