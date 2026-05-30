#!/usr/bin/env python3
"""
快速识别并删除用于识别位置的辅助图片
基于文件大小和颜色数量快速筛选
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def is_position_marker_quick(img_path):
    """快速判断是否是位置标记图片"""
    try:
        file_size = img_path.stat().st_size
        
        # 快速筛选：文件大于50KB的跳过
        if file_size > 50000:
            return False, None
        
        img = Image.open(img_path)
        
        # 如果文件很小（<5KB），直接检查
        if file_size < 5000:
            img_array = np.array(img)
            
            if img.mode == 'RGB':
                # 采样检查颜色数量
                h, w = img_array.shape[:2]
                step = max(1, min(h, w) // 50)  # 采样
                sampled = img_array[::step, ::step].reshape(-1, 3)
                unique_colors = len(np.unique(sampled, axis=0))
                
                if unique_colors <= 20:
                    return True, f"文件小({file_size} bytes)且颜色少({unique_colors})"
            
            elif img.mode == 'L':
                h, w = img_array.shape
                step = max(1, min(h, w) // 50)
                sampled = img_array[::step, ::step]
                unique_gray = len(np.unique(sampled))
                
                if unique_gray <= 20:
                    return True, f"文件小({file_size} bytes)且灰度值少({unique_gray})"
        
        # 对于稍大的文件，检查颜色数量
        elif file_size < 50000:
            img_array = np.array(img)
            
            if img.mode == 'RGB':
                h, w = img_array.shape[:2]
                step = max(1, min(h, w) // 100)  # 更稀疏的采样
                sampled = img_array[::step, ::step].reshape(-1, 3)
                unique_colors = len(np.unique(sampled, axis=0))
                
                if unique_colors <= 10:
                    return True, f"颜色数过少({unique_colors})"
            
            elif img.mode == 'L':
                h, w = img_array.shape
                step = max(1, min(h, w) // 100)
                sampled = img_array[::step, ::step]
                unique_gray = len(np.unique(sampled))
                
                if unique_gray <= 10:
                    return True, f"灰度值过少({unique_gray})"
        
        return False, None
        
    except Exception as e:
        return False, None

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    print("="*70)
    print("快速识别并删除用于识别位置的辅助图片")
    print("="*70)
    print(f"图片目录: {apple_dir}")
    print()
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(apple_dir.glob(f"*{ext}")))
        image_files.extend(list(apple_dir.glob(f"*{ext.upper()}")))
    
    total_files = len(image_files)
    print(f"找到 {total_files} 个图片文件")
    print("开始快速识别...")
    print()
    
    markers_to_delete = []
    
    for img_path in tqdm(image_files, desc="分析进度"):
        is_marker, reason = is_position_marker_quick(img_path)
        if is_marker:
            markers_to_delete.append((img_path, reason))
    
    print()
    print("="*70)
    print(f"识别完成！找到 {len(markers_to_delete)} 个可能是位置标记的图片")
    print("="*70)
    print()
    
    if len(markers_to_delete) > 0:
        print("待删除的文件（前30个）：")
        for img_path, reason in markers_to_delete[:30]:
            print(f"  - {img_path.name}: {reason}")
        
        if len(markers_to_delete) > 30:
            print(f"  ... 还有 {len(markers_to_delete) - 30} 个文件")
        
        print()
        print("开始删除文件和对应的JSON文件...")
        
        deleted_images = 0
        deleted_jsons = 0
        
        for img_path, reason in tqdm(markers_to_delete, desc="删除进度"):
            # 删除图片文件
            try:
                img_path.unlink()
                deleted_images += 1
            except Exception as e:
                print(f"  错误删除 {img_path.name}: {e}")
            
            # 删除对应的JSON文件
            json_path = json_dir / f"{img_path.stem}.json"
            if json_path.exists():
                try:
                    json_path.unlink()
                    deleted_jsons += 1
                except Exception as e:
                    print(f"  错误删除 {json_path.name}: {e}")
        
        print()
        print("="*70)
        print("删除完成！")
        print("="*70)
        print(f"删除图片文件: {deleted_images}")
        print(f"删除JSON文件: {deleted_jsons}")
        print("="*70)
    else:
        print("未找到需要删除的位置标记图片")

if __name__ == '__main__':
    main()
