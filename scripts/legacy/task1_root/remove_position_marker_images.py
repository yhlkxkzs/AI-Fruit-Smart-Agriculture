#!/usr/bin/env python3
"""
识别并删除用于识别位置的辅助图片
这些图片通常特征：
1. 颜色数量很少（单色或接近单色）
2. 文件很小
3. 可能是灰度图或只有很少颜色的标注图
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

def is_position_marker_image(img_path):
    """
    判断是否是用于识别位置的辅助图片
    返回: (is_marker, reason)
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # 获取文件大小
        file_size = img_path.stat().st_size
        
        # 先检查文件大小（快速筛选）
        if file_size > 50000:  # 大于50KB，不太可能是标注图
            return False, None
        
        # 1. 检查颜色数量
        if img.mode == 'RGB':
            # 使用采样来加速（只检查部分像素）
            h, w = img_array.shape[:2]
            sample_size = min(10000, h * w)
            step = max(1, (h * w) // sample_size)
            sampled = img_array[::step, ::step].reshape(-1, 3)
            unique_colors = len(np.unique(sampled, axis=0))
            
            # 如果颜色数量很少（<= 10），可能是标注图
            if unique_colors <= 10:
                return True, f"颜色数过少({unique_colors})"
            
            # 检查是否是灰度图（RGB模式但只有灰度值）
            if np.allclose(img_array[:,:,0], img_array[:,:,1], atol=5) and np.allclose(img_array[:,:,1], img_array[:,:,2], atol=5):
                # 灰度图，检查亮度分布（采样）
                gray_values = img_array[::step, ::step, 0]
                unique_gray = len(np.unique(gray_values))
                if unique_gray <= 10:
                    return True, f"灰度图颜色数过少({unique_gray})"
        
        elif img.mode == 'L':
            # 灰度图（采样）
            h, w = img_array.shape
            sample_size = min(10000, h * w)
            step = max(1, (h * w) // sample_size)
            sampled = img_array[::step, ::step]
            unique_gray = len(np.unique(sampled))
            if unique_gray <= 10:
                return True, f"灰度图颜色数过少({unique_gray})"
        
        # 2. 检查文件大小（标注图通常很小）
        if file_size < 5000:  # 小于5KB
            # 进一步检查是否是单色或接近单色
            if img.mode == 'RGB':
                h, w = img_array.shape[:2]
                sample_size = min(5000, h * w)
                step = max(1, (h * w) // sample_size)
                sampled = img_array[::step, ::step].reshape(-1, 3)
                unique_colors = len(np.unique(sampled, axis=0))
                if unique_colors <= 20:
                    return True, f"文件小({file_size} bytes)且颜色少({unique_colors})"
            elif img.mode == 'L':
                h, w = img_array.shape
                sample_size = min(5000, h * w)
                step = max(1, (h * w) // sample_size)
                sampled = img_array[::step, ::step]
                unique_gray = len(np.unique(sampled))
                if unique_gray <= 20:
                    return True, f"文件小({file_size} bytes)且灰度值少({unique_gray})"
        
        # 3. 检查是否是纯色或接近纯色（标准差很小）
        if img.mode == 'RGB':
            std_r = np.std(img_array[:,:,0])
            std_g = np.std(img_array[:,:,1])
            std_b = np.std(img_array[:,:,2])
            if std_r < 10 and std_g < 10 and std_b < 10:
                return True, f"颜色变化很小(std: {std_r:.1f}, {std_g:.1f}, {std_b:.1f})"
        elif img.mode == 'L':
            std = np.std(img_array)
            if std < 10:
                return True, f"灰度变化很小(std: {std:.1f})"
        
        return False, None
        
    except Exception as e:
        # 如果无法读取，可能是损坏的图片
        return False, f"读取错误: {e}"

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    print("="*70)
    print("识别并删除用于识别位置的辅助图片")
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
    print("开始识别...")
    print()
    
    markers_to_delete = []
    
    for img_path in tqdm(image_files, desc="分析进度"):
        is_marker, reason = is_position_marker_image(img_path)
        if is_marker:
            markers_to_delete.append((img_path, reason))
    
    print()
    print("="*70)
    print(f"识别完成！找到 {len(markers_to_delete)} 个可能是位置标记的图片")
    print("="*70)
    print()
    
    if len(markers_to_delete) > 0:
        print("待删除的文件（前20个）：")
        for img_path, reason in markers_to_delete[:20]:
            print(f"  - {img_path.name}: {reason}")
        
        if len(markers_to_delete) > 20:
            print(f"  ... 还有 {len(markers_to_delete) - 20} 个文件")
        
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
