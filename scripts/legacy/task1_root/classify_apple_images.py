#!/usr/bin/env python3
"""
分类苹果图像：删除黑白照片，区分苹果花和苹果果实，更新JSON标注
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

def is_binary_image(img_path):
    """检查是否为黑白二值图像"""
    try:
        img = Image.open(img_path)
        mode = img.mode
        
        # 模式'1'是1位像素（黑白二值图）
        if mode == '1':
            return True
        
        # 检查灰度图是否只有黑白值
        if mode in ['L', 'LA']:
            pixels = np.array(img)
            unique_values = np.unique(pixels)
            if len(unique_values) <= 2 and (0 in unique_values or 255 in unique_values):
                return True
        
        return False
    except Exception as e:
        print(f"  错误检查 {img_path.name}: {e}")
        return False

def classify_apple_image(img_path):
    """
    分类苹果图像：花或果实
    使用简单的启发式方法：
    - 检查文件名中是否包含'flower'、'blossom'、'bloom'等关键词
    - 检查图像的主要颜色（花通常是白色/粉色，果实通常是红色/绿色）
    - 检查图像的平均亮度（花通常较亮）
    """
    # 检查文件名
    name_lower = img_path.stem.lower()
    if any(keyword in name_lower for keyword in ['flower', 'blossom', 'bloom', '花']):
        return 'flower'
    
    try:
        img = Image.open(img_path)
        
        # 转换为RGB（如果不是）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # 计算平均RGB值
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # 计算平均亮度
        avg_brightness = (avg_r + avg_g + avg_b) / 3
        
        # 计算颜色饱和度（简单方法）
        max_channel = max(avg_r, avg_g, avg_b)
        min_channel = min(avg_r, avg_g, avg_b)
        saturation = (max_channel - min_channel) / max_channel if max_channel > 0 else 0
        
        # 启发式规则：
        # 1. 如果平均亮度很高（>200）且饱和度较低（<0.3），可能是花（白色/浅色）
        # 2. 如果红色通道明显高于其他通道，可能是红色果实
        # 3. 如果绿色通道明显高于其他通道，可能是绿色果实
        # 4. 否则，如果亮度较高，可能是花；如果亮度较低，可能是果实
        
        if avg_brightness > 200 and saturation < 0.3:
            return 'flower'  # 白色/浅色花
        elif avg_r > avg_g + 30 and avg_r > avg_b + 30:
            return 'fruit'  # 红色果实
        elif avg_g > avg_r + 30 and avg_g > avg_b + 30:
            return 'fruit'  # 绿色果实
        elif avg_brightness > 180:
            return 'flower'  # 较亮，可能是花
        else:
            return 'fruit'  # 默认是果实
            
    except Exception as e:
        print(f"  错误分类 {img_path.name}: {e}")
        return 'fruit'  # 默认返回果实

def update_json_annotation(json_path, subcategory):
    """更新JSON标注文件，添加子类别信息"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新category信息，添加subcategory字段
        if 'categories' in data and len(data['categories']) > 0:
            category = data['categories'][0]
            category['subcategory'] = subcategory
            category['name'] = f"apple_{subcategory}"  # 例如: apple_flower, apple_fruit
        
        # 更新annotation，添加subcategory信息
        if 'annotations' in data and len(data['annotations']) > 0:
            annotation = data['annotations'][0]
            annotation['subcategory'] = subcategory
        
        # 保存更新后的JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"  错误更新JSON {json_path.name}: {e}")
        return False

def process_apple_directory(apple_dir):
    """处理apple目录"""
    apple_dir = Path(apple_dir)
    if not apple_dir.exists():
        print(f"错误: 目录不存在 {apple_dir}")
        return
    
    print(f"处理目录: {apple_dir}")
    print("="*60)
    
    # 获取所有图像文件
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']:
        images.extend(list(apple_dir.glob(ext)))
    
    print(f"总图像数: {len(images)}")
    
    # 1. 识别并删除黑白二值图像
    binary_images = []
    for img_path in images:
        if is_binary_image(img_path):
            binary_images.append(img_path)
    
    print(f"\n发现 {len(binary_images)} 个黑白二值图像（将删除）")
    
    deleted_count = 0
    for img_path in binary_images:
        # 删除图像文件
        try:
            img_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"  错误删除图像 {img_path.name}: {e}")
        
        # 删除对应的JSON文件
        json_path = apple_dir / 'json' / f"{img_path.stem}.json"
        if json_path.exists():
            try:
                json_path.unlink()
            except Exception as e:
                print(f"  错误删除JSON {json_path.name}: {e}")
        
        # 删除对应的CSV文件
        csv_path = apple_dir / 'csv' / f"{img_path.stem}.csv"
        if csv_path.exists():
            try:
                csv_path.unlink()
            except Exception as e:
                print(f"  错误删除CSV {csv_path.name}: {e}")
    
    print(f"已删除 {deleted_count} 个黑白图像及其标注文件")
    
    # 2. 分类剩余图像（花或果实）
    remaining_images = [img for img in images if img not in binary_images]
    
    # 按stem去重，每个图像只处理一次（可能有多个格式版本）
    processed_stems = set()
    images_to_process = []
    for img_path in remaining_images:
        if img_path.stem not in processed_stems:
            processed_stems.add(img_path.stem)
            images_to_process.append(img_path)
    
    print(f"\n分类剩余 {len(images_to_process)} 个图像（去重后）...")
    
    flower_count = 0
    fruit_count = 0
    skipped_count = 0
    
    for img_path in images_to_process:
        # 检查JSON文件是否存在
        json_path = apple_dir / 'json' / f"{img_path.stem}.json"
        if not json_path.exists():
            skipped_count += 1
            continue
        
        # 分类图像
        subcategory = classify_apple_image(img_path)
        
        # 更新JSON文件
        if update_json_annotation(json_path, subcategory):
            if subcategory == 'flower':
                flower_count += 1
            else:
                fruit_count += 1
    
    print(f"\n分类结果:")
    print(f"  苹果花: {flower_count}")
    print(f"  苹果果实: {fruit_count}")
    print(f"  跳过（无JSON）: {skipped_count}")
    print(f"  总计: {flower_count + fruit_count}")
    
    print(f"\n完成！")

if __name__ == '__main__':
    import sys
    
    # 默认处理test/apple目录
    if len(sys.argv) > 1:
        apple_dir = sys.argv[1]
    else:
        apple_dir = '/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset/test/apple'
    
    process_apple_directory(apple_dir)
