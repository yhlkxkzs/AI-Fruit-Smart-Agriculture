#!/usr/bin/env python3
"""
改进的苹果图片分类器
使用更准确的图像分析方法来区分苹果花、果实和叶子
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

def analyze_apple_image_detailed(img_path):
    """
    详细分析苹果图片，返回详细的特征信息
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        height, width = img.shape[:2]
        
        # 1. 颜色分析
        avg_r = np.mean(img_rgb[:, :, 0])
        avg_g = np.mean(img_rgb[:, :, 1])
        avg_b = np.mean(img_rgb[:, :, 2])
        
        avg_h = np.mean(img_hsv[:, :, 0])
        avg_s = np.mean(img_hsv[:, :, 1])
        avg_v = np.mean(img_hsv[:, :, 2])
        
        # 2. 红色区域检测（成熟果实）
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 3. 粉色/白色区域检测（花）
        # 花的颜色通常是粉色、白色或淡粉色
        pink_lower1 = np.array([140, 30, 150])
        pink_upper1 = np.array([170, 100, 255])
        pink_mask1 = cv2.inRange(img_hsv, pink_lower1, pink_upper1)
        
        # 白色/淡色区域（花也可能是白色）
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(img_hsv, white_lower, white_upper)
        
        pink_white_mask = cv2.bitwise_or(pink_mask1, white_mask)
        pink_white_ratio = np.sum(pink_white_mask > 0) / (width * height)
        
        # 4. 绿色区域检测（叶子/未成熟果实）
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        # 5. 亮度分析
        brightness = avg_v
        
        # 6. 饱和度分析
        saturation = avg_s
        
        # 7. 边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # 8. 纹理分析
        texture_variance = np.var(gray)
        
        # 9. 形状分析 - 使用轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
        else:
            circularity = 0
        
        # 10. 颜色分布分析
        # 计算主要颜色的分布
        unique_colors = len(np.unique(img_rgb.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (width * height)
        
        return {
            'avg_rgb': (avg_r, avg_g, avg_b),
            'avg_hsv': (avg_h, avg_s, avg_v),
            'red_ratio': red_ratio,
            'pink_white_ratio': pink_white_ratio,
            'green_ratio': green_ratio,
            'brightness': brightness,
            'saturation': saturation,
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'circularity': circularity,
            'color_diversity': color_diversity,
            'size': (width, height)
        }
        
    except Exception as e:
        print(f"  错误分析图片 {img_path.name}: {e}")
        return None

def classify_apple_image_improved(img_path, features):
    """
    基于详细特征改进的分类逻辑
    """
    if features is None:
        return None
    
    red_ratio = features['red_ratio']
    pink_white_ratio = features['pink_white_ratio']
    green_ratio = features['green_ratio']
    brightness = features['brightness']
    saturation = features['saturation']
    edge_density = features['edge_density']
    texture_variance = features['texture_variance']
    circularity = features['circularity']
    avg_r, avg_g, avg_b = features['avg_rgb']
    
    # 文件名关键词
    filename = img_path.name.lower()
    has_flower_keyword = any(kw in filename for kw in ['flower', 'blossom', 'bloom', '花'])
    has_fruit_keyword = any(kw in filename for kw in ['fruit', 'apple', '果实', '果'])
    has_leaf_keyword = any(kw in filename for kw in ['leaf', 'leaves', '叶', '叶子'])
    
    score_flower = 0
    score_fruit = 0
    score_leaf = 0
    
    # ===== 花的特征 =====
    # 花通常是粉色、白色或淡色，亮度较高，细节丰富
    if pink_white_ratio > 0.2:
        score_flower += 5
    elif pink_white_ratio > 0.1:
        score_flower += 3
    elif pink_white_ratio > 0.05:
        score_flower += 1
    
    if brightness > 150:
        score_flower += 2
    elif brightness > 120:
        score_flower += 1
    
    # 花通常有丰富的细节（边缘密度高）
    if edge_density > 0.15:
        score_flower += 2
    elif edge_density > 0.10:
        score_flower += 1
    
    # 花的饱和度通常中等
    if 30 < saturation < 100:
        score_flower += 1
    
    # 花通常不是圆形（圆形度低）
    if circularity < 0.5:
        score_flower += 1
    
    if has_flower_keyword:
        score_flower += 3
    
    # 如果红色区域很少，但粉色/白色区域多，很可能是花
    if red_ratio < 0.1 and pink_white_ratio > 0.15:
        score_flower += 3
    
    # ===== 果实的特征 =====
    # 成熟果实通常是红色
    if red_ratio > 0.25:
        score_fruit += 6
    elif red_ratio > 0.15:
        score_fruit += 4
    elif red_ratio > 0.08:
        score_fruit += 2
    
    # 未成熟的绿色果实
    if green_ratio > 0.3 and red_ratio < 0.1:
        score_fruit += 3
    elif green_ratio > 0.2 and red_ratio < 0.05:
        score_fruit += 2
    
    # 果实通常较圆（圆形度高）
    if circularity > 0.6:
        score_fruit += 2
    elif circularity > 0.4:
        score_fruit += 1
    
    # 果实的边缘密度通常较低（形状较圆润）
    if edge_density < 0.12 and red_ratio > 0.1:
        score_fruit += 2
    
    # 果实的饱和度通常较高
    if saturation > 80 and red_ratio > 0.05:
        score_fruit += 2
    
    if has_fruit_keyword and not has_flower_keyword:
        score_fruit += 2
    
    # 如果红色区域多，但粉色/白色区域少，很可能是果实
    if red_ratio > 0.15 and pink_white_ratio < 0.1:
        score_fruit += 3
    
    # ===== 叶子的特征 =====
    # 叶子通常是绿色
    if green_ratio > 0.4:
        score_leaf += 5
    elif green_ratio > 0.3:
        score_leaf += 3
    elif green_ratio > 0.2:
        score_leaf += 1
    
    # 叶子通常不是红色或粉色
    if red_ratio < 0.05 and pink_white_ratio < 0.05:
        score_leaf += 2
    
    # 叶子通常有中等边缘密度（有叶脉）
    if 0.08 < edge_density < 0.15 and green_ratio > 0.2:
        score_leaf += 2
    
    # 叶子的饱和度通常中等偏高
    if 60 < saturation < 120 and green_ratio > 0.2:
        score_leaf += 1
    
    if has_leaf_keyword:
        score_leaf += 3
    
    # 如果绿色区域多，但红色和粉色区域都很少，很可能是叶子
    if green_ratio > 0.3 and red_ratio < 0.05 and pink_white_ratio < 0.05:
        score_leaf += 3
    
    # 决定最终类型
    if score_fruit > score_flower and score_fruit > score_leaf:
        return 'fruit'
    elif score_leaf > score_flower and score_leaf > score_fruit:
        return 'leaf'
    else:
        return 'flower'  # 默认或平局时返回flower

def verify_and_fix_json(json_path, img_path, detected_type):
    """
    验证并修正JSON文件
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False, None
        
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
            
            return True, (current_type, detected_type)
        
        return False, None
        
    except Exception as e:
        print(f"  错误处理 {json_path.name}: {e}")
        return False, None

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    if not json_dir.exists():
        print(f"错误：{json_dir} 不存在")
        return
    
    print("="*70)
    print("改进的苹果图片分类和修正")
    print("="*70)
    print(f"JSON文件目录: {json_dir}")
    print()
    
    json_files = sorted(list(json_dir.glob('*.json')))
    total_files = len(json_files)
    
    print(f"找到 {total_files} 个JSON文件")
    print("开始详细分析和修正...")
    print()
    
    fixed_count = 0
    error_count = 0
    unchanged_count = 0
    
    stats = {
        'flower_to_fruit': 0,
        'flower_to_leaf': 0,
        'fruit_to_flower': 0,
        'fruit_to_leaf': 0,
        'leaf_to_flower': 0,
        'leaf_to_fruit': 0,
    }
    
    for json_path in tqdm(json_files, desc="处理进度"):
        img_stem = json_path.stem
        img_files = list(apple_dir.glob(f"{img_stem}.*"))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            error_count += 1
            continue
        
        img_path = img_files[0]
        
        # 详细分析图片
        features = analyze_apple_image_detailed(img_path)
        if features is None:
            error_count += 1
            continue
        
        # 分类
        detected_type = classify_apple_image_improved(img_path, features)
        if detected_type is None:
            error_count += 1
            continue
        
        # 验证和修正
        fixed, change = verify_and_fix_json(json_path, img_path, detected_type)
        
        if fixed and change:
            old_type, new_type = change
            change_key = f"{old_type}_to_{new_type}"
            if change_key in stats:
                stats[change_key] += 1
            
            fixed_count += 1
            if fixed_count <= 20:  # 显示前20个修正
                print(f"  修正: {json_path.name} ({old_type} -> {new_type})")
        elif fixed:
            fixed_count += 1
        elif not fixed:
            unchanged_count += 1
        else:
            error_count += 1
    
    print()
    print("="*70)
    print("处理完成！")
    print("="*70)
    print(f"总文件数: {total_files}")
    print(f"修正文件数: {fixed_count}")
    print(f"未变更文件数: {unchanged_count}")
    print(f"错误文件数: {error_count}")
    print()
    print("修正统计：")
    for change_type, count in stats.items():
        if count > 0:
            print(f"  - {change_type}: {count}")
    print("="*70)

if __name__ == '__main__':
    main()
