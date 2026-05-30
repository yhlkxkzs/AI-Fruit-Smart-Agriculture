#!/usr/bin/env python3
"""
综合修正标注脚本
1. 修正 is_multi_fruit 标注：只有多种水果才为 true，同一种水果多个为 false
2. 重新验证和修正苹果图片分类（花/叶/果实）
3. 修正香蕉图片风格标注（真实/油画/卡通等）
4. 修正蓝莓图片分类（叶子不应标记为果实）
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import re

def detect_image_style_improved(img_path):
    """
    改进的图片风格检测
    更准确地识别真实照片、油画、卡通、素描等
    """
    try:
        img = Image.open(img_path)
        
        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        width, height = img.size
        
        # 转换为OpenCV格式（BGR）
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. 边缘检测 - 计算边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # 2. 颜色分析
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # 计算颜色数量（唯一颜色数量）
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / (width * height)
        
        # 3. 纹理分析 - 使用拉普拉斯算子计算纹理复杂度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # 4. 亮度分布
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 5. 检查是否为灰度图（可能是素描）
        if img.mode == 'L' or (saturation < 0.1 and brightness_std < 30):
            return 'sketch'
        
        # 6. 尺寸判断（1024x1024通常是合成图片）
        if width == 1024 and height == 1024:
            # 进一步分析
            if texture_variance < 100:
                return 'painting'  # 模糊或平滑，可能是油画
            elif texture_variance > 1000:
                return 'sketch'  # 锐利，细节多，可能是素描
            elif unique_colors < 500:
                return 'cartoon'  # 颜色少，可能是卡通
            else:
                return 'real'  # 默认真实
        
        # 7. 启发式规则判断风格
        # 油画特征：边缘密度高，纹理方差中等，颜色饱和度较高
        if edge_density > 0.15 and 0.3 < saturation < 0.7 and 100 < texture_variance < 1000:
            return 'painting'
        
        # 卡通特征：颜色饱和度很高，颜色多样性低，边缘清晰
        if saturation > 0.7 and color_diversity < 0.1 and edge_density > 0.1:
            return 'cartoon'
        
        # 真实照片特征：纹理方差高，颜色多样性高，边缘密度中等
        if texture_variance > 500 and color_diversity > 0.15:
            return 'real'
        
        # 默认判断
        if texture_variance > 300:
            return 'real'
        elif saturation > 0.6:
            return 'cartoon'
        else:
            return 'real'  # 默认真实照片
            
    except Exception as e:
        print(f"  错误检测风格 {img_path.name}: {e}")
        return 'real'

def classify_apple_image_improved(img_path):
    """
    改进的苹果图片分类
    使用更准确的图像分析方法区分花、叶、果实
    """
    try:
        # 1. 文件名关键词检查（最可靠）
        name_lower = img_path.stem.lower()
        if any(keyword in name_lower for keyword in ['flower', 'blossom', 'bloom', '花', 'blooming']):
            return 'flower'
        if any(keyword in name_lower for keyword in ['leaf', 'leaves', '叶', '叶子', 'foliage']):
            return 'leaf'
        
        # 2. 打开图片并转换为RGB
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # 3. 颜色分析
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        avg_brightness = (avg_r + avg_g + avg_b) / 3
        
        # HSV分析
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1]) / 255.0
        v_mean = np.mean(hsv[:, :, 2]) / 255.0
        
        # 4. 边缘检测和轮廓分析
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 5. 轮廓检测（用于形状分析）
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
        else:
            circularity = 0
            solidity = 0
        
        # 6. 纹理分析
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # 7. 颜色分布分析
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50)
        green_ratio = np.sum(green_mask) / (height * width)
        
        red_mask1 = (hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)
        red_mask2 = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        red_mask = red_mask1 & red_mask2
        red_ratio = np.sum(red_mask) / (height * width)
        
        # 8. 综合判断规则
        
        # 叶子判断：绿色为主
        if green_ratio > 0.3 and avg_g > avg_r + 30 and avg_g > avg_b + 30:
            return 'leaf'
        
        # 花朵判断（多个条件）
        flower_score = 0
        
        if avg_brightness > 180 and s_mean < 0.4:
            flower_score += 2
        if avg_r > 150 and avg_g > 150 and avg_b > 120:
            flower_score += 2
        if edge_density > 0.08:
            flower_score += 1
        if circularity < 0.6 and solidity < 0.85:
            flower_score += 1
        if 100 < texture_variance < 800:
            flower_score += 1
        
        if flower_score >= 4:
            return 'flower'
        
        # 果实判断
        fruit_score = 0
        
        if red_ratio > 0.2:
            fruit_score += 2
        if avg_r > avg_g + 20 and avg_r > avg_b + 20:
            fruit_score += 2
        if circularity > 0.6:
            fruit_score += 1
        if solidity > 0.85:
            fruit_score += 1
        if avg_g > avg_r + 20 and avg_g > avg_b + 20 and green_ratio > 0.2:
            fruit_score += 2
        
        if fruit_score >= 3:
            return 'fruit'
        
        # 默认判断
        if flower_score > fruit_score:
            return 'flower'
        elif green_ratio > 0.2:
            return 'leaf'
        else:
            return 'fruit'
            
    except Exception as e:
        print(f"  错误分类 {img_path.name}: {e}")
        return 'fruit'

def classify_blueberry_image(img_path):
    """
    分类蓝莓图片：区分果实和叶子
    """
    try:
        # 文件名关键词检查
        name_lower = img_path.stem.lower()
        if any(keyword in name_lower for keyword in ['leaf', 'leaves', '叶', '叶子', 'foliage']):
            return 'leaf'
        
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 颜色分析
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # 计算绿色像素比例（叶子）
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50)
        green_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # 计算蓝色/紫色像素比例（蓝莓果实）
        blue_purple_mask = ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 130)) & (hsv[:, :, 1] > 50)
        blue_purple_ratio = np.sum(blue_purple_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # 判断：如果绿色比例高，且绿色通道明显高于其他通道，则为叶子
        if green_ratio > 0.3 and avg_g > avg_r + 30 and avg_g > avg_b + 30:
            return 'leaf'
        
        # 否则为果实
        return 'fruit'
        
    except Exception as e:
        print(f"  错误分类蓝莓 {img_path.name}: {e}")
        return 'fruit'

def detect_multi_fruit_type(img_path, category_name):
    """
    检测是否为多种水果（不同种类）
    只有多种水果才返回 True，同一种水果多个返回 False
    """
    try:
        # 检查图片尺寸（64x64通常是DeepFruits数据集，包含多种水果）
        img = Image.open(img_path)
        width, height = img.size
        
        # DeepFruits数据集特征：64x64像素，通常是多种水果组合
        if width == 64 and height == 64:
            return True, 'deepfruits'
        
        # 1024x1024通常是fruit-SALAD数据集，可能包含多种水果
        if width == 1024 and height == 1024:
            # 进一步分析颜色区域来判断
            img_array = np.array(img.convert('RGB'))
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 分析颜色分布，如果颜色种类很多且分布均匀，可能是多种水果
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            if unique_colors > 1000:  # 颜色种类很多
                return True, 'color_peaks'
        
        # 默认：单种水果（即使是多个）
        return False, None
        
    except Exception as e:
        print(f"  错误检测多水果类型 {img_path.name}: {e}")
        return False, None

def fix_json_annotation(json_path, category_name, split_name):
    """
    修正JSON标注文件
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        changes = []
        
        if 'images' not in data or len(data['images']) == 0:
            return changes
        
        img_info = data['images'][0]
        img_file = img_info.get('file_name', '')
        img_path = Path(json_path).parent.parent / img_file
        
        if not img_path.exists():
            return changes
        
        # 1. 修正 is_multi_fruit
        is_multi, source = detect_multi_fruit_type(img_path, category_name)
        old_multi = img_info.get('is_multi_fruit', False)
        
        if old_multi != is_multi:
            img_info['is_multi_fruit'] = is_multi
            if 'annotations' in data and len(data['annotations']) > 0:
                data['annotations'][0]['is_multi_fruit'] = is_multi
            if 'categories' in data and len(data['categories']) > 0:
                data['categories'][0]['is_multi_fruit'] = is_multi
            changes.append(f"is_multi_fruit: {old_multi} -> {is_multi}")
        
        # 2. 修正苹果图片分类
        if category_name.lower() == 'apple':
            detected_type = classify_apple_image_improved(img_path)
            old_type = img_info.get('content_type', 'fruit')
            
            if old_type != detected_type:
                img_info['content_type'] = detected_type
                img_info['subcategory'] = detected_type
                
                if 'annotations' in data and len(data['annotations']) > 0:
                    data['annotations'][0]['content_type'] = detected_type
                
                if 'categories' in data and len(data['categories']) > 0:
                    if detected_type == 'flower':
                        data['categories'][0]['name'] = 'apple_flower'
                    elif detected_type == 'leaf':
                        data['categories'][0]['name'] = 'apple_leaf'
                    else:
                        data['categories'][0]['name'] = 'apple_fruit'
                    data['categories'][0]['subcategory'] = detected_type
                
                changes.append(f"content_type: {old_type} -> {detected_type}")
        
        # 3. 修正香蕉图片风格
        elif category_name.lower() == 'banana':
            detected_style = detect_image_style_improved(img_path)
            old_style = img_info.get('image_style', 'real')
            
            if old_style != detected_style:
                img_info['image_style'] = detected_style
                if 'categories' in data and len(data['categories']) > 0:
                    data['categories'][0]['image_style'] = detected_style
                changes.append(f"image_style: {old_style} -> {detected_style}")
        
        # 4. 修正蓝莓图片分类
        elif category_name.lower() == 'blueberry':
            detected_type = classify_blueberry_image(img_path)
            old_type = img_info.get('content_type', 'fruit')
            
            # 如果当前标注为fruit但实际是leaf，需要修正
            if old_type == 'fruit' and detected_type == 'leaf':
                img_info['content_type'] = 'leaf'
                img_info['subcategory'] = 'leaf'
                
                if 'annotations' in data and len(data['annotations']) > 0:
                    data['annotations'][0]['content_type'] = 'leaf'
                
                if 'categories' in data and len(data['categories']) > 0:
                    data['categories'][0]['name'] = 'blueberry_leaf'
                    data['categories'][0]['subcategory'] = 'leaf'
                
                changes.append(f"content_type: {old_type} -> {detected_type}")
        
        # 保存更新后的JSON
        if changes:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return changes
        
    except Exception as e:
        print(f"  错误修正JSON {json_path.name}: {e}")
        return []

def process_category(category_dir, category_name, split_name):
    """处理一个类别目录"""
    json_dir = category_dir / 'json'
    
    if not json_dir.exists():
        return {'checked': 0, 'fixed': 0, 'changes': []}
    
    json_files = list(json_dir.glob('*.json'))
    
    if not json_files:
        return {'checked': 0, 'fixed': 0, 'changes': []}
    
    stats = {'checked': 0, 'fixed': 0, 'changes': []}
    
    for json_path in tqdm(json_files, desc=f"  {split_name}/{category_name}", leave=False):
        stats['checked'] += 1
        changes = fix_json_annotation(json_path, category_name, split_name)
        
        if changes:
            stats['fixed'] += 1
            stats['changes'].extend([f"{json_path.name}: {c}" for c in changes])
    
    return stats

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    
    splits = ['train', 'val', 'test']
    
    # 需要处理的类别
    categories_to_fix = {
        'apple': '修正苹果图片分类（花/叶/果实）和is_multi_fruit',
        'banana': '修正香蕉图片风格标注',
        'blueberry': '修正蓝莓图片分类（叶子不应标记为果实）',
    }
    
    # 所有类别都需要修正is_multi_fruit
    all_categories = set()
    for split in splits:
        split_dir = dataset_root / split
        if split_dir.exists():
            for cat_dir in split_dir.iterdir():
                if cat_dir.is_dir() and cat_dir.name != 'json':
                    all_categories.add(cat_dir.name)
    
    total_stats = {
        'checked': 0,
        'fixed': 0,
        'changes_by_type': {
            'is_multi_fruit': 0,
            'apple_content': 0,
            'banana_style': 0,
            'blueberry_content': 0,
        }
    }
    
    print("="*60)
    print("综合修正标注脚本")
    print("="*60)
    print("\n修正内容：")
    print("1. is_multi_fruit: 只有多种水果才为true，同一种水果多个为false")
    print("2. 苹果图片分类: 重新验证花/叶/果实")
    print("3. 香蕉图片风格: 修正真实/油画/卡通等标注")
    print("4. 蓝莓图片分类: 叶子不应标记为果实")
    print("="*60)
    
    for split in splits:
        print(f"\n处理 {split} split...")
        split_dir = dataset_root / split
        
        if not split_dir.exists():
            continue
        
        for category_name in sorted(all_categories):
            category_dir = split_dir / category_name
            
            if not category_dir.exists():
                continue
            
            # 只处理需要特殊处理的类别，但所有类别都需要检查is_multi_fruit
            if category_name in categories_to_fix:
                print(f"\n处理 {split}/{category_name} ({categories_to_fix[category_name]})...")
                stats = process_category(category_dir, category_name, split)
                
                total_stats['checked'] += stats['checked']
                total_stats['fixed'] += stats['fixed']
                
                # 统计不同类型的修正
                for change in stats['changes']:
                    if 'is_multi_fruit' in change:
                        total_stats['changes_by_type']['is_multi_fruit'] += 1
                    elif category_name == 'apple' and 'content_type' in change:
                        total_stats['changes_by_type']['apple_content'] += 1
                    elif category_name == 'banana' and 'image_style' in change:
                        total_stats['changes_by_type']['banana_style'] += 1
                    elif category_name == 'blueberry' and 'content_type' in change:
                        total_stats['changes_by_type']['blueberry_content'] += 1
                
                if stats['fixed'] > 0:
                    print(f"  修正了 {stats['fixed']} 个文件")
            else:
                # 其他类别只检查is_multi_fruit
                stats = process_category(category_dir, category_name, split)
                total_stats['checked'] += stats['checked']
                total_stats['fixed'] += stats['fixed']
                
                for change in stats['changes']:
                    if 'is_multi_fruit' in change:
                        total_stats['changes_by_type']['is_multi_fruit'] += 1
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"检查文件总数: {total_stats['checked']}")
    print(f"修正文件总数: {total_stats['fixed']}")
    print("\n修正统计：")
    print(f"  - is_multi_fruit 修正: {total_stats['changes_by_type']['is_multi_fruit']}")
    print(f"  - 苹果内容类型修正: {total_stats['changes_by_type']['apple_content']}")
    print(f"  - 香蕉风格修正: {total_stats['changes_by_type']['banana_style']}")
    print(f"  - 蓝莓内容类型修正: {total_stats['changes_by_type']['blueberry_content']}")
    print("="*60)

if __name__ == '__main__':
    main()
