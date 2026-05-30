#!/usr/bin/env python3
"""
验证和修正苹果图片标注脚本
使用更准确的图像分析方法来区分苹果花、叶子和果实
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import re

def is_binary_image(img_path):
    """检查是否为黑白照片"""
    try:
        img = Image.open(img_path)
        mode = img.mode
        if mode == '1':
            return True
        if mode in ['L', 'LA']:
            pixels = np.array(img)
            unique_values = np.unique(pixels)
            if len(unique_values) <= 2 and (0 in unique_values or 255 in unique_values):
                return True
        return False
    except Exception as e:
        print(f"  错误检查 {img_path.name}: {e}")
        return False

def classify_apple_image_advanced(img_path):
    """
    使用更先进的图像分析方法分类苹果图片
    返回: 'flower', 'leaf', 'fruit'
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
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # 圆形度（果实通常更圆）
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # 凸包分析（花朵通常形状更复杂）
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area  # 实心度（果实更实心）
            else:
                solidity = 0
        else:
            circularity = 0
            solidity = 0
        
        # 6. 纹理分析（使用拉普拉斯算子）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # 7. 颜色分布分析
        # 计算绿色像素比例（叶子）
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50)
        green_ratio = np.sum(green_mask) / (height * width)
        
        # 计算红色像素比例（果实）
        red_mask1 = (hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)
        red_mask2 = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        red_mask = red_mask1 & red_mask2
        red_ratio = np.sum(red_mask) / (height * width)
        
        # 计算白色/粉色像素比例（花朵）
        white_pink_mask = (s_mean < 0.3) & (v_mean > 0.7)  # 低饱和度，高亮度
        white_pink_ratio = np.sum(white_pink_mask) / (height * width) if isinstance(white_pink_mask, np.ndarray) else 0
        
        # 8. 综合判断规则
        
        # 叶子判断：绿色为主，绿色像素比例高
        if green_ratio > 0.3 and avg_g > avg_r + 30 and avg_g > avg_b + 30:
            return 'leaf'
        
        # 花朵判断（多个条件）
        flower_score = 0
        
        # 条件1: 整体亮度高，饱和度低（白色/粉色花朵）
        if avg_brightness > 180 and s_mean < 0.4:
            flower_score += 2
        
        # 条件2: 红色和绿色通道都较高（粉色花朵）
        if avg_r > 150 and avg_g > 150 and avg_b > 120:
            flower_score += 2
        
        # 条件3: 边缘密度高（花朵边缘复杂）
        if edge_density > 0.08:
            flower_score += 1
        
        # 条件4: 圆形度低，实心度低（花朵形状不规则）
        if circularity < 0.6 and solidity < 0.85:
            flower_score += 1
        
        # 条件5: 纹理方差中等（花朵有细节但不复杂）
        if 100 < texture_variance < 800:
            flower_score += 1
        
        # 条件6: 白色/粉色像素比例高
        if white_pink_ratio > 0.2:
            flower_score += 1
        
        # 如果花朵得分高，判定为花朵
        if flower_score >= 4:
            return 'flower'
        
        # 果实判断：红色为主，圆形度高，实心度高
        fruit_score = 0
        
        # 条件1: 红色像素比例高
        if red_ratio > 0.2:
            fruit_score += 2
        
        # 条件2: 红色通道明显高于其他通道
        if avg_r > avg_g + 20 and avg_r > avg_b + 20:
            fruit_score += 2
        
        # 条件3: 圆形度高（果实通常更圆）
        if circularity > 0.6:
            fruit_score += 1
        
        # 条件4: 实心度高（果实更实心）
        if solidity > 0.85:
            fruit_score += 1
        
        # 条件5: 绿色通道高（绿色苹果）
        if avg_g > avg_r + 20 and avg_g > avg_b + 20 and green_ratio > 0.2:
            fruit_score += 2
        
        # 如果果实得分高，判定为果实
        if fruit_score >= 3:
            return 'fruit'
        
        # 默认判断：根据主要特征
        if flower_score > fruit_score:
            return 'flower'
        elif green_ratio > 0.2:
            return 'leaf'
        else:
            return 'fruit'
            
    except Exception as e:
        print(f"  错误分类 {img_path.name}: {e}")
        return 'fruit'  # 默认返回果实

def update_json_annotation(json_path, content_type, img_path):
    """更新JSON标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新images字段
        if 'images' in data and len(data['images']) > 0:
            data['images'][0]['content_type'] = content_type
            data['images'][0]['subcategory'] = content_type
            
            # 更新category名称
            if content_type == 'flower':
                category_name = 'apple_flower'
            elif content_type == 'leaf':
                category_name = 'apple_leaf'
            else:
                category_name = 'apple_fruit'
        
        # 更新annotations字段
        if 'annotations' in data and len(data['annotations']) > 0:
            data['annotations'][0]['content_type'] = content_type
        
        # 更新categories字段
        if 'categories' in data and len(data['categories']) > 0:
            data['categories'][0]['name'] = category_name
            data['categories'][0]['subcategory'] = content_type
        
        # 保存更新后的JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"  错误更新JSON {json_path.name}: {e}")
        return False

def process_apple_directory(apple_dir, split_name):
    """处理一个apple目录"""
    print(f"\n处理 {split_name}/apple...")
    
    image_dir = apple_dir
    json_dir = apple_dir / 'json'
    
    if not json_dir.exists():
        print(f"  跳过：JSON目录不存在")
        return {'deleted': 0, 'updated': 0, 'checked': 0}
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"  跳过：没有找到图片文件")
        return {'deleted': 0, 'updated': 0, 'checked': 0}
    
    stats = {'deleted': 0, 'updated': 0, 'checked': 0, 'changed': 0}
    
    for img_path in tqdm(image_files, desc=f"  {split_name}/apple"):
        stats['checked'] += 1
        
        # 1. 检查是否为黑白照片，如果是则删除
        if is_binary_image(img_path):
            try:
                img_path.unlink()
                # 删除对应的JSON文件
                json_path = json_dir / f"{img_path.stem}.json"
                if json_path.exists():
                    json_path.unlink()
                stats['deleted'] += 1
                continue
            except Exception as e:
                print(f"  错误删除 {img_path.name}: {e}")
                continue
        
        # 2. 分类图片
        detected_type = classify_apple_image_advanced(img_path)
        
        # 3. 检查并更新JSON文件
        json_path = json_dir / f"{img_path.stem}.json"
        if json_path.exists():
            # 读取当前标注
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                current_type = None
                if 'images' in data and len(data['images']) > 0:
                    current_type = data['images'][0].get('content_type')
                
                # 如果标注不一致，更新
                if current_type != detected_type:
                    if update_json_annotation(json_path, detected_type, img_path):
                        stats['updated'] += 1
                        stats['changed'] += 1
                        print(f"\n  修正: {img_path.name} {current_type} -> {detected_type}")
                else:
                    stats['updated'] += 1  # 标注正确，也算更新过
            except Exception as e:
                print(f"  错误读取JSON {json_path.name}: {e}")
        else:
            # JSON文件不存在，创建新的
            print(f"  警告: JSON文件不存在 {json_path.name}")
    
    return stats

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    
    splits = ['train', 'val', 'test']
    total_stats = {'deleted': 0, 'updated': 0, 'checked': 0, 'changed': 0}
    
    for split in splits:
        apple_dir = dataset_root / split / 'apple'
        if apple_dir.exists():
            stats = process_apple_directory(apple_dir, split)
            for key in total_stats:
                total_stats[key] += stats[key]
        else:
            print(f"\n跳过 {split}/apple: 目录不存在")
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"检查图片总数: {total_stats['checked']}")
    print(f"删除黑白照片: {total_stats['deleted']}")
    print(f"更新JSON文件: {total_stats['updated']}")
    print(f"修正标注错误: {total_stats['changed']}")
    print("="*60)

if __name__ == '__main__':
    main()
