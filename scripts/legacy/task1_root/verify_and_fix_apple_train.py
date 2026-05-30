#!/usr/bin/env python3
"""
系统性地验证和修正train/apple文件夹中的标注问题
重点修正：将苹果果实误标为苹果花的问题
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

def detect_apple_content_type(img_path):
    """
    更准确地检测苹果图片的内容类型（flower/leaf/fruit）
    使用多种图像特征进行综合判断
    """
    try:
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 获取图片尺寸
        height, width = img.shape[:2]
        
        # 1. 颜色分析
        # 计算平均RGB值
        avg_r = np.mean(img_rgb[:, :, 0])
        avg_g = np.mean(img_rgb[:, :, 1])
        avg_b = np.mean(img_rgb[:, :, 2])
        
        # 计算平均HSV值
        avg_h = np.mean(img_hsv[:, :, 0])
        avg_s = np.mean(img_hsv[:, :, 1])
        avg_v = np.mean(img_hsv[:, :, 2])
        
        # 2. 红色/粉色区域检测（花通常是粉色/白色，果实通常是红色/绿色）
        # 红色范围（HSV）
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 粉色/白色区域检测（花）
        pink_lower = np.array([140, 30, 150])
        pink_upper = np.array([170, 100, 255])
        pink_mask = cv2.inRange(img_hsv, pink_lower, pink_upper)
        pink_ratio = np.sum(pink_mask > 0) / (width * height)
        
        # 绿色区域检测（叶子/未成熟果实）
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        # 3. 亮度分析
        brightness = avg_v
        
        # 4. 饱和度分析
        saturation = avg_s
        
        # 5. 边缘检测（花通常有更多细节边缘）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # 6. 纹理分析（使用LBP或简单的方差）
        texture_variance = np.var(gray)
        
        # 7. 文件名关键词
        filename = img_path.name.lower()
        has_flower_keyword = any(kw in filename for kw in ['flower', 'blossom', 'bloom', '花'])
        has_fruit_keyword = any(kw in filename for kw in ['fruit', 'apple', '果实', '果'])
        
        # 综合判断逻辑
        score_flower = 0
        score_fruit = 0
        score_leaf = 0
        
        # 花的特征：
        # - 粉色/白色区域多
        # - 通常较亮
        # - 边缘密度高（细节多）
        # - 饱和度中等
        if pink_ratio > 0.15 or (pink_ratio > 0.08 and brightness > 150):
            score_flower += 3
        if brightness > 140 and saturation < 100:
            score_flower += 2
        if edge_density > 0.15:
            score_flower += 1
        if has_flower_keyword:
            score_flower += 2
        
        # 果实的特征：
        # - 红色区域多（成熟果实）
        # - 或绿色区域多（未成熟果实）
        # - 通常颜色较饱和
        # - 形状较圆润（边缘密度较低）
        if red_ratio > 0.2:
            score_fruit += 4
        elif red_ratio > 0.1:
            score_fruit += 2
        if green_ratio > 0.3 and red_ratio < 0.1:
            score_fruit += 2  # 未成熟的绿色苹果
        if saturation > 80 and red_ratio > 0.05:
            score_fruit += 2
        if edge_density < 0.12 and red_ratio > 0.1:
            score_fruit += 1  # 圆润的果实
        if has_fruit_keyword and not has_flower_keyword:
            score_fruit += 2
        
        # 叶子的特征：
        # - 绿色区域多
        # - 饱和度中等
        if green_ratio > 0.4 and red_ratio < 0.05 and pink_ratio < 0.05:
            score_leaf += 4
        if green_ratio > 0.3 and saturation > 60:
            score_leaf += 2
        
        # 特殊情况：如果红色区域很少，但粉色区域也很少，可能是叶子
        if red_ratio < 0.05 and pink_ratio < 0.05 and green_ratio > 0.2:
            score_leaf += 1
        
        # 决定最终类型
        if score_fruit > score_flower and score_fruit > score_leaf:
            return 'fruit'
        elif score_leaf > score_flower and score_leaf > score_fruit:
            return 'leaf'
        else:
            return 'flower'  # 默认或平局时返回flower
        
    except Exception as e:
        print(f"  错误分析图片 {img_path.name}: {e}")
        return None

def fix_json_annotation(json_path, img_path, detected_content_type):
    """
    修正JSON文件的标注
    确保images、annotations、categories字段的一致性
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False
        
        img_info = data['images'][0]
        current_content_type = img_info.get('content_type', '')
        current_subcategory = img_info.get('subcategory', '')
        current_category_name = data.get('categories', [{}])[0].get('name', '')
        
        # 如果检测到的类型与当前类型不同，需要修正
        if detected_content_type and detected_content_type != current_content_type:
            # 更新images字段
            img_info['content_type'] = detected_content_type
            img_info['subcategory'] = detected_content_type
            
            # 更新annotations字段
            if 'annotations' in data and len(data['annotations']) > 0:
                data['annotations'][0]['content_type'] = detected_content_type
            
            # 更新categories字段
            if 'categories' in data and len(data['categories']) > 0:
                category = data['categories'][0]
                category['subcategory'] = detected_content_type
                
                # 更新category name
                if detected_content_type == 'flower':
                    category['name'] = 'apple_flower'
                elif detected_content_type == 'fruit':
                    category['name'] = 'apple_fruit'
                elif detected_content_type == 'leaf':
                    category['name'] = 'apple_leaf'
            
            # 保存
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"  错误修正 {json_path.name}: {e}")
        return False

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    if not json_dir.exists():
        print(f"错误：{json_dir} 不存在")
        return
    
    print("="*70)
    print("系统性验证和修正 train/apple 文件夹中的标注问题")
    print("="*70)
    print(f"JSON文件目录: {json_dir}")
    print()
    
    json_files = sorted(list(json_dir.glob('*.json')))
    total_files = len(json_files)
    
    print(f"找到 {total_files} 个JSON文件")
    print("开始验证和修正...")
    print()
    
    fixed_count = 0
    error_count = 0
    unchanged_count = 0
    
    # 统计信息
    stats = {
        'flower_to_fruit': 0,
        'flower_to_leaf': 0,
        'fruit_to_flower': 0,
        'fruit_to_leaf': 0,
        'leaf_to_flower': 0,
        'leaf_to_fruit': 0,
    }
    
    for json_path in tqdm(json_files, desc="处理进度"):
        # 找到对应的图片文件
        img_stem = json_path.stem
        img_files = list(apple_dir.glob(f"{img_stem}.*"))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            error_count += 1
            continue
        
        img_path = img_files[0]
        
        # 检测内容类型
        detected_type = detect_apple_content_type(img_path)
        
        if detected_type is None:
            error_count += 1
            continue
        
        # 读取当前标注
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            current_type = data['images'][0].get('content_type', '')
        except:
            error_count += 1
            continue
        
        # 如果类型不同，修正
        if detected_type != current_type:
            # 记录修正类型
            change_key = f"{current_type}_to_{detected_type}"
            if change_key in stats:
                stats[change_key] += 1
            
            if fix_json_annotation(json_path, img_path, detected_type):
                fixed_count += 1
                if fixed_count <= 10:  # 显示前10个修正
                    print(f"  修正: {json_path.name} ({current_type} -> {detected_type})")
            else:
                error_count += 1
        else:
            unchanged_count += 1
    
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
