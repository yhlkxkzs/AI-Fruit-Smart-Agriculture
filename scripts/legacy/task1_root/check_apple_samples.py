#!/usr/bin/env python3
"""
检查苹果图片样本，分析图片内容并对比JSON标注
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_image(img_path):
    """分析图片并返回特征"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        height, width = img.shape[:2]
        
        # 颜色分析
        avg_r = np.mean(img_rgb[:, :, 0])
        avg_g = np.mean(img_rgb[:, :, 1])
        avg_b = np.mean(img_rgb[:, :, 2])
        
        # 红色区域
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 粉色/白色区域
        pink_lower = np.array([140, 30, 150])
        pink_upper = np.array([170, 100, 255])
        pink_mask = cv2.inRange(img_hsv, pink_lower, pink_upper)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(img_hsv, white_lower, white_upper)
        pink_white_ratio = np.sum(cv2.bitwise_or(pink_mask, white_mask) > 0) / (width * height)
        
        # 绿色区域
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        # 亮度
        brightness = np.mean(img_hsv[:, :, 2])
        saturation = np.mean(img_hsv[:, :, 1])
        
        return {
            'size': (width, height),
            'avg_rgb': (avg_r, avg_g, avg_b),
            'red_ratio': red_ratio,
            'pink_white_ratio': pink_white_ratio,
            'green_ratio': green_ratio,
            'brightness': brightness,
            'saturation': saturation
        }
    except Exception as e:
        print(f"错误: {e}")
        return None

def check_samples():
    """检查样本文件"""
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    # 检查前20个文件
    json_files = sorted(list(json_dir.glob('*.json')))[:20]
    
    print("="*80)
    print("检查苹果图片样本 - 对比图片内容和JSON标注")
    print("="*80)
    print()
    
    for json_path in json_files:
        img_stem = json_path.stem
        img_files = list(apple_dir.glob(f"{img_stem}.*"))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            continue
        
        img_path = img_files[0]
        
        # 读取JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_type = data['images'][0].get('content_type', 'unknown')
        except:
            continue
        
        # 分析图片
        features = analyze_image(img_path)
        if features is None:
            continue
        
        # 简单判断
        predicted_type = 'unknown'
        if features['red_ratio'] > 0.15:
            predicted_type = 'fruit'
        elif features['pink_white_ratio'] > 0.15:
            predicted_type = 'flower'
        elif features['green_ratio'] > 0.3:
            predicted_type = 'leaf'
        else:
            if features['brightness'] > 150:
                predicted_type = 'flower'
            else:
                predicted_type = 'fruit'
        
        match = "✓" if json_type == predicted_type else "✗"
        
        print(f"{match} {img_path.name}")
        print(f"  JSON标注: {json_type}")
        print(f"  图片分析: {predicted_type}")
        print(f"  特征: 红色={features['red_ratio']:.2%}, 粉色/白色={features['pink_white_ratio']:.2%}, "
              f"绿色={features['green_ratio']:.2%}, 亮度={features['brightness']:.1f}")
        print()

if __name__ == '__main__':
    check_samples()
